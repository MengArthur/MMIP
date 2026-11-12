"""
Experiment C: constrained decoding with a finite-state machine (lecture slides 55-57, Willard & Louf 2023).

The routing JSON is described as a sequence of elements, which together form a byte-level automaton:
  {"status":<enum>,"department":<enum>,"urgency":<enum>,"intent":<string>,"order_id":null|"SD########",
   "product":null|<string>,"missing_info":[<enum>,...]}
At every decoding step a LogitsProcessor computes which vocabulary tokens keep the automaton in a valid state
and sets the logits of all other tokens to -inf ("mask"), so the output is valid JSON that satisfies the schema
by construction - enums can only be allowed values, the order id can only match SD + 8 digits, strings are
length-limited and contain no quotes / escapes, and the only token allowed after "}" is end-of-sequence.

Tokens are handled as raw bytes (Qwen uses byte-level BPE, so one Chinese character may be split over several
tokens). Allowed-token sets are computed by walking the sorted vocabulary as a trie together with the automaton
and are cached per automaton state; inside free strings (where almost every token is allowed) precomputed masks
are used instead, which also keep the string valid UTF-8.
"""

import bisect
import time

import torch

from .schema import DEPARTMENTS, MAX_INTENT_CHARS, MAX_PRODUCT_CHARS, MISSING, STATUS, URGENCY

MAX_INTENT_BYTES, MAX_PRODUCT_BYTES = MAX_INTENT_CHARS, MAX_PRODUCT_CHARS   # bytes <= characters, so the char limits hold

QUOTE, BSLASH = 0x22, 0x5C


# ----------------------------------------------------------------------------- automaton elements
class Lit:
    def __init__(self, s):
        self.s = s.encode()

    def start(self):
        return 0

    def step(self, pos, b):
        return pos + 1 if pos < len(self.s) and self.s[pos] == b else None

    def done(self, pos):
        return pos == len(self.s)


class Enum:
    """One of several JSON strings (given without quotes)."""

    def __init__(self, options):
        self.opts = tuple(f'"{o}"'.encode() for o in options)

    def start(self):
        return self.opts

    def step(self, rem, b):
        nxt = tuple(r[1:] for r in rem if r and r[0] == b)
        return nxt or None

    def done(self, rem):
        return b'' in rem


class Str:
    """A JSON string without escapes. sub = (phase, n_bytes, pending UTF-8 continuation bytes);
    phase 0: before the opening quote, 1: inside, 2: closed. The upper length cap is enforced by the masks,
    the minimum length (non-empty strings) here."""

    def __init__(self, max_bytes, min_bytes=1):
        self.max, self.min = max_bytes, min_bytes

    def start(self):
        return (0, 0, 0)

    def step(self, sub, b):
        phase, n, c = sub
        if phase == 0:
            return (1, 0, 0) if b == QUOTE else None
        if phase == 2:
            return None
        if b == QUOTE:
            return (2, n, 0) if c == 0 and n >= self.min else None
        if b < 0x20 or b == BSLASH or b == 0x7F:
            return None
        if 0x80 <= b < 0xC0:                       # continuation byte
            return (1, n + 1, c - 1) if c > 0 else None
        if c > 0:
            return None
        if b >= 0xF8:
            return None
        return (1, n + 1, 3 if b >= 0xF0 else 2 if b >= 0xE0 else 1 if b >= 0xC0 else 0)

    def done(self, sub):
        return sub[0] == 2


class Pattern:
    """Fixed-length string of byte classes, e.g. "SD" followed by 8 digits (with the quotes)."""

    def __init__(self, classes):
        self.cls = classes

    def start(self):
        return 0

    def step(self, pos, b):
        return pos + 1 if pos < len(self.cls) and b in self.cls[pos] else None

    def done(self, pos):
        return pos == len(self.cls)


class NullOr:
    def __init__(self, inner):
        self.inner = inner

    def start(self):
        return ('?',)

    def step(self, sub, b):
        if sub[0] == '?':
            if b == ord('n'):
                return ('null', 1)
            s = self.inner.step(self.inner.start(), b)
            return ('in', s) if s is not None else None
        if sub[0] == 'null':
            return ('null', sub[1] + 1) if sub[1] < 4 and b == b'null'[sub[1]] else None
        s = self.inner.step(sub[1], b)
        return ('in', s) if s is not None else None

    def done(self, sub):
        return (sub[0] == 'null' and sub[1] == 4) or (sub[0] == 'in' and self.inner.done(sub[1]))


class EnumArray:
    """[] or ["a","b",...] with unique items from `options`.
    sub: ('open',) | ('first'|'next'|'after'|'closed', used) | ('opt', used, ((option, pos), ...))"""

    def __init__(self, options):
        self.opts = tuple(f'"{o}"'.encode() for o in options)

    def start(self):
        return ('open', ())

    def step(self, sub, b):
        kind = sub[0]
        if kind == 'open':
            return ('first', ()) if b == ord('[') else None
        used = sub[1]
        if kind in ('first', 'next'):
            if kind == 'first' and b == ord(']'):
                return ('closed', used)
            cands = tuple((o, 1) for o in self.opts if o not in used and o[0] == b)
            return ('opt', used, cands) if cands else None
        if kind == 'opt':
            cands = tuple((o, p + 1) for o, p in sub[2] if p < len(o) and o[p] == b)
            if cands:
                return ('opt', used, cands)
            finished = [o for o, p in sub[2] if p == len(o)]
            return self.step(('after', used + (finished[0],)), b) if finished else None
        if kind == 'after':
            if b == ord(',') and len(used) < len(self.opts):
                return ('next', used)
            return ('closed', used) if b == ord(']') else None
        return None

    def done(self, sub):
        return sub[0] == 'closed'


def route_schema():
    digits = frozenset(b'0123456789')
    oid = Pattern([frozenset([QUOTE]), frozenset(b'S'), frozenset(b'D')] + [digits] * 8 + [frozenset([QUOTE])])
    return [Lit('{"status":'), Enum(STATUS), Lit(',"department":'), Enum(DEPARTMENTS), Lit(',"urgency":'), Enum(URGENCY),
            Lit(',"intent":'), Str(MAX_INTENT_BYTES), Lit(',"order_id":'), NullOr(oid), Lit(',"product":'), NullOr(Str(MAX_PRODUCT_BYTES)),
            Lit(',"missing_info":'), EnumArray(MISSING), Lit('}')]


class Automaton:
    def __init__(self, elems):
        self.E = elems

    def start(self):
        return (0, self.E[0].start())

    def step(self, state, b):
        i, sub = state
        while i < len(self.E):
            ns = self.E[i].step(sub, b)
            if ns is not None:
                return (i, ns)
            if not self.E[i].done(sub):
                return None
            i += 1
            sub = self.E[i].start() if i < len(self.E) else None
        return None

    def feed(self, state, data):
        for b in data:
            state = self.step(state, b)
            if state is None:
                return None
        return state

    def final(self, state):
        i, sub = state
        return i == len(self.E) - 1 and self.E[i].done(sub)

    def in_string(self, state):
        """(element index, Str element, sub) if the automaton is inside a free string, else None."""
        i, sub = state
        el = self.E[i]
        if isinstance(el, NullOr) and sub[0] == 'in':
            el, sub = el.inner, sub[1]
        if isinstance(el, Str) and sub[0] == 1:
            return i, el, sub
        return None


# ----------------------------------------------------------------------------- vocabulary
def _bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(tok):
    """id -> raw bytes for every normal token; special / added tokens map to None."""
    dec = _bytes_to_unicode()
    special = set(getattr(tok, 'added_tokens_decoder', {}) or {}) | set(tok.all_special_ids)
    out = []
    for i, s in enumerate(tok.convert_ids_to_tokens(list(range(len(tok))))):
        if i in special or s is None:
            out.append(None)
            continue
        try:
            out.append(bytes(dec[ch] for ch in s))
        except KeyError:            # not a byte-level token (should not happen for Qwen)
            out.append(None)
    return out


class Vocab:
    def __init__(self, tok, vocab_size, device):
        t0 = time.perf_counter()
        tb = token_bytes(tok)
        groups = {}
        for i, b in enumerate(tb):
            if b:
                groups.setdefault(b, []).append(i)
        self.keys = sorted(groups)
        self.ids = [groups[k] for k in self.keys]
        self.tb = tb
        self.V = vocab_size
        self.device = device
        # string-content tokens, grouped by how they fit the UTF-8 state (pending continuation bytes)
        str_ok = {c: [] for c in range(4)}
        cont_only = {c: [] for c in range(4)}
        self.quote_tokens = []
        for i, b in enumerate(tb):
            if not b:
                continue
            if QUOTE in b:          # may close the string; checked against the automaton per state (cached)
                q = b.index(QUOTE)
                if all(0x20 <= x and x not in (BSLASH, 0x7F) for x in b[:q]):
                    self.quote_tokens.append(i)
                continue
            if any(x < 0x20 or x in (BSLASH, 0x7F) for x in b):
                continue
            k = next((j for j, x in enumerate(b) if not 0x80 <= x < 0xC0), len(b))   # leading continuation bytes
            if k == len(b):         # only continuation bytes: valid while at least k of them are pending
                for c in range(k, 4):
                    str_ok[c].append(i)
                    cont_only[c].append(i)
            elif k <= 3 and self._wellformed(b[k:]):
                str_ok[k].append(i)
        # byte length of every token, and the number of bytes before the first quote (string-length cap)
        self.nbytes = torch.tensor([len(b) if b else 0 for b in tb] + [0] * max(0, vocab_size - len(tb)), device=device)
        self.qpre = torch.tensor([b.index(QUOTE) if b and QUOTE in b else 0 for b in tb] + [0] * max(0, vocab_size - len(tb)), device=device)
        mk = lambda ids: torch.tensor(sorted(set(ids)), dtype=torch.long, device=device)
        self.str_ok = {c: mk(v) for c, v in str_ok.items()}
        self.cont_only = {c: mk(v) for c, v in cont_only.items()}
        self.build_seconds = time.perf_counter() - t0

    @staticmethod
    def _wellformed(b):
        """UTF-8 sequence that may end in the middle of a character, but has no stray continuation bytes."""
        need = 0
        for x in b:
            if 0x80 <= x < 0xC0:
                if need == 0:
                    return False
                need -= 1
            else:
                if need or x >= 0xF8:
                    return False
                need = 3 if x >= 0xF0 else 2 if x >= 0xE0 else 1 if x >= 0xC0 else 0
        return True


# ----------------------------------------------------------------------------- logits processor
class FSMLogitsProcessor:
    """Batch-aware LogitsProcessor for HF generate(). Create one per generate() call."""

    def __init__(self, vocab, automaton, prompt_len, eos_ids, cache):
        self.v, self.A = vocab, automaton
        self.prompt_len = prompt_len
        self.eos = torch.tensor(sorted(set(eos_ids)), dtype=torch.long, device=vocab.device)
        self.cache = cache                       # shared across calls: automaton state -> allowed ids
        self.states, self.done = None, None
        self.mask_seconds = 0.0
        self.violations = 0

    def _allowed(self, state):
        if self.A.final(state):
            return self.eos
        s = self.A.in_string(state)
        if s is not None:
            i, el, (phase, n, c) = s
            key = ('q', i, c, n >= el.min)
            if key not in self.cache:            # tokens that contain the closing quote (and maybe more JSON)
                ok = [t for t in self.v.quote_tokens if self.A.feed(state, self.v.tb[t]) is not None]
                self.cache[key] = torch.tensor(ok, dtype=torch.long, device=self.v.device)
            room = el.max - n                    # tokens must not push the string past its byte cap
            base = self.v.str_ok[c] if room > 0 else self.v.cont_only[c]
            base = base[self.v.nbytes[base] <= max(room, c)]
            quote = self.cache[key]
            return torch.cat([base, quote[self.v.qpre[quote] <= max(room, 0)]])
        if state not in self.cache:
            out = []
            self._dfs(0, 0, len(self.v.keys), state, out)
            self.cache[state] = torch.tensor(out, dtype=torch.long, device=self.v.device)
        return self.cache[state]

    def _dfs(self, depth, lo, hi, state, out):
        keys = self.v.keys
        idx = lo
        if len(keys[lo]) == depth:
            if depth > 0:
                out.extend(self.v.ids[lo])
            idx = lo + 1
        while idx < hi:
            b = keys[idx][depth]
            j = bisect.bisect_left(keys, keys[idx][:depth] + bytes([b + 1]), idx, hi) if b < 255 else hi
            ns = self.A.step(state, b)
            if ns is not None:
                self._dfs(depth + 1, idx, j, ns, out)
            idx = j

    def __call__(self, input_ids, scores):
        t0 = time.perf_counter()
        B = input_ids.shape[0]
        if self.states is None:
            self.states = [self.A.start() for _ in range(B)]
            self.done = [False] * B
        elif input_ids.shape[1] > self.prompt_len:
            for r in range(B):
                if self.done[r]:
                    continue
                t = int(input_ids[r, -1])
                if t in self.eos.tolist():
                    self.done[r] = True
                    continue
                ns = self.A.feed(self.states[r], self.v.tb[t] or b'')
                if ns is None:                     # cannot happen when the mask was applied; kept as a guard
                    self.violations += 1
                    self.done[r] = True
                else:
                    self.states[r] = ns
        out = torch.full_like(scores, float('-inf'))
        for r in range(B):
            allowed = self.eos if self.done[r] else self._allowed(self.states[r])
            allowed = allowed[allowed < scores.shape[1]]
            if allowed.numel() == 0:               # dead end (cannot happen with this schema); end the sequence
                self.violations += 1
                allowed = self.eos
            out[r, allowed] = scores[r, allowed]
        self.mask_seconds += time.perf_counter() - t0
        return out


class ConstrainedDecoder:
    """Holds the vocabulary tables and the allowed-token cache for one tokenizer."""

    def __init__(self, tok, vocab_size, device, eos_ids):
        self.vocab = Vocab(tok, vocab_size, device)
        self.automaton = Automaton(route_schema())
        self.eos_ids = eos_ids
        self.cache = {}

    def processor(self, prompt_len):
        return FSMLogitsProcessor(self.vocab, self.automaton, prompt_len, self.eos_ids, self.cache)
