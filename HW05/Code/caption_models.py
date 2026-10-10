"""
Caption decoders, decoding and caption metrics -- copied from Week 4 (HW04/Code/caption_models.py)
so that the comparison changes ONLY the image encoder. The decoders now take the encoder's
feature size as an argument (ResNet-50: 49 x 2048 in Week 4; CLIP ViT-B/32: 50 x 768 here).

  - LSTM decoder        : the mean of the encoder tokens initialises the LSTM state.
  - Transformer decoder : masked self-attention over the words so far + cross-attention
                          over all encoder tokens.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DEVICE

SPECIALS = ('<pad>', '<unk>', '<bos>', '<eos>')
PAD, UNK, BOS, EOS = 0, 1, 2, 3


# ----------------------------------------------------------------------------
# Decoders
# ----------------------------------------------------------------------------
class LSTMCaptioner(nn.Module):
    def __init__(self, vocab_size, feat_dim, embed=512, hidden=512, dropout=0.3):
        super().__init__()
        self.init_h = nn.Linear(feat_dim, hidden)
        self.init_c = nn.Linear(feat_dim, hidden)
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=PAD)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def init_state(self, feats):
        g = feats.mean(1)
        return (torch.tanh(self.init_h(g)).unsqueeze(0).contiguous(),
                torch.tanh(self.init_c(g)).unsqueeze(0).contiguous())

    def forward(self, feats, tokens):
        o, _ = self.lstm(self.drop(self.embed(tokens)), self.init_state(feats))
        return self.out(self.drop(o))

    def step(self, last, state):
        o, state = self.lstm(self.embed(last).unsqueeze(1), state)
        return F.log_softmax(self.out(o[:, 0]), -1), state

    @staticmethod
    def reorder_state(state, idx):
        return state[0][:, idx], state[1][:, idx]


class TransformerCaptioner(nn.Module):
    def __init__(self, vocab_size, feat_dim, n_regions, d=512, heads=8, layers=3, ff=2048, dropout=0.1, max_len=64):
        super().__init__()
        self.d = d
        self.mem_proj = nn.Linear(feat_dim, d)
        self.mem_pos = nn.Parameter(torch.zeros(1, n_regions, d))
        self.embed = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, vocab_size)
        nn.init.normal_(self.mem_pos, std=0.02)

    def encode(self, feats):
        return self.mem_proj(feats) + self.mem_pos

    def decode(self, tokens, memory):
        T_ = tokens.size(1)
        x = self.embed(tokens) * math.sqrt(self.d) + self.pos(torch.arange(T_, device=tokens.device))
        causal = torch.triu(torch.ones(T_, T_, dtype=torch.bool, device=tokens.device), 1)
        h = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=(tokens == PAD))
        return self.out(h)

    def forward(self, feats, tokens):
        return self.decode(tokens, self.encode(feats))

    def init_state(self, feats):
        memory = self.encode(feats)
        return memory, torch.empty(feats.size(0), 0, dtype=torch.long, device=feats.device)

    def step(self, last, state):
        memory, tokens = state
        tokens = torch.cat([tokens, last[:, None]], 1)
        return F.log_softmax(self.decode(tokens, memory)[:, -1], -1), (memory, tokens)

    @staticmethod
    def reorder_state(state, idx):
        return state[0][idx], state[1][idx]


def _mask_specials(logp):
    # never emit padding / <bos> / <unk> while decoding
    logp[:, [PAD, BOS, UNK]] = -float('inf')
    return logp


@torch.no_grad()
def greedy_decode(model, feats, max_len=30, batch_size=250):
    """feats: (N, 49, 2048) on any device -> list of token-id lists (no <bos>/<eos>)."""
    model.eval()
    out = []
    for i in range(0, len(feats), batch_size):
        f = feats[i:i + batch_size].to(DEVICE).float()
        state = model.init_state(f)
        last = torch.full((f.size(0),), BOS, dtype=torch.long, device=DEVICE)
        steps, done = [], torch.zeros(f.size(0), dtype=torch.bool, device=DEVICE)
        for _ in range(max_len):
            logp, state = model.step(last, state)
            last = _mask_specials(logp).argmax(1)
            steps.append(last)
            done |= last == EOS
            if done.all():
                break
        seqs = torch.stack(steps, 1).tolist()
        out += [s[:s.index(EOS)] if EOS in s else s for s in seqs]
    return out


@torch.no_grad()
def beam_search(model, feat, beam=3, max_len=30):
    """feat: (1, 49, 2048). Length-normalised beam search; returns the best token-id list."""
    model.eval()
    state = model.init_state(feat.to(DEVICE).float())
    seqs, scores = [[]], torch.zeros(1, device=DEVICE)
    last = torch.tensor([BOS], device=DEVICE)
    finished = []
    for _ in range(max_len):
        logp, state = model.step(last, state)
        total = scores[:, None] + _mask_specials(logp)
        vocab = total.size(1)
        top_s, top_i = total.view(-1).topk(min(2 * beam, total.numel()))
        keep_seqs, keep_scores, keep_src, keep_last = [], [], [], []
        for s, ix in zip(top_s.tolist(), top_i.tolist()):
            src, w = divmod(ix, vocab)
            if w == EOS:
                finished.append((s / max(1, len(seqs[src])), seqs[src]))
            else:
                keep_seqs.append(seqs[src] + [w]); keep_scores.append(s); keep_src.append(src); keep_last.append(w)
            if len(keep_seqs) == beam:
                break
        if len(finished) >= beam or not keep_seqs:
            break
        state = model.reorder_state(state, torch.tensor(keep_src, device=DEVICE))
        seqs, scores = keep_seqs, torch.tensor(keep_scores, device=DEVICE)
        last = torch.tensor(keep_last, device=DEVICE)
    if not finished:
        finished = [(s / max(1, len(q)), q) for s, q in zip(scores.tolist(), seqs)]
    return max(finished, key=lambda t: t[0])[1]


# ----------------------------------------------------------------------------
# Caption metrics
# ----------------------------------------------------------------------------
def bleu_scores(refs_tok, hyps_tok):
    """Corpus BLEU-1..4 against all 5 references (Papineni et al., 2002)."""
    from nltk.translate.bleu_score import corpus_bleu
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (1 / 3, 1 / 3, 1 / 3, 0), (0.25, 0.25, 0.25, 0.25)]
    return {f'BLEU-{n + 1}': float(corpus_bleu(refs_tok, hyps_tok, weights=w)) for n, w in enumerate(weights)}


class BertScore:
    """BERTScore F1 (Zhang et al., 2020) with RoBERTa-large, rescaled with the official baseline;
    each candidate is scored against all of its references and the best match is kept."""

    def __init__(self):
        from bert_score import BERTScorer
        self.scorer = BERTScorer(lang='en', rescale_with_baseline=True, device=str(DEVICE), batch_size=64)

    def __call__(self, cands, refs):
        cands = [c if c.strip() else 'empty' for c in cands]
        _, _, f1 = self.scorer.score(cands, refs)
        return f1.numpy()
