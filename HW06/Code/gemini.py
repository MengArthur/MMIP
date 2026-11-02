"""
Commercial-LLM judge (Gemini, google-genai SDK) used in Part A and Part B.

The API key is read from the environment / Colab Secrets (GEMINI_API_KEY), never stored in code.
Every response is cached in a JSON file keyed by the request, so re-running the notebook does not
call the API again and the reported numbers stay fixed. Without a key the judge is skipped and
only the rule-based metrics are reported.
"""

import io
import json
import os
import re
import time

DEFAULT_MODEL = 'gemini-3.5-flash-lite'   # the model that stayed within the free quota in HW04
_EXCLUDE = ('image', 'tts', 'live', 'audio', 'embedding', 'exp', 'preview', 'thinking', 'learnlm', 'gemma',
            'robotics', 'computer', 'native')


def get_api_key():
    """Environment variable first; otherwise Colab Secrets (copied into the environment, because
    userdata.get() can time out when called again late in a long run)."""
    key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if key:
        return key
    try:
        from google.colab import userdata
        key = userdata.get('GEMINI_API_KEY')
    except ImportError:
        return None
    except Exception as e:
        print(f'[gemini] could not read GEMINI_API_KEY from Colab Secrets: {type(e).__name__}: {e}')
        return None
    if key:
        os.environ['GEMINI_API_KEY'] = key
    return key


def pick_model(client, preferred=None):
    """`preferred` if this key can access it; otherwise the newest stable Gemini Flash(-Lite) model."""
    names = []
    for m in client.models.list():
        actions = getattr(m, 'supported_actions', None)
        if actions is None or 'generateContent' in actions:
            names.append(m.name.split('/')[-1])
    if preferred and preferred in names:
        return preferred
    flash = [n for n in names if n.startswith('gemini') and 'flash' in n and not any(b in n for b in _EXCLUDE)]
    if not flash:
        raise RuntimeError(f'No Gemini Flash model available for this API key; available: {names}')
    version = lambda n: tuple(float(v) for v in re.findall(r'\d+(?:\.\d+)?', n)[:1] or [0])
    return max(flash, key=lambda n: (version(n), 'lite' in n, -len(n)))


def _jpeg(image):
    buf = io.BytesIO()
    image.convert('RGB').save(buf, format='JPEG', quality=90)
    return buf.getvalue()


class GeminiJudge:
    def __init__(self, cache_path, model=DEFAULT_MODEL, rpm=10):
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding='utf-8') as f:
                self.cache = json.load(f)
        self.client, self.model = None, model
        key = get_api_key()
        if key:
            from google import genai
            self.client = genai.Client(api_key=key)
            self.model = pick_model(self.client, model)
        self.min_interval = 60.0 / rpm
        self._last = 0.0

    @property
    def available(self):
        return self.client is not None

    def _save(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=1)

    def _call(self, prompt, image=None, retries=6):
        from google.genai import types
        cfg = types.GenerateContentConfig(temperature=0.0, response_mime_type='application/json')
        contents = ([types.Part.from_bytes(data=_jpeg(image), mime_type='image/jpeg')] if image is not None else []) + [prompt]
        for attempt in range(retries):
            wait = self.min_interval - (time.time() - self._last)     # stay under the free-tier rate limit
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
                return (resp.text or '').strip()
            except Exception as e:
                if attempt == retries - 1:
                    raise
                backoff = min(60, 5 * 2 ** attempt)
                print(f'  [gemini] {type(e).__name__}: {str(e)[:120]} -> retry in {backoff}s')
                time.sleep(backoff)

    def judge(self, key, prompt, image=None):
        """Returns the parsed JSON list (cached by key), or None if no cached answer and no API key."""
        if key not in self.cache:
            if not self.available:
                return None
            raw = self._call(prompt, image)
            self.cache[key] = {'raw': raw, 'model': self.model}
            self._save()
        return parse_json_list(self.cache[key]['raw'])


def parse_json_list(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', text, re.S)
        try:
            data = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            data = []
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    return data


def scores_by_id(items, n_max=10):
    """[{'id': 'A', 'score': 7, ...}] -> {'A': {'score': 7, 'reason': ...}} (scores clipped to 1..n_max)."""
    out = {}
    for d in items or []:
        try:
            out[str(d['id']).strip().upper()] = {'score': max(1, min(n_max, int(d.get('score', 0)))),
                                                 'reason': str(d.get('reason', ''))}
        except (KeyError, TypeError, ValueError):
            continue
    return out
