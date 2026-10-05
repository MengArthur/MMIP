"""
Gemini API helper for Quiz 4 (google-genai SDK):
  - caption(): Gemini itself writes a caption (one of the compared methods)
  - judge():   Gemini grades whether each caption matches the image

The API key is read from the environment / Colab secrets, never stored in code.
Every response (and its latency) is cached in a JSON file, so re-running the
notebook does not call the API again and the reported numbers stay fixed.
"""

import io
import json
import os
import re
import time

CAPTION_PROMPT = ('Write one short sentence (at most 20 words) describing this photo, '
                  'in the style of an image-captioning dataset. Return only the sentence.')

JUDGE_PROMPT = """You are grading image captions. Look at the photo and judge each caption below independently.
For each caption return:
- "verdict": "match" if everything it states is correct for this photo, "partial" if the main subject is right but some detail is wrong or important content is missing, "mismatch" if the main subject or action is wrong.
- "score": integer from 1 to 5 (5 = accurate and specific, 3 = partly correct, 1 = unrelated to the photo).
- "reason": at most 15 words.
Return only a JSON array with one object per caption, in the given order, each with keys "id", "verdict", "score", "reason".

Captions:
"""

_EXCLUDE = ('image', 'tts', 'live', 'audio', 'embedding', 'lite', 'exp', 'preview', 'thinking', 'learnlm', 'gemma',
            'robotics', 'computer', 'native')


def get_api_key():
    """Environment variable first; otherwise Colab Secrets. A key read from Colab Secrets is
    copied into the environment, because userdata.get() needs the browser tab to answer and
    can time out when it is called again late in a long run."""
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
    """Use `preferred` if this key can access it; otherwise the newest stable Gemini Flash model."""
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
    return max(flash, key=lambda n: (version(n), -len(n)))


def _jpeg(image):
    buf = io.BytesIO()
    image.convert('RGB').save(buf, format='JPEG', quality=90)
    return buf.getvalue()


class Gemini:
    def __init__(self, api_key, cache_path, model=None, rpm=10):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = pick_model(self.client, model)
        self.min_interval = 60.0 / rpm
        self._last = 0.0
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding='utf-8') as f:
                self.cache = json.load(f)

    def _save(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=1)

    def _call(self, image, prompt, json_mode, retries=6):
        from google.genai import types
        cfg = types.GenerateContentConfig(temperature=0.0,
                                          response_mime_type='application/json' if json_mode else None)
        contents = [types.Part.from_bytes(data=_jpeg(image), mime_type='image/jpeg'), prompt]
        for attempt in range(retries):
            wait = self.min_interval - (time.time() - self._last)   # stay under the free-tier rate limit
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            t0 = time.perf_counter()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
                return (resp.text or '').strip(), time.perf_counter() - t0
            except Exception as e:
                if attempt == retries - 1:
                    raise
                backoff = min(60, 5 * 2 ** attempt)
                print(f'  [gemini] {type(e).__name__}: {str(e)[:120]} -> retry in {backoff}s')
                time.sleep(backoff)

    def caption(self, key, image):
        if key not in self.cache:
            text, latency = self._call(image, CAPTION_PROMPT, json_mode=False)
            self.cache[key] = {'caption': text, 'latency_s': round(latency, 3), 'model': self.model}
            self._save()
        return self.cache[key]

    def judge(self, key, image, labelled_captions):
        """labelled_captions: [(id, caption), ...] -> {id: {'verdict', 'score', 'reason'}}"""
        if key not in self.cache:
            prompt = JUDGE_PROMPT + '\n'.join(f'{i}: {c}' for i, c in labelled_captions)
            text, latency = self._call(image, prompt, json_mode=True)
            self.cache[key] = {'raw': text, 'latency_s': round(latency, 3), 'model': self.model}
            self._save()
        return _parse_judgement(self.cache[key]['raw'])


def _parse_judgement(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', text, re.S)
        data = json.loads(m.group(0)) if m else []
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    out = {}
    for d in data:
        try:
            out[str(d['id']).strip()] = {'verdict': str(d.get('verdict', '')).lower(), 'score': int(d.get('score', 0)),
                                         'reason': d.get('reason', '')}
        except (KeyError, TypeError, ValueError):
            continue
    return out
