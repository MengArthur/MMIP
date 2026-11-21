"""
作業二: multi-turn VideoRAG, adapted from VideoRAG (Jeong et al., Findings of ACL 2025; slides 56-63).

Indexing (once per video, cached in Output/video_index/):
  audio  -> Whisper large-v3-turbo transcript with timestamps (Simplified -> Traditional via OpenCC). Segments in
            silent parts (low audio energy) are dropped, because Whisper invents text such as "謝謝觀看" on silence.
  frames -> 1 frame / second -> Chinese-CLIP image embeddings -> K-means -> the frame closest to each centroid,
            at most 64 representative frames per video (slide 60)
  frames -> one-sentence description of every representative frame by Qwen3-VL-2B ("visual transcript"): the
            operation videos have no speech, so this is what makes them searchable by text
Answering a turn:
  1. rewrite the question with the conversation history (standalone question)
  2. video retrieval (slide 59): every text unit (transcript segment / frame description) is scored with bge-m3,
     every representative frame with Chinese-CLIP; per video the best scores of both are normalised and fused
  3. frame selection (slide 61): within the best video, 40 random subsets of 6 representative frames are scored by
     the similarity between the query and the subset's mean CLIP embedding, plus a temporal prior that rewards
     frames near the best-matching text units (the paper learns a 3-layer MLP for this; there is no training data
     here, so the zero-shot CLIP score is used); the best subset is kept
  4. Qwen3-VL-2B answers from the selected frames (time-stamped) + the top text units and cites [影片 mm:ss]
"""

import glob
import os
import re
import subprocess

import numpy as np
import torch
from PIL import Image

from .common import DATA_DIR, DEVICE, OUT_DIR, default_dtype, free_gpu, load_json, save_json

VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
INDEX_DIR = os.path.join(OUT_DIR, 'video_index')
ASR_ID = 'openai/whisper-large-v3-turbo'
CLIP_ID = 'OFA-Sys/chinese-clip-vit-base-patch16'
VLM_ID = 'Qwen/Qwen3-VL-2B-Instruct'
N_REPS, SUBSET, N_SUBSETS = 64, 6, 40
HALLUCINATIONS = ('謝謝觀看', '請不吝點贊', '訂閱', '字幕', '明鏡', '點點欄目', 'Amara')


def fmt(t):
    t = int(round(t))
    return f'{t // 60:02d}:{t % 60:02d}'


def list_videos():
    return sorted(p for p in glob.glob(os.path.join(VIDEO_DIR, '*')) if p.lower().endswith(('.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v')))


# ----------------------------------------------------------------------------- models
class Clip:
    def __init__(self, model_id=CLIP_ID):
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        self.proc = ChineseCLIPProcessor.from_pretrained(model_id)
        self.model = ChineseCLIPModel.from_pretrained(model_id, dtype=default_dtype()).to(DEVICE).eval()

    @torch.no_grad()
    def images(self, imgs, bs=32):
        out = []
        for s in range(0, len(imgs), bs):
            enc = self.proc(images=imgs[s:s + bs], return_tensors='pt').to(DEVICE)
            f = self.model.get_image_features(pixel_values=enc['pixel_values'].to(self.model.dtype))
            f = getattr(f, 'pooler_output', f)
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    @torch.no_grad()
    def texts(self, texts):
        enc = self.proc(text=list(texts), padding=True, truncation=True, max_length=52, return_tensors='pt').to(DEVICE)
        f = self.model.get_text_features(**enc)
        f = getattr(f, 'pooler_output', f)
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()


class VLM:
    """Qwen3-VL-2B for frame descriptions, query rewriting and answers."""

    def __init__(self, model_id=VLM_ID, max_pixels=448 * 448, load=True):
        from transformers import AutoProcessor
        self.model_id = model_id
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.proc.tokenizer.padding_side = 'left'
        self.proc.image_processor.size = {'longest_edge': max_pixels, 'shortest_edge': min(65536, max_pixels)}
        self.model = None
        if load:
            self.ensure_loaded()

    def ensure_loaded(self):
        if self.model is not None:
            return
        from transformers import AutoModelForImageTextToText
        kw = {'dtype': default_dtype()}
        if DEVICE.type == 'cuda':
            kw['device_map'] = {'': 0}
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kw).eval()
        self.model.generation_config.pad_token_id = self.proc.tokenizer.pad_token_id

    def unload(self):
        """Free the GPU for Qwen3-4B (the UI swaps the two models); reloaded on the next call."""
        self.model = None
        free_gpu()

    @torch.no_grad()
    def chat(self, batch, max_new_tokens=256):
        """batch: list of message lists whose user contents may contain {'type': 'image', 'image': PIL}."""
        self.ensure_loaded()
        texts, images = [], []
        for msgs in batch:
            texts.append(self.proc.apply_chat_template(
                [{'role': m['role'], 'content': [({'type': 'image'} if c['type'] == 'image' else c) for c in m['content']]}
                 for m in msgs], add_generation_prompt=True, tokenize=False))
            images.append([c['image'] for m in msgs for c in m['content'] if c['type'] == 'image'])
        has_img = any(images)
        enc = self.proc(text=texts, images=images if has_img else None, return_tensors='pt', padding=True)
        enc = {k: (v.to(self.model.device).to(self.model.dtype) if v.is_floating_point() else v.to(self.model.device))
               for k, v in enc.items() if torch.is_tensor(v)}
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                  top_k=None, pad_token_id=self.proc.tokenizer.pad_token_id)
        tok = self.proc.tokenizer
        res = []
        for ids in out[:, enc['input_ids'].shape[1]:].tolist():
            stop = {tok.pad_token_id, tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')}
            n = next((i for i, t in enumerate(ids) if t in stop), len(ids))
            res.append(tok.decode(ids[:n], skip_special_tokens=True).strip())
        return res


def text_msg(text):
    return [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]


# ----------------------------------------------------------------------------- indexing
def read_frames(path, fps=1.0, max_frames=900):
    import cv2
    cap = cv2.VideoCapture(path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n / vfps if n else 0
    step = max(1.0 / fps, dur / max_frames if dur else 0)
    frames, t = [], 0.0
    while t < dur or not dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            break
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        img.thumbnail((640, 640))
        frames.append((t, img))
        t += step
    cap.release()
    return frames, dur


def read_audio(path, sr=16000):
    """16 kHz mono float32 via ffmpeg; None if the video has no audio stream."""
    cmd = ['ffmpeg', '-v', 'quiet', '-i', path, '-vn', '-ac', '1', '-ar', str(sr), '-f', 's16le', '-']
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if not raw:
        return None
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def transcribe(asr, audio, sr=16000):
    """Whisper segments with timestamps; drops segments in near-silent audio and known silence hallucinations."""
    from opencc import OpenCC
    cc = OpenCC('s2twp')
    if audio is None or len(audio) < sr:
        return [], 0.0
    rms_all = float(np.sqrt(np.mean(audio ** 2)))
    win = np.sqrt(np.mean(audio[:len(audio) // sr * sr].reshape(-1, sr) ** 2, axis=1))     # 1-second RMS
    floor = max(0.003, 0.1 * float(np.percentile(win, 90)))                                # "silent" = 10% of loud parts
    out = asr({'raw': audio, 'sampling_rate': sr}, return_timestamps=True, chunk_length_s=30, batch_size=8,
              generate_kwargs={'language': 'zh', 'task': 'transcribe'})
    segs = []
    for ch in out.get('chunks', []):
        s, e = ch['timestamp']
        e = e if e is not None else len(audio) / sr
        part = audio[int(s * sr):int(e * sr)]
        energy = float(np.sqrt(np.mean(part ** 2))) if len(part) else 0.0
        text = cc.convert(ch['text'].strip())
        if not text or energy < floor or any(h in text for h in HALLUCINATIONS) or repetitive(text):
            continue
        segs.append({'start': float(s), 'end': float(e), 'text': text})
    return segs, rms_all


def repetitive(text):
    """Whisper's own hallucination check: text that compresses too well (compression ratio > 2.4) is a loop like
    「小夥伴們的小夥伴們…」."""
    import zlib
    b = text.encode('utf-8')
    return len(b) > 30 and len(b) / len(zlib.compress(b)) > 2.4


def merge_segments(segs, max_len=30.0):
    """Whisper chunks -> retrieval units of at most ~30 s (a few sentences each)."""
    units, cur = [], None
    for s in segs:
        if cur and s['end'] - cur['start'] <= max_len:
            cur['end'], cur['text'] = s['end'], cur['text'] + s['text']
        else:
            cur = dict(s)
            units.append(cur)
    return units


def representatives(times, E, k=N_REPS, seed=0):
    """K-means over CLIP embeddings; the frame closest to every centroid, in time order (slide 60)."""
    from sklearn.cluster import KMeans
    k = min(k, len(E))
    if k == len(E):
        return list(range(len(E)))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(E)
    reps = set()
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        reps.add(int(idx[np.argmin(((E[idx] - km.cluster_centers_[c]) ** 2).sum(1))]))
    return sorted(reps, key=lambda i: times[i])


CAPTION = '用一句繁體中文（30 字以內）描述這個影片畫面：正在進行什麼操作、出現哪些物品，以及畫面上看得到的文字。'


def index_video(path, asr, clip, vlm):
    name = os.path.splitext(os.path.basename(path))[0]
    cache = os.path.join(INDEX_DIR, name + '.json')
    if os.path.exists(cache):
        return load_json(cache)
    frames, dur = read_frames(path)
    times = [t for t, _ in frames]
    E = clip.images([im for _, im in frames])
    reps = representatives(times, E)
    segs, rms = transcribe(asr, read_audio(path)) if asr is not None else ([], 0.0)
    units = [{'kind': 'speech', 'start': u['start'], 'end': u['end'], 'text': u['text']} for u in merge_segments(segs)]
    caps = []
    for s in range(0, len(reps), 8):
        caps += vlm.chat([[{'role': 'user', 'content': [{'type': 'image', 'image': frames[i][1]}, {'type': 'text', 'text': CAPTION}]}]
                          for i in reps[s:s + 8]], 64)
    units += [{'kind': 'visual', 'start': times[i], 'end': times[i] + 1, 'text': c} for i, c in zip(reps, caps)]
    os.makedirs(os.path.join(INDEX_DIR, name), exist_ok=True)
    rep_rows = []
    for j, i in enumerate(reps):
        fp = os.path.join(INDEX_DIR, name, f'{j:03d}_{times[i]:.0f}s.jpg')
        frames[i][1].save(fp, quality=88)
        rep_rows.append({'t': times[i], 'file': os.path.relpath(fp, OUT_DIR), 'caption': caps[j]})
    np.save(os.path.join(INDEX_DIR, name, 'clip.npy'), E[reps])
    rec = {'name': name, 'path': os.path.relpath(path, DATA_DIR), 'duration': dur, 'n_frames_1fps': len(frames),
           'audio_rms': rms, 'has_speech': any(u['kind'] == 'speech' for u in units), 'reps': rep_rows, 'units': units}
    save_json(rec, cache)
    return rec


def build_index(videos=None):
    """Indexes every video in Data/videos (ASR model is loaded only while needed)."""
    videos = videos or list_videos()
    todo = [v for v in videos if not os.path.exists(os.path.join(INDEX_DIR, os.path.splitext(os.path.basename(v))[0] + '.json'))]
    asr = None
    if todo:
        from transformers import pipeline
        asr = pipeline('automatic-speech-recognition', model=ASR_ID, dtype=default_dtype(), device=DEVICE)
    clip = Clip()
    vlm = VLM() if todo else None
    recs = []
    for v in videos:
        print(f'[video] indexing {os.path.basename(v)}')
        recs.append(index_video(v, asr, clip, vlm))
    del asr, vlm
    free_gpu()
    return recs, clip


# ----------------------------------------------------------------------------- retrieval + answering
REWRITE = ('以下是使用者與影片問答助理的對話紀錄與最新問題。請把最新問題改寫成不需要看對話紀錄也能理解的完整問題（補上省略的對象與情境），'
           '只輸出改寫後的問題。\n\n對話紀錄：\n{history}\n\n最新問題：{question}')
ANSWER = ('你是影片問答助理。下面是從影片庫中檢索到的片段：先是影片畫面（每張前面標示影片名稱與時間），再是語音逐字稿與畫面描述。'
          '請只根據這些內容回答，並在依據的內容後面用【實際的影片名稱 mm:ss】標示時間點，例如【{example}】；'
          '影片名稱只能是：{names}。如果內容不足以回答，請直接說明。回答用繁體中文，不超過 120 字。'
          '\n\n文字資料：\n{text}\n\n問題：{question}')


class VideoRAG:
    def __init__(self, recs, clip, emb, vlm, n_frames=SUBSET, n_text=4, seed=0):
        self.recs, self.clip, self.emb, self.vlm = recs, clip, emb, vlm
        self.n_frames, self.n_text, self.seed = n_frames, n_text, seed
        self.units = [(vi, u) for vi, r in enumerate(recs) for u in r['units']]
        self.U = emb.encode([u['text'] for _, u in self.units], name='video_units') if self.units else np.zeros((0, 1024))
        self.F = [np.load(os.path.join(INDEX_DIR, r['name'], 'clip.npy')) for r in recs]
        self._img = {}

    def image(self, rel):
        if rel not in self._img:
            self._img[rel] = Image.open(os.path.join(OUT_DIR, rel)).convert('RGB')
        return self._img[rel]

    def standalone(self, question, history):
        if not history:
            return question
        h = '\n'.join(f'使用者：{q}\n助理：{a[:150]}' for q, a in history[-3:])
        return self.vlm.chat([text_msg(REWRITE.format(history=h, question=question))], 80)[0].split('\n')[0] or question

    def retrieve_videos(self, q, w_text=0.7):
        """Per-video score: the best text unit (bge-m3) and the best frame (CLIP), each min-max normalised over the
        videos, combined 0.7 / 0.3 (text is the more reliable signal). Rank fusion was tried first, but with only a
        handful of videos it produced near-ties, so the actual similarities are fused."""
        qv = self.emb.encode([q])[0]
        qc = self.clip.texts([q])[0]
        ts = self.U @ qv if len(self.U) else np.zeros(0)
        best_text = np.full(len(self.recs), -1.0)
        for (vi, _), s in zip(self.units, ts):
            best_text[vi] = max(best_text[vi], s)
        best_img = np.array([float((F @ qc).max()) if len(F) else -1.0 for F in self.F])
        mm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
        fused = w_text * mm(best_text) + (1 - w_text) * mm(best_img)
        order = list(np.argsort(-fused))
        return order, {'text': best_text, 'image': best_img, 'fused': fused}, qv, qc, ts

    def select_frames(self, vi, qc, prior_times, method='videorag'):
        """slide 61 adapted: best of 40 random 6-frame subsets by CLIP(mean of subset) . query + temporal prior."""
        reps, F = self.recs[vi]['reps'], self.F[vi]
        m = min(self.n_frames, len(reps))
        if method == 'uniform':
            return list(np.linspace(0, len(reps) - 1, m).round().astype(int))
        rng = np.random.default_rng(self.seed)
        if method == 'random':
            return sorted(rng.choice(len(reps), m, replace=False).tolist())
        t = np.array([r['t'] for r in reps])
        prior = np.zeros(len(reps))
        for pt in prior_times:
            prior = np.maximum(prior, np.exp(-np.abs(t - pt) / 15.0))   # frames within ~15 s of a matching text unit
        rel = F @ qc
        cand = np.argsort(-(rel + 0.1 * prior))[:max(16, m)]             # sample the subsets from the 16 best frames
        best, best_s = None, -1e9
        for _ in range(N_SUBSETS):
            sub = rng.choice(cand, m, replace=False)
            v = F[sub].mean(0)
            s = float(v @ qc / (np.linalg.norm(v) + 1e-8)) + 0.1 * float(prior[sub].mean())
            if s > best_s:
                best, best_s = sub, s
        return sorted(best.tolist(), key=lambda i: reps[i]['t'])

    def answer(self, question, history=None, mode='both', frames_method='videorag', top_videos=1):
        """mode: both | text (no frames) | frames (no text units)."""
        q = self.standalone(question, history or [])
        order, vs, qv, qc, ts = self.retrieve_videos(q)
        vids = order[:top_videos]
        units = sorted([i for i in range(len(self.units)) if self.units[i][0] in vids], key=lambda i: -ts[i])[:self.n_text]
        prior_times = [self.units[i][1]['start'] for i in units]
        content, used_frames = [], []
        if mode in ('both', 'frames'):
            for vi in vids:
                for j in self.select_frames(vi, qc, prior_times, frames_method):
                    r = self.recs[vi]['reps'][j]
                    content += [{'type': 'text', 'text': f'【{self.recs[vi]["name"]} {fmt(r["t"])}】'},
                                {'type': 'image', 'image': self.image(r['file'])}]
                    used_frames.append({'video': self.recs[vi]['name'], 't': r['t'], 'file': r['file']})
        text_units = []
        if mode in ('both', 'text'):
            for i in sorted(units, key=lambda i: self.units[i][1]['start']):
                vi, u = self.units[i]
                kind = '語音' if u['kind'] == 'speech' else '畫面描述'
                text_units.append({'video': self.recs[vi]['name'], 'start': u['start'], 'end': u['end'], 'kind': u['kind'],
                                   'text': u['text'], 'score': float(ts[i])})
        text = '\n'.join(f'【{x["video"]} {fmt(x["start"])}–{fmt(x["end"])}】（{"語音" if x["kind"] == "speech" else "畫面描述"}）{x["text"]}'
                         for x in text_units) or '（無）'
        msgs = []
        for hq, ha in (history or [])[-2:]:
            msgs += [{'role': 'user', 'content': [{'type': 'text', 'text': hq}]},
                     {'role': 'assistant', 'content': [{'type': 'text', 'text': ha}]}]
        names = [self.recs[v]['name'] for v in vids]
        ex_t = used_frames[0]['t'] if used_frames else (text_units[0]['start'] if text_units else 0)
        prompt = ANSWER.format(text=text, question=q, names='、'.join(names), example=f'{names[0]} {fmt(ex_t)}')
        msgs.append({'role': 'user', 'content': content + [{'type': 'text', 'text': prompt}]})
        ans = self.vlm.chat([msgs], 256)[0]
        return {'question': question, 'standalone': q, 'answer': ans, 'videos': [self.recs[v]['name'] for v in order[:3]],
                'video_scores': {self.recs[v]['name']: round(float(vs['fused'][v]), 5) for v in order[:3]},
                'frames': used_frames, 'text_units': text_units}


def citations(answer):
    """【影片 mm:ss】 references in an answer -> [(video, seconds)]."""
    out = []
    for name, mm, ss in re.findall(r'【([^】\s]+)\s+(\d{1,2}):(\d{2})', answer or ''):
        out.append((name, int(mm) * 60 + int(ss)))
    return out
