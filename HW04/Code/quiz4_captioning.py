"""
Quiz 4: Image Captioning on Flickr8k.

  Basic    - train two captioning models (CNN-LSTM, CNN-Transformer) and generate captions;
             Gemini judges whether the generated captions match the image.
  Advanced - BLEU-1..4 and BERTScore; compare response time and quality of six methods:
             our two models x {greedy, beam search}, pretrained BLIP (zero-shot) and Gemini.
"""

import math
import os
import textwrap
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .caption_models import (SPECIALS, PAD, BOS, EOS, ENC_TF, build_encoder, encode_images, extract_features,
                             LSTMCaptioner, TransformerCaptioner, greedy_decode, beam_search, bleu_scores, BertScore)
from .common import DEVICE, plt, set_seed, count_params, ckpt_paths, save_json, load_json, warmup_cosine, plot_histories, save_fig_path
from .data import Vocab, tokenize_caption
from . import gemini as gm

EPOCHS = 15
BATCH = 128
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
MIN_FREQ = 5          # words seen fewer than 5 times in training captions -> <unk>
MAX_CAP_LEN = 30
BEAM = 3
N_SUBSET = 20         # test images used for timing and for the Gemini calls (limited by the free API quota)
GEMINI_MODEL = 'gemini-3.5-flash-lite'   # same model for Gemini captions and for judging, so scores are comparable
BLIP_ID = 'Salesforce/blip-image-captioning-base'
SPLITS = ('train', 'validation', 'test')


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def _caption_tensors(captions, vocab):
    img_idx, rows = [], []
    for i, caps in enumerate(captions):
        for c in caps:
            rows.append([BOS] + vocab.encode(tokenize_caption(c))[:MAX_CAP_LEN] + [EOS])
            img_idx.append(i)
    seq = torch.full((len(rows), max(map(len, rows))), PAD, dtype=torch.long)
    for k, r in enumerate(rows):
        seq[k, :len(r)] = torch.tensor(r)
    return torch.tensor(img_idx), seq


def _batches(img_idx, seq, shuffle, gen=None):
    order = torch.randperm(len(seq), generator=gen) if shuffle else torch.arange(len(seq))
    for i in range(0, len(seq), BATCH):
        b = order[i:i + BATCH]
        s = seq[b]
        yield img_idx[b], s[:, :int((s != PAD).sum(1).max())]


def _loss(model, feats, ii, s):
    f = feats[ii.to(feats.device)].to(DEVICE).float()
    s = s.to(DEVICE)
    logits = model(f, s[:, :-1])                       # teacher forcing: predict token t+1 from tokens <= t
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), s[:, 1:].reshape(-1), ignore_index=PAD)


def _words(vocab, ids_list):
    return [vocab.decode(ids) for ids in ids_list]


def fit_captioner(name, model, feats, tensors, refs_val, vocab, ckpt_dir, force=False):
    """Keeps the epoch with the best validation BLEU-4 (greedy decoding)."""
    pt, js = ckpt_paths(ckpt_dir, name)
    model.to(DEVICE)
    if not force and os.path.exists(pt) and os.path.exists(js):
        model.load_state_dict(torch.load(pt, map_location=DEVICE))
        hist = load_json(js)
        print(f"[cache] {name}: loaded best checkpoint (epoch {hist['best_epoch']}, val BLEU-4 {hist['best_val_bleu4']:.4f})")
        return hist

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_batches = math.ceil(len(tensors['train'][1]) / BATCH)
    sched = warmup_cosine(opt, EPOCHS * n_batches, WARMUP_EPOCHS * n_batches)
    gen = torch.Generator().manual_seed(42)
    hist = {'train_loss': [], 'val_loss': [], 'val_bleu4': []}
    best, best_state, best_epoch = -1.0, None, 0
    t_start = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        tot, n = 0.0, 0
        for ii, s in _batches(*tensors['train'], shuffle=True, gen=gen):
            loss = _loss(model, feats['train'], ii, s)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item() * len(ii); n += len(ii)
        model.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for ii, s in _batches(*tensors['validation'], shuffle=False):
                vt += _loss(model, feats['validation'], ii, s).item() * len(ii); vn += len(ii)
        bleu4 = bleu_scores(refs_val, _words(vocab, greedy_decode(model, feats['validation'])))['BLEU-4']
        for k, v in zip(hist, (tot / n, vt / vn, bleu4)):
            hist[k].append(round(v, 5))
        if bleu4 > best:
            best, best_epoch = bleu4, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'  [{name}] epoch {ep:2d}/{EPOCHS} | train loss {tot / n:.4f} | val loss {vt / vn:.4f} | '
              f'val BLEU-4 {bleu4:.4f} | {time.perf_counter() - t0:.0f}s')
    model.load_state_dict(best_state)
    torch.save(best_state, pt)
    hist.update({'name': name, 'epochs': EPOCHS, 'best_epoch': best_epoch, 'best_val_bleu4': round(best, 5),
                 'train_minutes': round((time.perf_counter() - t_start) / 60, 2), 'device': str(DEVICE), 'lr': LR})
    save_json(hist, js)
    return hist


# ----------------------------------------------------------------------------
# Caption generators (end-to-end: PIL image -> caption string)
# ----------------------------------------------------------------------------
def _sync():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def _timed(fn, images, warmup=3):
    for img in images[:warmup]:                       # warm-up (CUDA kernels, caches) not timed
        fn(img)
    out, secs = [], []
    for img in images:
        _sync(); t0 = time.perf_counter()
        out.append(fn(img))
        _sync(); secs.append(time.perf_counter() - t0)
    return out, np.array(secs)


def _ours(encoder, model, vocab, beam):
    @torch.no_grad()
    def fn(img):
        f = encode_images(encoder, ENC_TF(img.convert('RGB')).unsqueeze(0).to(DEVICE))
        ids = beam_search(model, f, BEAM) if beam else greedy_decode(model, f)[0]
        return ' '.join(vocab.decode(ids))
    return fn


class _Blip:
    def __init__(self):
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.dtype = torch.float16 if DEVICE.type == 'cuda' else torch.float32
        self.proc = BlipProcessor.from_pretrained(BLIP_ID)
        self.model = BlipForConditionalGeneration.from_pretrained(BLIP_ID).to(DEVICE, dtype=self.dtype).eval()

    @torch.no_grad()
    def __call__(self, images):
        inputs = self.proc(images=[im.convert('RGB') for im in images], return_tensors='pt').to(DEVICE, self.dtype)
        out = self.model.generate(**inputs, max_new_tokens=30, num_beams=BEAM)
        return [c.strip() for c in self.proc.batch_decode(out, skip_special_tokens=True)]

    def one(self, img):
        return self([img])[0]


def _norm(caption):
    """All methods are scored on the same normalisation: lower-case words only."""
    return ' '.join(tokenize_caption(caption))


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _samples_figure(images, idx, captions, judge, path):
    fig, axes = plt.subplots(2, len(idx), figsize=(5 * len(idx), 7), gridspec_kw={'height_ratios': [1, 0.6]})
    for k, i in enumerate(idx):
        axes[0, k].imshow(images[k].convert('RGB')); axes[0, k].axis('off')
        axes[0, k].set_title(f'test image #{i}', fontweight='bold')
        lines = []
        for m, caps in captions.items():
            score = judge.get((i, m), {}).get('score')
            tag = f' [Gemini {score}/5]' if score else ''
            lines.append(textwrap.fill(f'{m}{tag}: {caps[i]}', 55))
        axes[1, k].text(0, 1, '\n'.join(lines), va='top', fontsize=8.5, family='monospace'); axes[1, k].axis('off')
    fig.suptitle('Quiz 4: captions generated by each method ([Gemini x/5] = Gemini judge score)', fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def _tradeoff_figure(subset_table, judge_counts, n_subset, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    ax = axes[0]
    t = subset_table.dropna(subset=['median_latency_ms'])
    ax.scatter(t['median_latency_ms'], t['BERTScore_F1'], s=60)
    for _, r in t.iterrows():
        ax.annotate(r['method'], (r['median_latency_ms'], r['BERTScore_F1']), xytext=(6, 4), textcoords='offset points', fontsize=9)
    ax.set_xscale('log'); ax.set_xlabel('median response time per image (ms, log scale)')
    ax.set_ylabel('BERTScore F1 (rescaled)'); ax.grid(alpha=0.3)
    ax.set_title(f'Speed vs quality on the same {n_subset} test images', fontweight='bold')
    ax = axes[1]
    if judge_counts is not None:
        bottom = np.zeros(len(judge_counts))
        for verdict, col in [('match', '#2ca02c'), ('partial', '#ff7f0e'), ('mismatch', '#d62728')]:
            v = judge_counts[verdict].values
            ax.barh(judge_counts.index, v, left=bottom, color=col, label=verdict)
            bottom += v
        ax.invert_yaxis(); ax.set_xlabel('share of captions'); ax.set_xlim(0, 1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
        ax.set_title(f'Gemini judge verdicts on {n_subset} test images (does the caption match the image?)',
                     fontweight='bold')
    else:
        ax.axis('off'); ax.text(0.5, 0.5, 'Gemini API key not set: judge skipped', ha='center')
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run_quiz4(flickr, data_dir, ckpt_dir, output_dir, force=False, pretrained=True,
              gemini_model=GEMINI_MODEL, gemini_rpm=10, n_subset=N_SUBSET, use_blip=True):
    print('=' * 70); print('QUIZ 4: IMAGE CAPTIONING (Flickr8k)'); print('=' * 70)
    set_seed(42)
    caps, hf = flickr['captions'], flickr['hf']
    vocab = Vocab((tokenize_caption(c) for cs in caps['train'] for c in cs), specials=SPECIALS, min_freq=MIN_FREQ)
    print(f'Caption vocabulary: {len(vocab):,} words (frequency >= {MIN_FREQ} in training captions)')

    encoder = build_encoder(pretrained).to(DEVICE)
    feats = {s: extract_features(encoder, hf[s], f'{data_dir}/flickr8k_{s}_resnet50.pt') for s in SPLITS}
    feats = {s: (f.to(DEVICE) if s != 'test' else f) for s, f in feats.items()}
    tensors = {s: _caption_tensors(caps[s], vocab) for s in ('train', 'validation')}
    refs = {s: [[tokenize_caption(c) for c in cs] for cs in caps[s]] for s in ('validation', 'test')}

    models, histories, rows = {}, {}, []
    for name, cls in [('LSTM', LSTMCaptioner), ('Transformer', TransformerCaptioner)]:
        set_seed(42)
        model = cls(len(vocab))
        histories[f'CNN-{name}'] = fit_captioner(f'quiz4_cnn_{name.lower()}', model, feats, tensors,
                                                 refs['validation'], vocab, ckpt_dir, force)
        models[name] = model

    # ---- full test set (1,000 images): our models + BLIP ------------------------------
    gen = {}
    for name, model in models.items():
        gen[f'{name} greedy'] = [' '.join(w) for w in _words(vocab, greedy_decode(model, feats['test']))]
        gen[f'{name} beam{BEAM}'] = [' '.join(vocab.decode(beam_search(model, feats['test'][i:i + 1], BEAM)))
                                     for i in range(len(feats['test']))]
    blip = None
    if use_blip:
        blip = _Blip()
        test_images = [hf['test'][i]['image'] for i in range(len(caps['test']))]
        gen['BLIP-base'] = sum((blip(test_images[i:i + 32]) for i in range(0, len(test_images), 32)), [])
    gen = {m: [_norm(c) for c in v] for m, v in gen.items()}

    bert = BertScore()
    refs_str = [[_norm(c) for c in cs] for cs in caps['test']]
    full_rows = []
    for m, hyp in gen.items():
        f1 = bert(hyp, refs_str)
        full_rows.append({'method': m, **{k: round(v, 4) for k, v in bleu_scores(refs['test'], [h.split() for h in hyp]).items()},
                          'BERTScore_F1': round(float(f1.mean()), 4)})
    full_table = pd.DataFrame(full_rows)
    full_table.to_csv(save_fig_path(output_dir, 'quiz4_test_metrics.csv'), index=False)
    print(full_table.to_string(index=False))

    # ---- subset (n_subset images): response time + Gemini ----------------------------
    subset = np.sort(np.random.default_rng(0).choice(len(caps['test']), n_subset, replace=False))
    images = [hf['test'][int(i)]['image'] for i in subset]
    latency = {}
    for name, model in models.items():
        for beam in (False, True):
            m = f'{name} beam{BEAM}' if beam else f'{name} greedy'
            _, latency[m] = _timed(_ours(encoder, model, vocab, beam), images)
    if blip is not None:
        _, latency['BLIP-base'] = _timed(blip.one, images)
        del blip
    sub_caps = {m: {int(i): gen[m][i] for i in subset} for m in gen}

    key = gm.get_api_key()
    judge, judge_rows, gemini_model_used = {}, [], None
    if key:
        g = gm.Gemini(key, save_fig_path(output_dir, 'quiz4_gemini_cache.json'), model=gemini_model, rpm=gemini_rpm)
        gemini_model_used = g.model
        print(f'Gemini model: {g.model} ({2 * n_subset} API calls at most, cached in Output/quiz4_gemini_cache.json)')
        res = [g.caption(f'caption:{int(i)}', img) for i, img in zip(subset, images)]
        sub_caps['Gemini'] = {int(i): _norm(r['caption']) for i, r in zip(subset, res)}
        latency['Gemini'] = np.array([r['latency_s'] for r in res])
        candidates = {**sub_caps, 'Human reference': {int(i): _norm(caps['test'][i][0]) for i in subset}}
        rng = np.random.default_rng(1)
        for i, img in zip(subset, images):
            methods = list(candidates)
            rng.shuffle(methods)                      # blind: Gemini only sees letters, in random order
            labels = [chr(ord('A') + k) for k in range(len(methods))]
            verdicts = g.judge(f'judge:{int(i)}', img, [(l, candidates[m][int(i)]) for l, m in zip(labels, methods)])
            for l, m in zip(labels, methods):
                v = verdicts.get(l, {})
                judge[(int(i), m)] = v
                judge_rows.append({'test_index': int(i), 'method': m, 'caption': candidates[m][int(i)],
                                   'verdict': v.get('verdict'), 'score': v.get('score'), 'reason': v.get('reason')})
        pd.DataFrame(judge_rows).to_csv(save_fig_path(output_dir, 'quiz4_gemini_judgements.csv'), index=False)
    else:
        print('[skip] GEMINI_API_KEY not found (Colab: add it under Secrets) -> Gemini caption & judge skipped')

    refs_sub = [refs['test'][i] for i in subset]
    refs_sub_str = [refs_str[i] for i in subset]
    judge_df = pd.DataFrame(judge_rows)
    sub_rows = []
    for m, c in sub_caps.items():
        hyp = [c[int(i)] for i in subset]
        row = {'method': m, 'BLEU-4': round(bleu_scores(refs_sub, [h.split() for h in hyp])['BLEU-4'], 4),
               'BERTScore_F1': round(float(bert(hyp, refs_sub_str).mean()), 4),
               'mean_latency_ms': round(1000 * latency[m].mean(), 1), 'median_latency_ms': round(1000 * np.median(latency[m]), 1)}
        if len(judge_df):
            j = judge_df[judge_df['method'] == m]
            row.update({'gemini_mean_score': round(j['score'].mean(), 2), 'gemini_match_rate': round((j['verdict'] == 'match').mean(), 3)})
        sub_rows.append(row)
    if len(judge_df):
        j = judge_df[judge_df['method'] == 'Human reference']
        sub_rows.append({'method': 'Human reference', 'gemini_mean_score': round(j['score'].mean(), 2),
                         'gemini_match_rate': round((j['verdict'] == 'match').mean(), 3)})
    subset_table = pd.DataFrame(sub_rows)
    params = {'LSTM': count_params(models['LSTM']), 'Transformer': count_params(models['Transformer'])}
    enc_params = count_params(encoder)
    subset_table['trainable_params'] = [params.get(m.split()[0]) for m in subset_table['method']]
    subset_table.to_csv(save_fig_path(output_dir, 'quiz4_method_comparison.csv'), index=False)
    print(subset_table.to_string(index=False))

    judge_counts = None
    if len(judge_df):
        judge_counts = (judge_df.groupby('method')['verdict'].value_counts(normalize=True).unstack(fill_value=0)
                        .reindex(columns=['match', 'partial', 'mismatch'], fill_value=0)
                        .reindex(list(sub_caps) + ['Human reference']))

    show = subset[:4]
    figs = {'curves': plot_histories(histories, save_fig_path(output_dir, 'quiz4_training_curves.png'),
                                     'Quiz 4: captioning models on Flickr8k', right_key='bleu4',
                                     right_label='Validation BLEU-4 (greedy)'),
            'samples': _samples_figure(images[:4], show,
                                       {**sub_caps, 'Human reference': {int(i): caps['test'][i][0] for i in show}},
                                       judge, save_fig_path(output_dir, 'quiz4_sample_captions.png')),
            'tradeoff': _tradeoff_figure(subset_table, judge_counts, n_subset,
                                         save_fig_path(output_dir, 'quiz4_speed_vs_quality.png'))}

    pd.DataFrame({'test_index': np.arange(len(caps['test'])), 'reference_0': [c[0] for c in caps['test']], **gen}
                 ).to_csv(save_fig_path(output_dir, 'quiz4_test_captions.csv'), index=False)
    return {'full_table': full_table, 'subset_table': subset_table, 'judge': judge_df, 'figures': figs,
            'histories': histories, 'vocab_size': len(vocab), 'gemini_model': gemini_model_used,
            'encoder_params': enc_params, 'decoder_params': params, 'subset': subset}
