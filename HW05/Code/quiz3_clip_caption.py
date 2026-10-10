"""
Quiz 3: CLIP image captioning on Flickr8k (the Week 4 dataset).

Encoder : frozen CLIP ViT-B/32 vision transformer -> 50 tokens x 768 ([CLS] + 7x7 patches).
Decoder : the Week 4 LSTM and Transformer decoders, trained from scratch on the CLIP tokens.
Everything except the encoder is identical to Week 4 (split, vocabulary, hyper-parameters,
decoding, BLEU / BERTScore normalisation), so the comparison isolates the encoder:
    Week 4 : ImageNet ResNet-50 (49 x 2048)   vs   Week 5 : CLIP ViT-B/32 (50 x 768)
"""

import math
import os
import textwrap
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .caption_models import (SPECIALS, PAD, BOS, EOS, LSTMCaptioner, TransformerCaptioner, greedy_decode,
                             beam_search, bleu_scores, BertScore)
from .common import DEVICE, plt, set_seed, count_params, ckpt_paths, save_json, load_json, out_path, warmup_cosine
from .data import Vocab, tokenize_caption
from .encoders import clip_patch_tokens, batched_images

# identical to Week 4 (HW04/Code/quiz4_captioning.py)
EPOCHS, BATCH, LR, WEIGHT_DECAY, WARMUP_EPOCHS = 15, 128, 3e-4, 1e-4, 1
MIN_FREQ, MAX_CAP_LEN, BEAM = 5, 30, 3
SPLITS = ('train', 'validation', 'test')


def clip_features(flickr, data_dir):
    feats = {}
    for s in SPLITS:
        path = os.path.join(data_dir, f'q3_clip_tokens_{s}.pt')
        if os.path.exists(path):
            feats[s] = torch.load(path); continue
        t0 = time.perf_counter()
        feats[s] = torch.cat([clip_patch_tokens(ims).half() for ims in batched_images(flickr['hf'][s])])
        torch.save(feats[s], path)
        print(f'[CLIP features] {s}: {tuple(feats[s].shape)} | {time.perf_counter() - t0:.0f}s')
    return feats


def _caption_tensors(captions, vocab):
    img_idx, rows = [], []
    for i, caps in enumerate(captions):
        for c in caps:
            rows.append([BOS] + vocab.encode(tokenize_caption(c))[:MAX_CAP_LEN] + [EOS]); img_idx.append(i)
    seq = torch.full((len(rows), max(map(len, rows))), PAD, dtype=torch.long)
    for k, r in enumerate(rows):
        seq[k, :len(r)] = torch.tensor(r)
    return torch.tensor(img_idx), seq


def _batches(img_idx, seq, shuffle, gen=None):
    order = torch.randperm(len(seq), generator=gen) if shuffle else torch.arange(len(seq))
    for i in range(0, len(seq), BATCH):
        b = order[i:i + BATCH]; s = seq[b]
        yield img_idx[b], s[:, :int((s != PAD).sum(1).max())]


def _loss(model, feats, ii, s):
    f = feats[ii.to(feats.device)].to(DEVICE).float(); s = s.to(DEVICE)
    logits = model(f, s[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), s[:, 1:].reshape(-1), ignore_index=PAD)


def _words(vocab, ids_list):
    return [vocab.decode(ids) for ids in ids_list]


def fit_captioner(name, model, feats, tensors, refs_val, vocab, ckpt_dir, force=False):
    """Keeps the epoch with the best validation BLEU-4 (greedy), exactly as in Week 4."""
    pt, js = ckpt_paths(ckpt_dir, name)
    model.to(DEVICE)
    if not force and os.path.exists(pt) and os.path.exists(js):
        model.load_state_dict(torch.load(pt, map_location=DEVICE))
        hist = load_json(js)
        print(f"[cache] {name}: best epoch {hist['best_epoch']}, val BLEU-4 {hist['best_val_bleu4']:.4f}")
        return hist
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_batches = math.ceil(len(tensors['train'][1]) / BATCH)
    sched = warmup_cosine(opt, EPOCHS * n_batches, WARMUP_EPOCHS * n_batches)
    gen = torch.Generator().manual_seed(42)
    hist, best, best_state, t_start = {'train_loss': [], 'val_loss': [], 'val_bleu4': []}, -1.0, None, time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        t0 = time.perf_counter(); model.train(); tot, n = 0.0, 0
        for ii, s in _batches(*tensors['train'], shuffle=True, gen=gen):
            loss = _loss(model, feats['train'], ii, s)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); tot += loss.item() * len(ii); n += len(ii)
        model.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for ii, s in _batches(*tensors['validation'], shuffle=False):
                vt += _loss(model, feats['validation'], ii, s).item() * len(ii); vn += len(ii)
        bleu4 = bleu_scores(refs_val, _words(vocab, greedy_decode(model, feats['validation'])))['BLEU-4']
        for k, v in zip(hist, (tot / n, vt / vn, bleu4)):
            hist[k].append(round(v, 5))
        if bleu4 > best:
            best, hist['best_epoch'] = bleu4, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'  [{name}] epoch {ep:2d}/{EPOCHS} | train loss {tot / n:.4f} | val loss {vt / vn:.4f} | '
              f'val BLEU-4 {bleu4:.4f} | {time.perf_counter() - t0:.0f}s')
    model.load_state_dict(best_state); torch.save(best_state, pt)
    hist.update({'best_val_bleu4': round(best, 5), 'epochs': EPOCHS, 'device': str(DEVICE),
                 'train_minutes': round((time.perf_counter() - t_start) / 60, 2)})
    save_json(hist, js)
    return hist


def _norm(caption):
    return ' '.join(tokenize_caption(caption))


def run_quiz3(flickr, data_dir, ckpt_dir, output_dir, force=False):
    print('=' * 70); print('QUIZ 3: CLIP IMAGE CAPTIONING (Flickr8k, compared with Week 4)'); print('=' * 70)
    set_seed(42)
    caps = flickr['captions']
    vocab = Vocab((tokenize_caption(c) for cs in caps['train'] for c in cs), specials=SPECIALS, min_freq=MIN_FREQ)
    print(f'Caption vocabulary: {len(vocab):,} words (same rule as Week 4: frequency >= {MIN_FREQ})')
    feats = clip_features(flickr, data_dir)
    feats = {s: (f.to(DEVICE) if s != 'test' else f) for s, f in feats.items()}
    n_tok, dim = feats['train'].shape[1:]
    tensors = {s: _caption_tensors(caps[s], vocab) for s in ('train', 'validation')}
    refs = {s: [[tokenize_caption(c) for c in cs] for cs in caps[s]] for s in ('validation', 'test')}

    models, hists = {}, {}
    for name, build in [('LSTM', lambda: LSTMCaptioner(len(vocab), dim)),
                        ('Transformer', lambda: TransformerCaptioner(len(vocab), dim, n_tok))]:
        set_seed(42)
        model = build()
        hists[f'CLIP-{name}'] = fit_captioner(f'quiz3_clip_{name.lower()}', model, feats, tensors, refs['validation'], vocab, ckpt_dir, force)
        models[name] = model

    gen = {}
    for name, model in models.items():
        gen[f'{name} greedy'] = [' '.join(w) for w in _words(vocab, greedy_decode(model, feats['test']))]
        gen[f'{name} beam{BEAM}'] = [' '.join(vocab.decode(beam_search(model, feats['test'][i:i + 1], BEAM)))
                                     for i in range(len(feats['test']))]
    gen = {m: [_norm(c) for c in v] for m, v in gen.items()}
    bert = BertScore()
    refs_str = [[_norm(c) for c in cs] for cs in caps['test']]
    rows = []
    for m, hyp in gen.items():
        rows.append({'encoder': 'CLIP ViT-B/32 (Week 5)', 'method': m,
                     **{k: round(v, 4) for k, v in bleu_scores(refs['test'], [h.split() for h in hyp]).items()},
                     'BERTScore_F1': round(float(bert(hyp, refs_str).mean()), 4)})
    clip_table = pd.DataFrame(rows)
    clip_table.to_csv(out_path(output_dir, 'quiz3_clip_test_metrics.csv'), index=False)

    week4 = pd.read_csv(os.path.join(data_dir, 'week4_test_metrics.csv'))
    week4.insert(0, 'encoder', ['ImageNet ResNet-50 (Week 4)' if m != 'BLIP-base' else 'BLIP-base (pretrained, Week 4 reference)'
                                for m in week4['method']])
    comparison = pd.concat([week4, clip_table], ignore_index=True)
    deltas = []
    for m in gen:
        w = week4[week4['method'] == m].iloc[0]; c = clip_table[clip_table['method'] == m].iloc[0]
        deltas.append({'method': m, 'BLEU-4 Week 4': w['BLEU-4'], 'BLEU-4 CLIP': c['BLEU-4'],
                       'BLEU-4 change': round(c['BLEU-4'] - w['BLEU-4'], 4), 'BERTScore Week 4': w['BERTScore_F1'],
                       'BERTScore CLIP': c['BERTScore_F1'], 'BERTScore change': round(c['BERTScore_F1'] - w['BERTScore_F1'], 4)})
    delta = pd.DataFrame(deltas)
    comparison.to_csv(out_path(output_dir, 'quiz3_week4_vs_clip.csv'), index=False)
    delta.to_csv(out_path(output_dir, 'quiz3_week4_vs_clip_delta.csv'), index=False)
    print(comparison.to_string(index=False)); print(delta.to_string(index=False))

    figs = {'history': _history_figure(hists, out_path(output_dir, 'quiz3_training_curves.png')),
            'compare': _compare_figure(delta, out_path(output_dir, 'quiz3_week4_vs_clip.png')),
            'samples': _samples_figure(flickr, gen, data_dir, out_path(output_dir, 'quiz3_sample_captions.png'))}
    pd.DataFrame({'test_index': np.arange(len(caps['test'])), 'reference_0': [c[0] for c in caps['test']], **gen}
                 ).to_csv(out_path(output_dir, 'quiz3_test_captions.csv'), index=False)
    params = {f'CLIP-{k}': count_params(v) for k, v in models.items()}
    return {'comparison': comparison, 'delta': delta, 'figures': figs, 'histories': hists, 'params': params,
            'vocab_size': len(vocab)}


def _history_figure(hists, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, h in hists.items():
        ep = range(1, len(h['train_loss']) + 1)
        ax[0].plot(ep, h['train_loss'], '-', label=f'{k} train'); ax[0].plot(ep, h['val_loss'], '--', label=f'{k} val')
        ax[1].plot(ep, h['val_bleu4'], label=k); ax[1].axvline(h['best_epoch'], lw=0.8, alpha=0.5)
    ax[0].set(title='Cross-entropy loss', xlabel='Epoch'); ax[1].set(title='Validation BLEU-4 (greedy)', xlabel='Epoch')
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def _compare_figure(delta, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(delta)); w = 0.38
    for a, metric in zip(ax, ('BLEU-4', 'BERTScore')):
        for j, (col, lab) in enumerate([(f'{metric} Week 4', 'Week 4: ResNet-50 encoder'), (f'{metric} CLIP', 'Week 5: CLIP encoder')]):
            bars = a.bar(x + (j - 0.5) * w, delta[col], w, label=lab)
            for b, v in zip(bars, delta[col]):
                a.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        a.set_xticks(x, delta['method'], fontsize=8.5); a.set_title(f'{metric} on the 1,000 Flickr8k test images', fontweight='bold')
        top = max(delta[f'{metric} Week 4'].max(), delta[f'{metric} CLIP'].max())
        a.set_ylim(0, top * 1.25); a.legend(fontsize=8, ncol=2, loc='upper center'); a.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def _samples_figure(flickr, gen, data_dir, path, idx=(2, 5, 8, 15)):
    week4 = pd.read_csv(os.path.join(data_dir, 'week4_test_captions.csv'))
    fig, ax = plt.subplots(2, len(idx), figsize=(5 * len(idx), 7.5), gridspec_kw={'height_ratios': [1, 0.7]})
    for k, i in enumerate(idx):
        ax[0, k].imshow(flickr['hf']['test'][i]['image'].convert('RGB')); ax[0, k].axis('off')
        ax[0, k].set_title(f'test image #{i}', fontweight='bold')
        lines = [f'Human: {flickr["captions"]["test"][i][0]}',
                 f'Week 4 Transformer beam3: {week4.loc[i, "Transformer beam3"]}',
                 f'CLIP Transformer beam3: {gen["Transformer beam3"][i]}',
                 f'CLIP LSTM beam3: {gen["LSTM beam3"][i]}']
        ax[1, k].text(0, 1, '\n'.join(textwrap.fill(l, 52) for l in lines), va='top', fontsize=8.5, family='monospace')
        ax[1, k].axis('off')
    fig.suptitle('Quiz 3: Week 4 (ResNet-50 encoder) vs Week 5 (CLIP encoder) captions', fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight')
    return fig
