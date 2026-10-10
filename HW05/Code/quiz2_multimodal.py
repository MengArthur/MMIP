"""
Quiz 2: multimodal (image + text) classification -- image-text matching on Flickr8k.

Each sample is an (image, caption) pair; label 1 = the caption describes this image
(one of its 5 human captions), 0 = a caption of a different, randomly chosen image.
Following the lecture (Image + Text Feature Fusion): a frozen image backbone and a
frozen text backbone extract features, which are concatenated and fed to an MLP that
outputs the match probability (0 ~ 1).

Backbone combinations (advanced: swap backbones and compare):
  ResNet-50 + BERT | ViT-B/16 + BERT | CLIP-image + BERT | CLIP-image + CLIP-text
"""

import os
import textwrap
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .common import DEVICE, plt, set_seed, count_params, ckpt_paths, save_json, load_json, out_path, free_gpu
from .data import tokenize_caption
from .encoders import ImageEncoder, TextEncoder, batched_images

SPLITS = ('train', 'validation', 'test')
COMBOS = [('ResNet-50 + BERT', 'resnet50', 'bert'), ('ViT-B/16 + BERT', 'vit_b_16', 'bert'),
          ('CLIP-image + BERT', 'clip', 'bert'), ('CLIP-image + CLIP-text', 'clip', 'clip')]
EPOCHS, BATCH, LR = 20, 512, 1e-3
# main subject words used to build the harder test set (negative caption shares the subject)
SUBJECTS = ('dog', 'dogs', 'man', 'men', 'woman', 'women', 'boy', 'boys', 'girl', 'girls', 'child', 'children',
            'people', 'person', 'player', 'players', 'biker', 'skier', 'surfer', 'climber', 'baby', 'kids')
CUSTOM_TEXTS = ['a dog runs across the grass', 'a child is playing in the water', 'a man rides a bicycle on the street',
                'a black cat sleeping on a sofa']


# ---------------------------------------------------------------- features (cached)
def extract_features(flickr, data_dir):
    """Image features for every image and text features for every caption, per split and backbone."""
    feats = {}
    for split in SPLITS:
        path = os.path.join(data_dir, f'q2_features_{split}.pt')
        if os.path.exists(path):
            feats[split] = torch.load(path); continue
        t0 = time.perf_counter()
        img_names = sorted({c[1] for c in COMBOS}); txt_names = sorted({c[2] for c in COMBOS})
        img_enc = {n: ImageEncoder(n) for n in img_names}
        out = {f'img_{n}': [] for n in img_names}
        for ims in batched_images(flickr['hf'][split]):
            for n, enc in img_enc.items():
                out[f'img_{n}'].append(enc(ims).half())
        img_enc.clear(); free_gpu()
        caps = [c for cs in flickr['captions'][split] for c in cs]                 # image i, caption k -> 5*i + k
        for n in txt_names:
            enc = TextEncoder(n)
            out[f'txt_{n}'] = [enc(caps[i:i + 256]).half() for i in range(0, len(caps), 256)]
            del enc; free_gpu()
        feats[split] = {k: torch.cat(v) for k, v in out.items()}
        torch.save(feats[split], path)
        print(f'[features] {split}: ' + ', '.join(f'{k} {tuple(v.shape)}' for k, v in feats[split].items())
              + f' | {time.perf_counter() - t0:.0f}s')
    return feats


def make_pairs(n_images, seed):
    """Every image gets its 5 own captions (label 1) and 5 captions of random other images (label 0)."""
    rng = np.random.default_rng(seed)
    img = np.repeat(np.arange(n_images), 5)
    k = np.tile(np.arange(5), n_images)
    other = (img + rng.integers(1, n_images, size=img.size)) % n_images          # never the same image
    pos = np.stack([img, img * 5 + k, np.ones_like(img)], 1)
    neg = np.stack([img, other * 5 + k, np.zeros_like(img)], 1)
    return np.concatenate([pos, neg])                                           # (img_idx, caption_idx, label)


def make_hard_pairs(captions, seed):
    """Harder test pairs: each negative caption comes from another image whose caption has the
    SAME main subject (e.g. a dog photo paired with a caption of a different dog photo), so the
    model must check the action / scene, not just the object. Falls back to a random image when
    a caption has no subject word from SUBJECTS."""
    rng = np.random.default_rng(seed)
    subj = [[next((t for t in tokenize_caption(c) if t in SUBJECTS), None) for c in cs] for cs in captions]
    pool = {}
    for i, ss in enumerate(subj):
        for k, w in enumerate(ss):
            if w:
                pool.setdefault(w, []).append((i, k))
    n, rows = len(captions), []
    for i in range(n):
        for k in range(5):
            rows.append((i, i * 5 + k, 1))
            cand = [c for c in pool.get(subj[i][k], []) if c[0] != i]
            if cand:
                j, kk = cand[rng.integers(len(cand))]
            else:
                j, kk = (i + rng.integers(1, n)) % n, k
            rows.append((i, j * 5 + kk, 0))
    return np.array(rows)


class FusionMLP(nn.Module):
    """LayerNorm each modality, concatenate, MLP -> one logit (match / not match)."""

    def __init__(self, d_img, d_txt, hidden=512, dropout=0.3):
        super().__init__()
        self.n_img, self.n_txt = nn.LayerNorm(d_img), nn.LayerNorm(d_txt)
        self.mlp = nn.Sequential(nn.Linear(d_img + d_txt, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, img, txt):
        return self.mlp(torch.cat([self.n_img(img), self.n_txt(txt)], 1)).squeeze(1)


def _gather(feats, split, combo, pairs):
    _, i, t = combo
    f = feats[split]
    return (f[f'img_{i}'][pairs[:, 0]].float().to(DEVICE), f[f'txt_{t}'][pairs[:, 1]].float().to(DEVICE),
            torch.from_numpy(pairs[:, 2]).float().to(DEVICE))


@torch.no_grad()
def _predict(model, img, txt, bs=4096):
    model.eval()
    return torch.cat([torch.sigmoid(model(img[i:i + bs], txt[i:i + bs])) for i in range(0, len(img), bs)]).cpu().numpy()


def train_combo(combo, feats, pairs, ckpt_dir, force=False):
    name = 'quiz2_' + combo[0].replace(' + ', '__').replace('/', '').replace(' ', '_').replace('-', '')
    pt, js = ckpt_paths(ckpt_dir, name)
    Xi, Xt, y = _gather(feats, 'train', combo, pairs['train'])
    Vi, Vt, vy = _gather(feats, 'validation', combo, pairs['validation'])
    set_seed(42)
    model = FusionMLP(Xi.shape[1], Xt.shape[1]).to(DEVICE)
    if os.path.exists(pt) and os.path.exists(js) and not force:
        model.load_state_dict(torch.load(pt, map_location=DEVICE))
        hist = load_json(js); print(f"[cache] {combo[0]}: best epoch {hist['best_epoch']}, val AUC {hist['best_val_auc']:.4f}")
        return model, hist
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    hist, best, best_state, t0 = {'train_loss': [], 'val_auc': []}, -1, None, time.perf_counter()
    g = torch.Generator(device='cpu').manual_seed(42)
    for ep in range(1, EPOCHS + 1):
        model.train(); perm = torch.randperm(len(y), generator=g).to(DEVICE); tot = 0.0
        for s in range(0, len(y), BATCH):
            b = perm[s:s + BATCH]
            loss = F.binary_cross_entropy_with_logits(model(Xi[b], Xt[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        auc = roc_auc_score(vy.cpu().numpy(), _predict(model, Vi, Vt))
        hist['train_loss'].append(round(tot / len(y), 5)); hist['val_auc'].append(round(float(auc), 5))
        if auc > best:
            best, hist['best_epoch'] = auc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); torch.save(best_state, pt)
    hist.update({'best_val_auc': round(float(best), 5), 'epochs': EPOCHS, 'train_seconds': round(time.perf_counter() - t0, 1)})
    save_json(hist, js)
    print(f"  [{combo[0]}] best epoch {hist['best_epoch']}, val AUC {best:.4f}, {hist['train_seconds']}s")
    return model, hist


def _examples_figure(flickr, feats, model, combo, pairs, path, n=4, seed=3):
    """For test images: score the true caption and a mismatched caption."""
    rng = np.random.default_rng(seed)
    imgs = rng.choice(len(flickr['captions']['test']), n, replace=False)
    caps = [c for cs in flickr['captions']['test'] for c in cs]
    fig, ax = plt.subplots(1, n, figsize=(4.6 * n, 5.6))
    for a, i in zip(ax, imgs):
        other = (i + 1 + rng.integers(len(flickr['captions']['test']) - 1)) % len(flickr['captions']['test'])
        cand = np.array([[i, i * 5, 1], [i, other * 5, 0]])
        Xi, Xt, _ = _gather(feats, 'test', combo, cand)
        p = _predict(model, Xi, Xt)
        a.imshow(flickr['hf']['test'][int(i)]['image'].convert('RGB')); a.axis('off')
        lines = [f'{"TRUE" if lab else "WRONG"} caption -> match prob {pp:.2f}\n' + textwrap.fill(caps[c], 45)
                 for (_, c, lab), pp in zip(cand, p)]
        a.set_title('\n\n'.join(lines), fontsize=8.5, loc='left')
    fig.suptitle(f'Quiz 2 inference ({combo[0]}): probability that the caption matches the image', fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight')
    return fig


def custom_text_inference(flickr, feats, model, combo, top=3):
    """Free-text queries (not in the dataset): score every test image and keep the top matches."""
    enc = TextEncoder(combo[2])
    t = enc(CUSTOM_TEXTS).to(DEVICE)
    del enc; free_gpu()
    img = feats['test'][f'img_{combo[1]}'].float().to(DEVICE)
    model.eval()
    with torch.no_grad():
        P = torch.stack([torch.sigmoid(model(img, t[j:j + 1].expand(len(img), -1))) for j in range(len(CUSTOM_TEXTS))])
    P = P.cpu().numpy()                                                    # (n_texts, n_test_images)
    rows = []
    for j, text in enumerate(CUSTOM_TEXTS):
        for r, i in enumerate(np.argsort(-P[j])[:top]):
            rows.append({'query text': text, 'rank': r + 1, 'test_index': int(i), 'match prob': round(float(P[j, i]), 4),
                         'images above 0.5': int((P[j] >= 0.5).sum()),
                         'a human caption of this image': flickr['captions']['test'][i][0]})
    return pd.DataFrame(rows)


def run_quiz2(flickr, data_dir, ckpt_dir, output_dir, force=False):
    print('=' * 70); print('QUIZ 2: MULTIMODAL CLASSIFICATION (image-text matching, Flickr8k)'); print('=' * 70)
    feats = extract_features(flickr, data_dir)
    pairs = {s: make_pairs(len(flickr['captions'][s]), seed=i) for i, s in enumerate(SPLITS)}
    pairs['test_hard'] = make_hard_pairs(flickr['captions']['test'], seed=7)
    for s in SPLITS:
        print(f'{s:10s}: {len(flickr["captions"][s])} images -> {len(pairs[s])} pairs (50% match)')
    print(f'test_hard : {len(pairs["test_hard"])} pairs (negatives share the main subject of the true caption)')
    rows, hists, models = [], {}, {}
    for combo in COMBOS:
        model, hist = train_combo(combo, feats, pairs, ckpt_dir, force)
        Ti, Tt, ty = _gather(feats, 'test', combo, pairs['test'])
        p = _predict(model, Ti, Tt); yt = ty.cpu().numpy(); pred = p >= 0.5
        Hi, Ht, hy = _gather(feats, 'test', combo, pairs['test_hard'])
        ph = _predict(model, Hi, Ht); yh = hy.cpu().numpy()
        rows.append({'backbones': combo[0], 'image_dim': Ti.shape[1], 'text_dim': Tt.shape[1],
                     'Accuracy': round(accuracy_score(yt, pred), 4), 'Precision': round(precision_score(yt, pred), 4),
                     'Recall': round(recall_score(yt, pred), 4), 'F1': round(f1_score(yt, pred), 4),
                     'ROC-AUC': round(roc_auc_score(yt, p), 4),
                     'Accuracy (hard test)': round(accuracy_score(yh, ph >= 0.5), 4),
                     'ROC-AUC (hard test)': round(roc_auc_score(yh, ph), 4), 'best_epoch': hist['best_epoch'],
                     'fusion_params': count_params(model)})
        hists[combo[0]], models[combo[0]] = hist, model
    table = pd.DataFrame(rows)
    table.to_csv(out_path(output_dir, 'quiz2_backbone_comparison.csv'), index=False)
    print(table.to_string(index=False))

    best = max(COMBOS, key=lambda c: table.loc[table['backbones'] == c[0], 'ROC-AUC'].item())
    figs = {'examples': _examples_figure(flickr, feats, models[best[0]], best, pairs,
                                         out_path(output_dir, 'quiz2_inference_examples.png')),
            'examples_baseline': _examples_figure(flickr, feats, models[COMBOS[0][0]], COMBOS[0], pairs,
                                                  out_path(output_dir, 'quiz2_inference_examples_resnet_bert.png'))}
    custom = custom_text_inference(flickr, feats, models[best[0]], best)
    custom.to_csv(out_path(output_dir, 'quiz2_custom_text_inference.csv'), index=False)
    top = custom['rank'].max()
    fig, ax = plt.subplots(len(CUSTOM_TEXTS), top, figsize=(3.4 * top, 2.9 * len(CUSTOM_TEXTS)))
    for j, text in enumerate(CUSTOM_TEXTS):
        sub = custom[custom['query text'] == text]
        for k, (_, r) in enumerate(sub.iterrows()):
            a = ax[j, k]
            a.imshow(flickr['hf']['test'][int(r['test_index'])]['image'].convert('RGB')); a.set_xticks([]); a.set_yticks([])
            a.set_title(f'#{k + 1}  match prob {r["match prob"]:.2f}', fontsize=8)
        ax[j, 0].set_ylabel(textwrap.fill(f'"{text}"', 22), fontsize=9, rotation=0, ha='right', va='center', labelpad=10)
    fig.suptitle(f'Free-text queries: top-{top} of the 1,000 test images ({best[0]})', fontweight='bold')
    fig.tight_layout(); fig.savefig(out_path(output_dir, 'quiz2_custom_text_inference.png'), dpi=120, bbox_inches='tight')
    figs['custom'] = fig

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, h in hists.items():
        ep = range(1, len(h['train_loss']) + 1)
        ax[0].plot(ep, h['train_loss'], label=k); ax[1].plot(ep, h['val_auc'], label=k)
    ax[0].set(title='Training loss (BCE)', xlabel='Epoch'); ax[1].set(title='Validation ROC-AUC', xlabel='Epoch')
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path(output_dir, 'quiz2_training_curves.png'), dpi=150, bbox_inches='tight')
    figs['history'] = fig
    return {'table': table, 'custom': custom, 'figures': figs, 'best': best[0]}
