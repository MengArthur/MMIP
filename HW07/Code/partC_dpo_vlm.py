"""
Part C advanced: DPO fine-tuning of a VLM (Qwen3-VL-2B-Instruct, LoRA) to reduce object hallucination
("describe a photo without mentioning things that are not there"), plus robustness to image degradations.

  Adv. 1  preference data from 150 COCO val2017 images (Data/partC_coco_subset.json), up to 450 pairs:
            <=150 description pairs  the base model describes each image; objects it mentions that are not in
                                   the COCO annotations are hallucinations -> rejected = its own description,
                                   chosen = the description without the hallucinated sentences (skipped if that
                                   would remove more than half of the text). If it did not
                                   hallucinate, rejected = its description with one sentence about an absent but
                                   commonly co-occurring object inserted (chosen = original)
            300 yes/no pairs       "Is there a <obj> in the image?" for one present and one absent object
  Adv. 2  DPO (LoRA on the language model); evaluated on 80 held-out COCO images:
            POPE (480 yes/no questions: 3 present, 2 adversarial-absent, 1 random-absent object per image)
            CHAIR_s / CHAIR_i / object recall of "Describe this image in detail." descriptions
  Adv. 3  POPE under image degradations (darker, brighter, low contrast, JPEG q=5, blur, low resolution,
          Gaussian noise) for the base and the DPO model
"""

import io
import os
import random
import re
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from .common import DATA_DIR, load_json, save_json, out_path, ckpt_path, free_gpu, plt, gpu_info, run_batched
from . import dpo
from .coco_objects import chair, mentioned_objects, SYNONYMS
from .vlm import VLMS, load_vlm, generate, score_yes_no

VLM_NAME = 'Qwen3-VL-2B'
MAX_PIXELS = 448 * 448          # ~196 visual tokens per image: faster DPO / POPE, same setting before and after DPO
DESC_PROMPT = 'Describe this image in detail.'
DESC_MAX_NEW = 192
BATCH = 8
DPO_CFG = dict(beta=0.1, lr=5e-5, epochs=2, grad_accum=8)
LORA_R = 16
N_VAL = 45
WHERE = ['in the background', 'on the left side', 'on the right side', 'near the center', 'in the foreground']
DEGRADATIONS = ['original', 'dark (x0.35)', 'bright (x1.8)', 'low contrast (x0.4)', 'JPEG q=5', 'blur r=4',
                'low-res 64 px', 'noise sigma=50']


# ----------------------------------------------------------------------------- data
def coco_image(file_name, cache_dir=os.path.join(DATA_DIR, 'coco_images')):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, file_name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(f'http://images.cocodataset.org/val2017/{file_name}', path)
    return Image.open(path).convert('RGB')


def degrade(im, kind, seed=0):
    if kind == 'original':
        return im
    if kind.startswith('dark'):
        return ImageEnhance.Brightness(im).enhance(0.35)
    if kind.startswith('bright'):
        return ImageEnhance.Brightness(im).enhance(1.8)
    if kind.startswith('low contrast'):
        return ImageEnhance.Contrast(im).enhance(0.4)
    if kind.startswith('JPEG'):
        buf = io.BytesIO(); im.save(buf, format='JPEG', quality=5); buf.seek(0)
        return Image.open(buf).convert('RGB')
    if kind.startswith('blur'):
        return im.filter(ImageFilter.GaussianBlur(4))
    if kind.startswith('low-res'):
        s = 64 / max(im.size)
        return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR).resize(im.size, Image.BILINEAR)
    if kind.startswith('noise'):
        a = np.asarray(im).astype(np.float32) + np.random.default_rng(seed).normal(0, 50, (im.height, im.width, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    raise ValueError(kind)


def _article(obj):
    return ('an ' if obj[0] in 'aeiou' else 'a ') + obj


def complete_sentences(text):
    """Drop a sentence cut off by max_new_tokens."""
    text = text.strip()
    if text and text[-1] not in '.!?"':
        cut = max(text.rfind('. '), text.rfind('.\n'), text.rfind('!'), text.rfind('?'))
        if cut > 0:
            text = text[:cut + 1]
    return text.strip()


def _sentences(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]


def remove_hallucinated(text, hallucinated):
    keep = [s for s in _sentences(text) if not (mentioned_objects(s) & set(hallucinated))]
    return ' '.join(keep)


def build_pairs(train_rows, base_desc):
    """Description pairs (on-policy where the base model hallucinated) + present / absent yes-no pairs."""
    pairs, stats = [], {'desc_on_policy': 0, 'desc_injected': 0, 'desc_skipped': 0, 'yes': 0, 'no': 0}
    for row, desc in zip(train_rows, base_desc):
        rng = random.Random(row['image_id'])
        desc = complete_sentences(desc)
        hall = sorted(mentioned_objects(desc) - set(row['objects']))
        if hall:
            cleaned = remove_hallucinated(desc, hall)
            if len(cleaned) >= 0.5 * len(desc):          # keep only if most of the description survives
                pairs.append({'image': row['file_name'], 'prompt': DESC_PROMPT, 'chosen': cleaned, 'rejected': desc,
                              'kind': 'desc_on_policy', 'hallucinated': hall})
                stats['desc_on_policy'] += 1
            else:                                        # a stub "chosen" would only teach short answers
                stats['desc_skipped'] += 1
        elif desc:
            pool = [o for o in row['adversarial_pool'] if o not in mentioned_objects(desc)]
            obj = pool[0] if pool else rng.choice([o for o in SYNONYMS if o not in row['objects']])
            sents = _sentences(desc)
            k = min(len(sents), 1 + rng.randrange(max(1, len(sents) - 1)))
            fake = f'There is also {_article(obj)} {rng.choice(WHERE)}.'
            pairs.append({'image': row['file_name'], 'prompt': DESC_PROMPT, 'chosen': desc,
                          'rejected': ' '.join(sents[:k] + [fake] + sents[k:]), 'kind': 'desc_injected', 'hallucinated': [obj]})
            stats['desc_injected'] += 1
        pos = rng.choice(row['salient'])
        neg = row['adversarial_pool'][0] if row['adversarial_pool'] else rng.choice([o for o in SYNONYMS if o not in row['objects']])
        for obj, present in ((pos, True), (neg, False)):
            q = f'Is there {_article(obj)} in the image? Answer with Yes or No.'
            yes, no = f'Yes, there is {_article(obj)} in the image.', f'No, there is no {obj} in the image.'
            pairs.append({'image': row['file_name'], 'prompt': q, 'chosen': yes if present else no,
                          'rejected': no if present else yes, 'kind': 'yes' if present else 'no'})
            stats['yes' if present else 'no'] += 1
    return pairs, stats


def pope_questions(rows):
    qs = []
    for r in rows:
        for o in r['pope_pos']:
            qs.append({'file_name': r['file_name'], 'object': o, 'label': 1, 'type': 'present'})
        for o in r['pope_neg_adversarial']:
            qs.append({'file_name': r['file_name'], 'object': o, 'label': 0, 'type': 'absent (adversarial)'})
        for o in r['pope_neg_random']:
            qs.append({'file_name': r['file_name'], 'object': o, 'label': 0, 'type': 'absent (random)'})
    return qs


# ----------------------------------------------------------------------------- evaluation
def describe(model, proc, files, path):
    if os.path.exists(path):
        return load_json(path)
    fn = lambda chunk: [r['text'] for r in generate(model, proc, [coco_image(f) for f in chunk], [DESC_PROMPT] * len(chunk), DESC_MAX_NEW)]
    texts, _ = run_batched(fn, files, BATCH)
    save_json(texts, path)
    return texts


def pope(model, proc, qs, kind, path):
    if os.path.exists(path):
        return load_json(path)
    fn = lambda chunk: score_yes_no(model, proc, [degrade(coco_image(q['file_name']), kind, seed=int(q['file_name'][:12])) for q in chunk],
                                    [f'Is there {_article(q["object"])} in the image?' for q in chunk])
    p, _ = run_batched(fn, qs, BATCH)
    save_json(p, path)
    return p


def pope_metrics(qs, p_yes):
    y = np.array([q['label'] for q in qs]); pred = (np.array(p_yes) > 0.5).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    types = np.array([q['type'] for q in qs])
    from sklearn.metrics import roc_auc_score
    return {'accuracy': round(float((pred == y).mean()), 4), 'precision': round(prec, 4), 'recall': round(rec, 4),
            'f1': round(2 * prec * rec / max(1e-9, prec + rec), 4), 'yes_ratio': round(float(pred.mean()), 4),
            'auc': round(float(roc_auc_score(y, p_yes)), 4),
            'acc_adversarial_neg': round(float((pred[types == 'absent (adversarial)'] == 0).mean()), 4),
            'acc_random_neg': round(float((pred[types == 'absent (random)'] == 0).mean()), 4)}


def evaluate(model, proc, tag, sub, run_dir):
    qs = pope_questions(sub['eval'])
    pope_rows = []
    for kind in DEGRADATIONS:
        safe = re.sub(r'[^a-z0-9]+', '_', kind.lower()).strip('_')
        p = pope(model, proc, qs, kind, os.path.join(run_dir, f'{tag}_pope_{safe}.json'))
        pope_rows.append({'model': tag, 'condition': kind, **pope_metrics(qs, p)})
        print(f'  [{tag}] POPE {kind:20s} acc {pope_rows[-1]["accuracy"]:.4f}  F1 {pope_rows[-1]["f1"]:.4f}  yes {pope_rows[-1]["yes_ratio"]:.3f}')
    texts = describe(model, proc, [r['file_name'] for r in sub['eval']], os.path.join(run_dir, f'{tag}_eval_desc.json'))
    texts = [complete_sentences(t) for t in texts]
    ch, per = chair(texts, [r['objects'] for r in sub['eval']], [r['salient'] for r in sub['eval']])
    ch.update({'model': tag, 'avg_words': round(float(np.mean([len(t.split()) for t in texts])), 1)})
    return pope_rows, ch, texts, per


def run_partC_vlm():
    run_dir = out_path('partC_vlm_runs')
    os.makedirs(run_dir, exist_ok=True)
    adapter_dir = ckpt_path('partC_vlm_dpo_lora')
    sub = load_json(os.path.join(DATA_DIR, 'partC_coco_subset.json'))
    for r in sub['train'] + sub['eval']:                 # download / cache the 230 COCO images once
        coco_image(r['file_name'])

    model, proc = load_vlm(VLMS[VLM_NAME], max_pixels=MAX_PIXELS)
    base_train_desc = describe(model, proc, [r['file_name'] for r in sub['train']], os.path.join(run_dir, 'base_train_desc.json'))
    base_pope, base_chair, base_texts, base_per = evaluate(model, proc, 'Base', sub, run_dir)

    pairs, stats = build_pairs(sub['train'], base_train_desc)
    save_json(pairs, out_path('partC_vlm_dpo_pairs.json'))
    print(f'[Part C/VLM] {len(pairs)} preference pairs: {stats}')
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pairs))
    val_pairs, train_pairs = [pairs[i] for i in idx[:N_VAL]], [pairs[i] for i in idx[N_VAL:]]
    encode = dpo.vlm_encoder(model, proc, coco_image)
    if dpo.trained(adapter_dir):
        pm, hist = dpo.load_lora(model, adapter_dir), dpo.load_history(adapter_dir)
        print(f'[Part C/VLM] loaded trained adapter from {adapter_dir}')
    else:
        pm = dpo.add_lora(model, r=LORA_R)
        hist = dpo.train_dpo(pm, train_pairs, val_pairs, encode, adapter_dir, **DPO_CFG)
    dpo_pope, dpo_chair, dpo_texts, dpo_per = evaluate(pm, proc, 'DPO', sub, run_dir)
    del pm, model; free_gpu()

    pope_tbl = pd.DataFrame(base_pope + dpo_pope)
    chair_tbl = pd.DataFrame([base_chair, dpo_chair])[['model', 'CHAIR_s', 'CHAIR_i', 'recall', 'objects_per_caption', 'avg_words']]
    examples = pd.DataFrame([{'file_name': r['file_name'], 'objects': ', '.join(r['objects']),
                              'base_hallucinated': ', '.join(b['hallucinated']), 'dpo_hallucinated': ', '.join(d['hallucinated']),
                              'base_description': bt, 'dpo_description': dt}
                             for r, b, d, bt, dt in zip(sub['eval'], base_per, dpo_per, base_texts, dpo_texts)])
    pope_tbl.to_csv(out_path('partC_vlm_pope.csv'), index=False, encoding='utf-8-sig')
    chair_tbl.to_csv(out_path('partC_vlm_chair.csv'), index=False, encoding='utf-8-sig')
    examples.to_csv(out_path('partC_vlm_descriptions.csv'), index=False, encoding='utf-8-sig')
    save_json({'gpu': gpu_info(), 'model': VLMS[VLM_NAME], 'max_pixels': MAX_PIXELS, 'pairs': len(pairs), 'pair_stats': stats,
               'n_train': len(train_pairs), 'n_val': len(val_pairs), 'dpo': DPO_CFG, 'lora_r': LORA_R,
               'train_seconds': hist.get('train_seconds'), 'best_epoch': hist.get('best_epoch')}, out_path('partC_vlm_meta.json'))
    from .partC_dpo_llm import plot_history
    figs = {'history': plot_history(hist, 'Part C advanced: DPO on Qwen3-VL-2B (object hallucination)', out_path('partC_vlm_training.png'))}
    figs.update(plot_partC_vlm(pope_tbl, chair_tbl, sub, base_texts, dpo_texts, base_per, dpo_per))
    return {'pope': pope_tbl, 'chair': chair_tbl, 'examples': examples, 'history': hist, 'pair_stats': stats, 'figures': figs}


def plot_partC_vlm(pope_tbl, chair_tbl, sub, base_texts, dpo_texts, base_per, dpo_per):
    figs = {}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    orig = pope_tbl[pope_tbl.condition == 'original']
    keys = ['accuracy', 'f1', 'acc_adversarial_neg', 'acc_random_neg', 'yes_ratio']
    x = np.arange(len(keys)); w = 0.38
    for j, (_, r) in enumerate(orig.iterrows()):
        ax[0].bar(x + (j - 0.5) * w, [r[k] for k in keys], w, label=r.model)
        for xi, k in zip(x, keys):
            ax[0].text(xi + (j - 0.5) * w, r[k], f'{r[k]:.3f}', ha='center', va='bottom', fontsize=7)
    ax[0].set_xticks(x, ['Acc', 'F1', 'Acc adv. neg', 'Acc rand. neg', 'Yes ratio'], fontsize=8)
    ax[0].set_ylim(0, 1.1); ax[0].legend(); ax[0].set_title('POPE (original images, 480 questions)')
    keys = ['CHAIR_s', 'CHAIR_i', 'recall']
    for j, (_, r) in enumerate(chair_tbl.iterrows()):
        ax[1].bar(x[:3] + (j - 0.5) * w, [r[k] for k in keys], w, label=r.model)
        for xi, k in zip(x[:3], keys):
            ax[1].text(xi + (j - 0.5) * w, r[k], f'{r[k]:.3f}', ha='center', va='bottom', fontsize=8)
    ax[1].set_xticks(x[:3], ['CHAIR_s (lower better)', 'CHAIR_i (lower better)', 'Object recall'], fontsize=8)
    ax[1].legend(); ax[1].set_title('Descriptions of 80 held-out images')
    for m, g in pope_tbl.groupby('model', sort=False):
        ax[2].plot(range(len(g)), g.accuracy, 'o-', label=m)
    ax[2].set_xticks(range(len(DEGRADATIONS)), DEGRADATIONS, rotation=35, ha='right', fontsize=8)
    ax[2].set_ylabel('POPE accuracy'); ax[2].legend(); ax[2].set_title('Robustness to image degradations')
    fig.tight_layout(); fig.savefig(out_path('partC_vlm_results.png'), dpi=150); figs['results'] = fig

    ex = sub['eval'][0]
    im = coco_image(ex['file_name'])
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
    for a in ax.ravel():
        a.axis('off')
    for a, kind in zip(ax.ravel(), DEGRADATIONS):
        a.imshow(degrade(im, kind)); a.set_title(kind)
    fig.tight_layout(); fig.savefig(out_path('partC_vlm_degradations.png'), dpi=110); figs['degradations'] = fig

    import textwrap
    idx = ([i for i, b in enumerate(base_per) if b['hallucinated']] + [i for i, b in enumerate(base_per) if not b['hallucinated']])[:4]
    fig, ax = plt.subplots(1, len(idx), figsize=(5 * len(idx), 9))
    for a, i in zip(np.atleast_1d(ax), idx):
        r = sub['eval'][i]
        a.imshow(coco_image(r['file_name'])); a.axis('off')
        a.set_title(textwrap.fill(f'GT objects: {", ".join(r["objects"])}', 55) + '\n\n' +
                    textwrap.fill(f'BASE (halluc.: {", ".join(base_per[i]["hallucinated"]) or "-"}): {textwrap.shorten(base_texts[i], 300)}', 55) + '\n\n' +
                    textwrap.fill(f'DPO (halluc.: {", ".join(dpo_per[i]["hallucinated"]) or "-"}): {textwrap.shorten(dpo_texts[i], 300)}', 55),
                    fontsize=7, loc='left')
    fig.tight_layout(); fig.savefig(out_path('partC_vlm_examples.png'), dpi=110); figs['examples'] = fig
    return figs
