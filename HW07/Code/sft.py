"""
Experiment B: teach Qwen3-VL-2B the routing spec by training (lecture section 2, slides 14-26).

  B-SFT      LoRA supervised fine-tuning on the 720 training dialogues; target = the canonical compact JSON
  B-SFT+DPO  continue from the SFT adapter with DPO on preference pairs (slides 16, 23-26): chosen = the gold
             JSON, rejected = a broken version of it - either a format error from the slides (trailing comma,
             single quotes, missing comma, Markdown fence, Python None, extra text) or a content error that the
             rules forbid (route although information is missing, a confusable department, an order id carried
             over from the previous topic, an attack treated as a normal request).
The DPO loss includes the SFT term (RPO, sft_alpha=1) because plain DPO on pairs that differ in a few tokens
lowered the probability of the chosen answers too in HW06 (likelihood displacement).
"""

import json
import os
import random

import numpy as np
import torch

from . import dpo, schema, trainer
from .common import save_json
from .routers import SYSTEM_SHORT
from .vlm import encode, to_device


def sft_loss(model, enc, ctx):
    n = enc.pop('n_resp')
    with ctx:
        lp = dpo.sequence_logp(model, enc, n)
    return -lp / n


def _batch(model, proc, row):
    enc = encode(proc, row['turns'], SYSTEM_SHORT, None, response=schema.to_json(row['label']))
    n = enc['n_resp']
    enc = to_device({k: v for k, v in enc.items() if k != 'n_resp'}, model)
    enc['n_resp'] = n
    return enc


@torch.no_grad()
def val_loss(model, proc, rows):
    ctx, _ = dpo._autocast(model)
    model.eval()
    return float(np.mean([sft_loss(model, _batch(model, proc, r), ctx).item() for r in rows]))


def train_sft(model, proc, train_rows, val_rows, out_dir, epochs=2, lr=1e-4, grad_accum=8, warmup_ratio=0.05,
              max_grad_norm=1.0, seed=42, log_every=10, save_every=25):
    """model: PEFT model with LoRA. Keeps the epoch with the lowest validation loss; writes history.json.
    Resumable: a disconnected run continues from Checkpoints/<out_dir>/resume.pt (trainer.fit)."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    hist = {'lr': lr, 'epochs': epochs, 'grad_accum': grad_accum, 'n_train': len(train_rows), 'n_val': len(val_rows),
            'steps': [], 'epochs_log': []}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    ctx, use_scaler = dpo._autocast(model)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    def step(i):
        loss = sft_loss(model, _batch(model, proc, train_rows[i]), ctx)
        return loss, {'loss': loss.item()}

    hist, store = trainer.fit(out_dir, len(train_rows), epochs, step, params, opt, scaler,
                              lambda: {'loss': val_loss(model, proc, val_rows)},
                              lambda: get_peft_model_state_dict(model), lambda sd: set_peft_model_state_dict(model, sd),
                              hist, model.train, grad_accum, warmup_ratio, max_grad_norm, seed, save_every, log_every, 'sft')
    model.save_pretrained(out_dir)
    save_json(hist, os.path.join(out_dir, 'history.json'))
    store.clear()
    model.eval()
    model.config.use_cache = True
    return hist


# ----------------------------------------------------------------------------- DPO preference pairs
CONFUSABLE = {'return_refund': 'repair_warranty', 'repair_warranty': 'tech_support', 'tech_support': 'repair_warranty',
              'order_shipping': 'return_refund', 'billing_invoice': 'return_refund', 'account_security': 'tech_support',
              'presales': 'tech_support'}


def _format_error(js, rng):
    kind = rng.choice(['trailing_comma', 'single_quotes', 'missing_comma', 'markdown', 'python_none', 'extra_text'])
    if kind == 'trailing_comma':
        return js[:-1] + ',}'
    if kind == 'single_quotes':
        return js.replace('"', "'")
    if kind == 'missing_comma':
        i = js.index(',"urgency"')
        return js[:i] + js[i + 1:]
    if kind == 'markdown':
        return '```json\n' + js + '\n```'
    if kind == 'python_none':
        return js.replace('null', 'None') if 'null' in js else js[:-1] + ',}'
    return js + '\n以上是分流結果。'


def _content_error(row, rng):
    lab = dict(schema.canonical(row['label']))
    cat, st = row['category'], lab['status']
    if st == 'clarify' and lab['missing_info']:                     # route although information is missing
        lab.update(status='route', missing_info=[], department=lab['department'] if lab['department'] != 'none' else 'tech_support')
    elif st in ('reject', 'off_topic'):                             # attack / chit-chat treated as a request
        lab.update(status='route', department=rng.choice(['return_refund', 'presales']), urgency='normal')
    elif cat.startswith('switch_new') and lab['order_id'] is None:  # order id carried over from the first topic
        first = row['turns'][0]['text']
        m = schema.ORDER_ID_RE.search(first)
        if not m:
            return None
        lab.update(order_id=m.group(0), status='route', missing_info=[])
    elif lab['department'] in CONFUSABLE:
        lab['department'] = CONFUSABLE[lab['department']]
    else:
        return None
    return schema.to_json(lab)


def make_pairs(rows, n, seed=0):
    """Half format errors, half content errors (when the sample allows one)."""
    rng = random.Random(seed)
    pairs = []
    for row in rng.sample(rows, min(n, len(rows))):
        chosen = schema.to_json(row['label'])
        rejected = _content_error(row, rng) if rng.random() < 0.5 else None
        kind = 'content'
        if rejected is None or rejected == chosen:
            rejected, kind = _format_error(chosen, rng), 'format'
        if rejected != chosen:
            pairs.append({'id': row['id'], 'turns': row['turns'], 'chosen': chosen, 'rejected': rejected, 'kind': kind})
    return pairs


def dialogue_encoder(model, proc):
    """encode(pair, 'chosen'|'rejected') for dpo.train_dpo."""
    def enc(pair, which):
        e = encode(proc, pair['turns'], SYSTEM_SHORT, None, response=pair[which])
        n = e['n_resp']
        return to_device({k: v for k, v in e.items() if k != 'n_resp'}, model), n
    return enc


def dumps(pairs):
    return json.dumps(pairs, ensure_ascii=False, indent=1)
