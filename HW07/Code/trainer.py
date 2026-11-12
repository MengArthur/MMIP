"""
Resumable training loop shared by SFT, DPO and System One (Colab free sessions disconnect).

Every `save_every` optimizer steps the trainable weights, optimizer, LR schedule, AMP scaler, history and the
position inside the epoch are written to <out_dir>/resume.pt (atomically). The sample order of every epoch is a
fixed permutation (seed + epoch), so a restarted run continues with exactly the samples it had not seen yet; at
most `save_every` updates are lost. The best epoch's weights are kept in resume_best.pt. Both files are deleted
once training has finished and the adapter + history.json are saved.
"""

import math
import os
import time

import numpy as np
import torch


class Store:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.state, self.best = os.path.join(out_dir, 'resume.pt'), os.path.join(out_dir, 'resume_best.pt')

    @staticmethod
    def save(obj, path):
        tmp = path + '.tmp'
        torch.save(obj, tmp)
        os.replace(tmp, path)            # a disconnect during saving never leaves a broken file

    @staticmethod
    def load(path):
        return torch.load(path, map_location='cpu', weights_only=False) if os.path.exists(path) else None

    def clear(self):
        for p in (self.state, self.best):
            if os.path.exists(p):
                os.remove(p)


def _cpu(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def fit(out_dir, n, epochs, step_fn, params, opt, scaler, eval_fn, get_state, set_state, hist, train_mode,
        grad_accum=8, warmup_ratio=0.05, max_grad_norm=1.0, seed=42, save_every=25, log_every=10, tag='train'):
    """step_fn(i) -> (loss tensor, {metric: float} incl. 'loss') for training item i.
    eval_fn() -> {metric: float} incl. 'loss' (validation). get_state() / set_state(sd): trainable weights.
    Keeps the epoch with the lowest validation loss. Returns the history (hist is filled in place)."""
    total = math.ceil(n / grad_accum) * epochs
    warm = max(1, int(warmup_ratio * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    store = Store(out_dir)
    st = store.load(store.state)
    if st is not None:
        set_state(st['weights'])
        opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); scaler.load_state_dict(st['scaler'])
        hist.clear(); hist.update(st['hist'])
        ep0, k0, step, best, run, elapsed = st['epoch'], st['k'], st['step'], st['best'], st['run'], st['elapsed']
        print(f'[{tag}] resumed: epoch {ep0}, sample {k0}/{n}, step {step}/{total} ({elapsed / 60:.1f} min already trained)')
    else:
        v = eval_fn()
        hist['epochs_log'].append({'epoch': 0, **{f'val_{k}': float(x) for k, x in v.items()}})
        print(f'[{tag}] epoch 0  val loss {v["loss"]:.4f}')
        ep0, k0, step, best, run, elapsed = 1, 0, 0, float('inf'), {}, 0.0
    t0 = time.perf_counter() - elapsed

    def snapshot(epoch, k):
        store.save({'weights': _cpu(get_state()), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'scaler': scaler.state_dict(), 'hist': hist, 'epoch': epoch, 'k': k, 'step': step, 'best': best,
                    'run': run, 'elapsed': time.perf_counter() - t0}, store.state)

    for ep in range(ep0, epochs + 1):
        order = np.random.default_rng(seed + ep).permutation(n)
        start = k0 if ep == ep0 else 0
        if start == 0:
            run = {}                                   # per-epoch metric lists
        train_mode()
        for k in range(start, n):
            loss, metrics = step_fn(int(order[k]))
            scaler.scale(loss / grad_accum).backward()
            for key, x in metrics.items():
                run.setdefault(key, []).append(float(x))
            if (k + 1) % grad_accum == 0 or k == n - 1:
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                m = k % grad_accum + 1                 # samples in this update
                rec = {'step': step, 'epoch': ep, **{key: float(np.mean(x[-m:])) for key, x in run.items()},
                       'grad_norm': float(gn), 'lr': sched.get_last_lr()[0], 'seconds': round(time.perf_counter() - t0, 1)}
                hist['steps'].append(rec)
                if step % log_every == 0:
                    extra = '  '.join(f'{key} {rec[key]:+.3f}' for key in run if key != 'loss')
                    print(f'  step {step:4d}/{total}  loss {rec["loss"]:.4f}  {extra}  {rec["seconds"]:.0f}s')
                if step % save_every == 0 and k < n - 1:
                    snapshot(ep, k + 1)
        v = eval_fn()
        hist['epochs_log'].append({'epoch': ep, 'train_loss': float(np.mean(run['loss'])),
                                   **{f'val_{key}': float(x) for key, x in v.items()}})
        print(f'[{tag}] epoch {ep}  train loss {np.mean(run["loss"]):.4f}  ' +
              '  '.join(f'val {key} {float(x):.4f}' for key, x in v.items()))
        if v['loss'] < best:
            best, hist['best_epoch'] = v['loss'], ep
            store.save(_cpu(get_state()), store.best)
        snapshot(ep + 1, 0)
    set_state(store.load(store.best))
    hist['train_seconds'] = round(time.perf_counter() - t0, 1)
    return hist, store
