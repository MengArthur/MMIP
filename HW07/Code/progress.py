"""Which stages are finished, interrupted (resumable) or not started - shown at the top of the notebook."""

import json
import os

import pandas as pd

from . import experiments as E
from .common import out_path

HW3 = {'hw3_demo_transcript_v2.json': ('作業三 示範對話', 9), 'hw3_topic_results_v2.json': ('作業三 UX 話題銜接', 24),
       'hw3_false_positive_results_v2.json': ('作業三 誤判率', 50), 'hw3_offtopic_results_v2.json': ('作業三 無脈絡問題', 32),
       'hw3_attack_results_v2.json': ('作業三 攻擊', 56)}


def _training(d):
    if os.path.exists(os.path.join(d, 'history.json')):
        return '完成'
    p = os.path.join(d, 'resume.pt')
    if os.path.exists(p):
        import torch
        st = torch.load(p, map_location='cpu', weights_only=False)
        return f'中斷：epoch {st["epoch"]}、已完成 {st["step"]} 步，重新執行會接續'
    return '未開始'


def status():
    rows = [('訓練 B SFT', _training(E.SFT_DIR)), ('訓練 B SFT+DPO', _training(E.DPO_DIR)),
            ('訓練 D System One v2', _training(E.S1_DIR))]
    for m in E.METHODS:
        part = out_path(f'pred_{m}.partial.json')
        if E.done(m):
            s = '完成'
        elif os.path.exists(part):
            with open(part, encoding='utf-8') as f:
                s = f'中斷：已完成 {len(json.load(f)["records"])}/52 題，重新執行會接續'
        else:
            s = '未開始'
        rows.append((f'評估 {E.LABEL[m]}', s))
    for f, (name, total) in HW3.items():
        p = out_path(f)
        if not os.path.exists(p):
            rows.append((name, '未開始'))
            continue
        with open(p, encoding='utf-8') as fh:
            n = len(json.load(fh))
        rows.append((name, '完成' if f.startswith('hw3_demo') or n >= total else f'進行中：{n}/{total} 段對話'))
    return pd.DataFrame(rows, columns=['階段', '狀態'])
