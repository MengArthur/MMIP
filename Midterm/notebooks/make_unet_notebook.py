"""Generates notebooks/RouteVision_unet.ipynb (run once after editing)."""
import json
import os

cells = []
md = lambda s: cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')})
code = lambda s: cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')})

md('''
# RouteVision：U-Net 逐像素預測繞線使用率圖

前面的方法都是「8×8 區塊，熱點或不是」。這份 notebook 加入**逐像素預測**：U-Net 直接預測 routing 後整張 256×256 的繞線使用率圖（監督式回歸，仍是鑑別式模型），再取每個區塊的平均當熱點分數，和其他方法放在同一張表比較。

- **模型**：U-Net，編碼器為 ImageNet 預訓練的 ResNet-18（輸入改成 4 通道），約 14 M 參數。
- **兩個版本（消融）**：`U-Net`（只用影像）、`U-Net (multimodal)`（設計參數以 FiLM 調整瓶頸層特徵）。
- **評估**：和其他方法一樣三個設計輪流當測試。像素層級：Pearson、SSIM、NRMSE，對照 eGR（原始值，同單位）與用訓練設計線性校正的 eGR、RUDY。區塊層級：ROC-AUC、F1 等，閾值在驗證集決定。
- **事先定好的規則**：網頁使用三設計平均像素 Pearson 較高的版本。

### 執行步驟（Colab，T4 GPU，約 20–30 分鐘）
1. 把 `RouteVision_v3.zip` 上傳到 Google Drive 的 `MyDrive/`（和之前放同一個地方）。
2. 「執行階段 → 變更執行階段類型」選 **T4 GPU**。
3. 「執行階段 → 全部執行」。模型存在 Drive 的 `MyDrive/RouteVision/checkpoints_unet/`，斷線後重跑會從中斷處繼續。
4. 最後一格會下載 `RouteVision_results_unet.zip`（約 120 MB），把它交給本機專案（Claude 會解壓並匯出給網頁）。
''')

code('''
# 環境設定：掛載 Drive；解壓新版程式（覆蓋程式碼，但不覆蓋已存在的資料與模型）
import os, sys, json
from google.colab import drive
drive.mount('/content/drive')
PROJECT = '/content/drive/MyDrive/RouteVision'
ZIP = '/content/drive/MyDrive/RouteVision_v3.zip'
!unzip -q -n {ZIP} -d /content/drive/MyDrive/
!unzip -q -o {ZIP} 'RouteVision/routevision/*' 'RouteVision/notebooks/*' -d /content/drive/MyDrive/
os.chdir(PROJECT); sys.path.insert(0, PROJECT)
%matplotlib inline
import torch
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU（請改用 T4 GPU）')
''')

md('''
## 1. 三個設計輪流測試 × 兩個 U-Net 版本
每一輪：在其餘兩個設計上訓練（85% 訓練、15% 驗證，和其他方法完全相同的切分），以驗證集 MSE 最低的 epoch 為準，只在測試設計上評估。
''')

code('''
from routevision import pixel as P
import pandas as pd
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
DESIGNS = ['Vortex-small', 'nvdla-small', 'openc910-1']
patch, pixel, chosen = P.cross_design('data/processed', DESIGNS)
print(json.dumps(chosen, indent=1, ensure_ascii=False))
''')

md('''
## 2. 結果
- 像素層級：各設計當測試時的 Pearson，以及三設計的平均 ± 標準差。
- 區塊層級：和其他方法同一套指標；`layouts_beating_eGR` 為逐張版圖 F1 贏過 eGR 的數量（3 個設計共 288 張）。
''')

code('''
px, pa = P.summarize(patch, pixel)
display(pixel.pivot_table(index='method', columns='test_design', values='Pearson', sort=False).round(4))
display(px)
display(pa)
''')

code('''
import matplotlib.pyplot as plt
from IPython.display import Image, display as show
for d in DESIGNS:
    print(d); show(Image(f'outputs/unet/{d}/maps.png'))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for d in DESIGNS:
    for name, stem in (('U-Net', 'UNet'), ('U-Net (multimodal)', 'UNet_multimodal')):
        h = json.load(open(f'checkpoints_unet/{d}/{stem}.json'))
        ax[0].plot(h['train_mse'], label=f'{d} {name}'); ax[1].plot(h['val_pearson'], label=f'{d} {name}')
ax[0].set(title='Training MSE', xlabel='Epoch', yscale='log'); ax[1].set(title='Validation pixel Pearson', xlabel='Epoch')
for a in ax: a.legend(fontsize=7); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('outputs/unet/training_curves.png', dpi=150); plt.show()
''')

md('## 3. 打包下載（結果與網頁要用的 Vortex-small 那一輪模型）')

code('''
!cd {PROJECT} && rm -f /content/RouteVision_results_unet.zip && zip -q -r /content/RouteVision_results_unet.zip outputs/unet checkpoints_unet/Vortex-small
from google.colab import files
files.download('/content/RouteVision_results_unet.zip')
''')

for i, c in enumerate(cells):
    c['id'] = f'cell-{i}'
    lines = c['source'].split('\n')
    c['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
      'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RouteVision_unet.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', out)
