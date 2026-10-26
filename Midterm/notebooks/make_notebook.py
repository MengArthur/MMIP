"""Generates notebooks/RouteVision_train.ipynb (run once after editing)."""
import json
import os

cells = []
md = lambda s: cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')})
code = lambda s: cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')})

md('''
# RouteVision：晶片版圖繞線壅塞熱點預測（期中 Milestone 訓練 Notebook）

**任務**：輸入 placement 後的版圖特徵圖（cell 密度、RUDY、腳位 RUDY、macro），預測每個 32×32 區塊在最終 global routing 後是否為壅塞熱點（區塊平均繞線使用率 ≥ 0.6）。

**比較的方法**：EDA 粗估（RUDY、eGR）、OpenCV 特徵 + Random Forest、MLP、CNN、多模態（影像 + 設計參數）、ViT。

**切分方式**：依「設計」切分，測試設計完全不參與訓練，用來檢驗模型能不能用在沒看過的晶片上。

### 執行步驟（Colab，T4 GPU，約 30–40 分鐘）
1. 把 `RouteVision.zip` 上傳到 Google Drive 的 `MyDrive/`。
2. 「執行階段 → 變更執行階段類型」選 **T4 GPU**。
3. 「執行階段 → 全部執行」。訓練好的模型存在 Drive 的 `MyDrive/RouteVision/checkpoints/`，斷線後重跑會直接讀取，不會重新訓練。
4. 最後一格會下載 `RouteVision_results.zip`（模型、閾值、結果圖表、demo 範例）；解壓到本機專案資料夾後即可執行網頁 demo。
''')

code('''
# 環境設定：掛載 Drive、解壓專案（只在第一次解壓）
import os, sys
from google.colab import drive
drive.mount('/content/drive')
PROJECT = '/content/drive/MyDrive/RouteVision'
if not os.path.exists(f'{PROJECT}/routevision'):
    !unzip -q -o /content/drive/MyDrive/RouteVision.zip -d /content/drive/MyDrive/
os.chdir(PROJECT); sys.path.insert(0, PROJECT)
%matplotlib inline
import torch
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU（請改用 T4 GPU）')
''')

md('''
## 1. 資料：CircuitNet-N14（前處理後的檔案）
`data/processed/*.npz` 已包含在專案 zip 中。若檔案不存在，下一格會從 Hugging Face 下載原始資料並自動前處理（每個設計約 5–15 分鐘）。
''')

code('''
DESIGNS = ['Vortex-small', 'nvdla-small', 'openc910-1']
TEST_DESIGN = 'Vortex-small'          # 完全不參與訓練的設計
missing = [d for d in DESIGNS if not os.path.exists(f'data/processed/{d}.npz')]
if missing:
    !python -m routevision.prepare --designs {' '.join(missing)} --raw_dir /content/raw --out_dir data/processed
from routevision import experiments as E
data, split, norm, stores = E.prepare('data/processed', DESIGNS, TEST_DESIGN, ckpt_dir='checkpoints')
''')

md('''
## 2. 訓練與評估所有方法
第一次執行會訓練（MLP、CNN、多模態各約 3–5 分鐘，ViT 約 10–15 分鐘）；之後直接讀取 `checkpoints/`。閾值一律在 validation set 上決定，test set 只用來評估。
''')

code('''
res = E.run(stores, ckpt_dir='checkpoints', out_dir='outputs', methods=('MLP', 'CNN', 'Multimodal', 'ViT'))
res['table']
''')

md('## 3. 結果圖表')

code('''
import matplotlib.pyplot as plt
yt = stores['test'].flat('y')
E.plot_curves(yt, res['scores'], 'outputs/roc_pr_curves.png'); plt.show()
E.plot_history(res['history'], 'outputs/training_curves.png'); plt.show()
show = [0, len(stores['test'].items) // 2, len(stores['test'].items) - 1]
E.plot_layouts(stores['test'], res['scores'], res['thresholds'], show, 'outputs/hotspot_maps.png'); plt.show()
E.plot_gradcam(res['models']['CNN'], stores['test'], show, 'outputs/gradcam_cnn.png', 'CNN'); plt.show()
E.plot_gradcam(res['models']['Multimodal'], stores['test'], show, 'outputs/gradcam_multimodal.png', 'Multimodal'); plt.show()
''')

md('## 4. 打包下載（模型、閾值、圖表、demo 範例）')

code('''
os.makedirs('demo', exist_ok=True)
E.export_demo(data, split, 'demo/demo_layouts.npz')
!cd {PROJECT} && rm -f /content/RouteVision_results.zip && zip -q -r /content/RouteVision_results.zip checkpoints outputs demo
from google.colab import files
files.download('/content/RouteVision_results.zip')
''')

for i, c in enumerate(cells):
    c['id'] = f'cell-{i}'
    lines = c['source'].split('\n')
    c['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
      'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RouteVision_train.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', out)
