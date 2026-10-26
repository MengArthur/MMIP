"""Generates notebooks/RouteVision_improve.ipynb (run once after editing)."""
import json
import os

cells = []
md = lambda s: cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')})
code = lambda s: cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')})

md('''
# RouteVision：改進實驗（輪流測試 + 資料增強 + 集成）

第一版只用 Vortex-small 當測試設計。這份 notebook 做三件事：

1. **Leave-one-design-out**：Vortex-small、nvdla-small、openc910-1 輪流當「完全沒參與訓練」的測試設計，回答「只測一個設計可信嗎」。
2. **資料增強**：訓練時隨機左右、上下鏡像整張版圖（區塊位置特徵跟著鏡像；標註不變）。每個測試設計都跑「無增強」與「有增強」兩組，直接比較。
3. **集成（Ensemble）**：Random Forest、Multimodal、ViT 的機率取平均，不需額外訓練；閾值一樣在驗證集決定。

最後依「3 個測試設計平均」選出較好的設定，把 Vortex-small 那一輪的模型匯出給網頁 demo 使用（舊模型保留在 `checkpoints_v1/`）。

### 執行步驟（Colab，T4 GPU，約 50–60 分鐘）
1. 把 `RouteVision_v2.zip` 上傳到 Google Drive 的 `MyDrive/`（和第一版放同一個地方）。
2. 「執行階段 → 變更執行階段類型」選 **T4 GPU**。
3. 「執行階段 → 全部執行」。每一輪的模型存在 Drive 的 `MyDrive/RouteVision/checkpoints_lodo/`，斷線後重跑會從中斷處繼續。
4. 最後一格會下載 `RouteVision_results_v2.zip`；解壓到本機專案資料夾即可（會覆蓋 `checkpoints/`、`outputs/`、`demo/`）。
''')

code('''
# 環境設定：掛載 Drive；解壓新版程式（覆蓋程式碼，但不覆蓋已存在的資料與模型）
import os, sys, json, shutil
from google.colab import drive
drive.mount('/content/drive')
PROJECT = '/content/drive/MyDrive/RouteVision'
ZIP = '/content/drive/MyDrive/RouteVision_v2.zip'
!unzip -q -n {ZIP} -d /content/drive/MyDrive/
!unzip -q -o {ZIP} 'RouteVision/routevision/*' 'RouteVision/app/*' 'RouteVision/notebooks/*' 'RouteVision/tests/*' 'RouteVision/README.md' 'RouteVision/requirements.txt' -d /content/drive/MyDrive/
os.chdir(PROJECT); sys.path.insert(0, PROJECT)
%matplotlib inline
import torch
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU（請改用 T4 GPU）')
''')

md('''
## 1. 沿用第一版的模型
第一版就是「測試 Vortex-small、無增強」這一輪，程式完全相同，所以直接沿用 `checkpoints/` 的模型，省下約 10 分鐘。
只有在目標資料夾還沒有模型、而且來源確實是無增強訓練的模型時才會複製。
''')

code('''
DESIGNS = ['Vortex-small', 'nvdla-small', 'openc910-1']
CK_ROOT, OUT_ROOT = 'checkpoints_lodo', 'outputs/lodo'
src = 'checkpoints_v1' if os.path.exists('checkpoints_v1') else 'checkpoints'
dst = f'{CK_ROOT}/Vortex-small/no_aug'
os.makedirs(dst, exist_ok=True)
for m in ['MLP', 'CNN', 'Multimodal', 'ViT']:
    js = f'{src}/{m}.json'
    if os.path.exists(f'{src}/{m}.pt') and os.path.exists(js) and not os.path.exists(f'{dst}/{m}.pt') \\
            and not json.load(open(js)).get('augment', False):
        shutil.copy2(f'{src}/{m}.pt', dst); shutil.copy2(js, dst); print('reuse', m)
if os.path.exists(f'{src}/RandomForest.joblib') and not os.path.exists(f'{dst}/RandomForest.joblib'):
    shutil.copy2(f'{src}/RandomForest.joblib', dst); print('reuse RandomForest')
''')

md('''
## 2. 輪流測試 × 有無資料增強
3 個測試設計 × 2 種設定。每一輪：在其餘兩個設計上訓練（85% 訓練、15% 驗證），閾值在驗證集決定，只在測試設計上評估。
''')

code('''
from routevision import experiments as E
import pandas as pd
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
df, summary = E.cross_design('data/processed', DESIGNS, ckpt_root=CK_ROOT, out_root=OUT_ROOT)
summary
''')

md('''
## 3. 結果
- 長條：3 個測試設計的平均；黑點：各個測試設計。
- `layouts_beating_eGR`：逐張版圖比較 F1，贏過 eGR 的版圖數（3 個設計共 288 張）。
''')

code('''
import matplotlib.pyplot as plt
E.plot_cross_design(df, f'{OUT_ROOT}/cross_design_roc_auc.png', 'ROC-AUC'); plt.show()
E.plot_cross_design(df, f'{OUT_ROOT}/cross_design_macro_f1.png', 'Macro-F1'); plt.show()
df.pivot_table(index='method', columns=['setting', 'test_design'], values='ROC-AUC', sort=False).round(4)
''')

md('''
## 4. 選出設定並匯出給網頁 demo
依 3 個測試設計平均的 ROC-AUC（所有學習方法的平均）選擇有無增強，再把「測試 Vortex-small」那一輪的模型複製到 `checkpoints/`、`outputs/`。
''')

code('''
best, means = E.choose_setting(df)
print('mean ROC-AUC of the learned methods:', means, '-> use', best)
E.export_final(CK_ROOT, OUT_ROOT, 'Vortex-small', best)
json.dump({'setting': best, 'mean_roc_auc_learned': means}, open(f'{OUT_ROOT}/chosen_setting.json', 'w'), indent=1)

# 用匯出的模型重畫第一版的圖（模型已快取，只做推論）
data, split, norm, stores = E.prepare('data/processed', DESIGNS, 'Vortex-small', ckpt_dir='checkpoints')
res = E.run(stores, ckpt_dir='checkpoints', out_dir='outputs', augment=(best == 'aug'))
display(res['table'])
yt = stores['test'].flat('y')
E.plot_curves(yt, res['scores'], 'outputs/roc_pr_curves.png'); plt.show()
E.plot_history(res['history'], 'outputs/training_curves.png'); plt.show()
show = [0, len(stores['test'].items) // 2, len(stores['test'].items) - 1]
E.plot_layouts(stores['test'], res['scores'], res['thresholds'], show, 'outputs/hotspot_maps.png'); plt.show()
E.plot_gradcam(res['models']['CNN'], stores['test'], show, 'outputs/gradcam_cnn.png', 'CNN'); plt.show()
E.plot_gradcam(res['models']['Multimodal'], stores['test'], show, 'outputs/gradcam_multimodal.png', 'Multimodal'); plt.show()
''')

md('## 5. 重新挑選網頁的內建範例，打包下載')

code('''
!python app/make_demo_set.py
!cd {PROJECT} && rm -f /content/RouteVision_results_v2.zip && zip -q -r /content/RouteVision_results_v2.zip checkpoints outputs demo
from google.colab import files
files.download('/content/RouteVision_results_v2.zip')
''')

for i, c in enumerate(cells):
    c['id'] = f'cell-{i}'
    lines = c['source'].split('\n')
    c['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
      'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RouteVision_improve.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', out)
