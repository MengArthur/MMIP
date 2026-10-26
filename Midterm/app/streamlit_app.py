"""
RouteVision demo: upload (or pick) a placement layout, compare discriminative
methods for routing-congestion hotspot prediction, and inspect Grad-CAM.

    streamlit run app/streamlit_app.py
"""

import os
import sys

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import CAM_MODELS, CHANNELS, METHODS, BadLayout, Engine, load_demos, parse_npz
from routevision import dataset as D

INFO = {m: desc for m, (_, desc) in METHODS.items()}

st.set_page_config(page_title='RouteVision', layout='wide')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Noto Sans CJK TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@st.cache_resource
def load_engine():
    try:
        return Engine()
    except FileNotFoundError:
        return None


def heat(ax, img, title, cmap='inferno', vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax); ax.set_title(title, fontsize=9); ax.axis('off')


# ---------------------------------------------------------------- UI
st.title('RouteVision｜晶片版圖繞線壅塞預測')
st.caption('上傳 placement 後的版圖特徵圖，比較不同鑑別式方法預測的繞線壅塞熱點（CircuitNet-N14）')

engine = load_engine()
if engine is None:
    st.error('找不到 checkpoints/。請先在 Colab 執行 notebooks/RouteVision_train.ipynb，'
             '把 RouteVision_results.zip 解壓到專案根目錄。')
    st.stop()
thr, models = engine.thr, engine.models

with st.sidebar:
    st.header('1. 選擇版圖')
    src = st.radio('來源', ['內建範例', '上傳 .npz'], label_visibility='collapsed')
    layout, has_label, has_egr, name = None, False, False, ''
    if src.startswith('內建'):
        demos = load_demos()
        if demos:
            k = st.selectbox('範例', range(len(demos)), format_func=lambda i: f'{demos[i][2]}｜{demos[i][0]}')
            name, layout, _ = demos[k]
            has_label = has_egr = True
        else:
            st.warning('找不到 demo/demo_layouts.npz')
    else:
        f = st.file_uploader('單一版圖 .npz（X: 4×256×256、params: 6；label、egr 可選）', type=['npz'])
        if f:
            try:
                layout, has_label, has_egr = parse_npz(f.getvalue())
                name = f.name
            except BadLayout as e:
                st.error(str(e))
    st.header('2. 選擇方法')
    store_methods = list(INFO)
    method = st.selectbox('方法', store_methods, index=store_methods.index('CNN') if 'CNN' in models else 0)
    st.caption(INFO[method])

if layout is None:
    st.info('請在左側選擇或上傳一個版圖。')
    st.stop()

store = engine.store(layout, has_egr)
cache = st.session_state.setdefault('probs_cache', {})
if name not in cache:                       # every method runs once per layout; switching methods is instant
    with st.spinner('所有方法推論中（第一次約 10–20 秒，之後切換方法不需重算）…'):
        cache[name] = engine.predict_all(store)
probs = cache[name]
if method not in probs:
    st.warning(f'{method} 的模型檔不在 checkpoints/ 中，請改選其他方法。')
    st.stop()

st.subheader(f'版圖：{name}')
fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
for a, (c, t) in zip(ax, enumerate(CHANNELS)):
    heat(a, store.X[0, c], t)
st.pyplot(fig)

p = probs[method]
t = thr.get(method, 0.5)
pred = p >= t
cols = st.columns(3)
with cols[0]:
    fig, a = plt.subplots(figsize=(4, 4))
    heat(a, np.kron(p, np.ones((D.PATCH, D.PATCH))), f'{method}: hotspot score', cmap='Reds')
    st.pyplot(fig)
with cols[1]:
    fig, a = plt.subplots(figsize=(4, 4))
    a.imshow(store.X[0, 1], cmap='gray')
    a.imshow(np.kron(pred, np.ones((D.PATCH, D.PATCH))), cmap='Reds', alpha=0.45, vmin=0, vmax=1)
    a.set_title(f'predicted hotspots (threshold {t:.2f})', fontsize=9); a.axis('off')
    st.pyplot(fig)
with cols[2]:
    if has_label:
        fig, a = plt.subplots(figsize=(4, 4))
        heat(a, np.kron(store.y[0], np.ones((D.PATCH, D.PATCH))), 'ground truth hotspots', cmap='Reds', vmin=0, vmax=1)
        st.pyplot(fig)
    else:
        st.info('上傳的檔案沒有 label，無法顯示真實熱點。')

if has_label:
    rows = [{'方法': r['method'], '命中熱點': f"{r['tp']}/{r['n_true']}", '誤報': r['fp'], 'F1（此版圖）': r['f1']}
            for r in engine.compare(probs, store.y[0])]
    st.subheader('各方法在這個版圖上的比較')
    st.dataframe(rows, hide_index=True, width='stretch')
    st.caption('完整的測試集指標（ROC-AUC、Macro-F1 等）請見 outputs/test_results.csv。')

if method in CAM_MODELS and method in models:
    st.subheader('Grad-CAM：模型根據哪些區域判斷')
    with st.spinner('計算 Grad-CAM…'):
        cam = engine.gradcam(method, store)
    fig, a = plt.subplots(1, 2, figsize=(9, 4.2))
    heat(a[0], store.X[0, 0], 'cell density', cmap='gray')
    a[1].imshow(store.X[0, 0], cmap='gray'); a[1].imshow(cam, cmap='jet', alpha=0.45)
    a[1].set_title('Grad-CAM (weighted by predicted hotspot probability)', fontsize=9); a[1].axis('off')
    st.pyplot(fig)
