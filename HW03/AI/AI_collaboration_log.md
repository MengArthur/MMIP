# 人工智慧工具協作紀錄 (AI Collaboration Log) — HW03

- **課程**：多模態影像資料處理 (MMIP) — Week 03 卷積神經網路 (CNN)
- **學生**：林孟霆（`315833033`）
- **協作 AI 工具**：Claude Code（Anthropic）；Google Gemini（Colab 執行流程建議）
- **規範依據**：課程簡報第 79–81 頁（Quiz 1–3）

---

## 1. AI 協作範圍

| 環節 | AI 協助內容 | 人工確認重點 |
| :--- | :--- | :--- |
| 規範解析 | 逐條列出第 79–81 頁每一小題的配分與要求，對應到程式模組與 README 章節 | 確認沒有遺漏任何小題（尤其 Top-5、Macro-AUC、參數量比較、兩個 Kernel、XAI） |
| 資料集 | 選用 CIFAR-10；官方多倫多伺服器下載速度低於 1 KB/s，改由作者群在 Hugging Face 上傳的官方版本下載 | 確認影像數量（50,000 / 10,000）與類別分佈（每類 5,000 / 1,000）與官方一致 |
| 模型設計 | 自行設計 Plain CNN（無 shortcut 的 VGG 式架構）；經典 Backbone 選用課堂介紹的 ResNet-18 | 確認 Plain CNN 沒有殘差連接，符合「Plain」定義 |
| 運算資源評估 | 本機僅有 CPU，量測後每個 epoch 約需 60–80 秒（Plain CNN）與 5 分鐘（ResNet-18），全部實驗需要數小時；因此程式改為自動偵測並使用 GPU，改在 Google Colab 的 Tesla T4 上訓練，epoch 數也提高到 Plain 30 / ResNet 15 | 確認 10 組實驗的訓練紀錄（`Checkpoints/*.json`）都記錄為 `cuda`，epoch 數完整 |
| Colab 執行方式 | Colab 上的「上傳 HW03.zip → 自動解壓縮 → 訓練完打包下載」流程參考 Google Gemini 的建議，修改了 Notebook 第一格與最後一格 | 確認修改只影響檔案路徑與下載，不影響訓練與評估邏輯 |
| 實驗設計 | 採「一次只改一個因素」的對照設計：Plain CNN 改學習率 / Dropout；ResNet-18 比較預訓練 vs 從頭訓練、凍結 vs 全微調、學習率 | 確認每組變因與基準只差一項 |
| 資料擴增 | 以張量批次實作 Random Crop（padding 4）、水平翻轉、亮度/對比擾動，不使用多行程 DataLoader，避免 Windows + Jupyter 相容問題 | 確認沒有使用會改變標籤語意的擴增（例如垂直翻轉） |
| XAI | 手寫 Grad-CAM（forward/backward hook）；Kernel 以明確規則自動挑選兩個（亮度邊緣型、色彩型），不以人工挑圖 | 對照圖片確認分析文字與實際視覺結果一致 |
| 結果檢查 | 逐張檢查 Colab 產出的圖：(1) Validation 曲線跳動屬於固定學習率下的正常現象與過擬合徵兆，並在本機以 CPU 重新評估 ResNet-18 checkpoint，Test Top-1 同樣是 0.8980，確認訓練結果可重現；(2) 原本以「權重能量」挑出的 Kernel #24 是極高頻濾波器，特徵圖看不出內容，因此改成以 Validation 影像上的反應強度挑選（選到水平邊緣 kernel #26），只重畫這張圖、不重新訓練 | 確認新 Kernel 的權重方向與特徵圖反應一致，README 與 Notebook 說明同步更新 |

---

## 2. 誠信與品質控管原則

1. **所有數據皆來自實際執行**：README 表格中的每個數字皆來自 `HW03_main.py` 產生的 `Output/*.csv`；Notebook 讀取同一批 checkpoint，兩者數字一致。
2. **不寫死結論**：模型優劣、擴增是否有效等結論，皆依實際結果撰寫；若擴增在有限 epoch 內沒有提升，就如實說明。
3. **測試集僅用於最終評估**：模型挑選（best epoch）只看 validation set；test set 不參與訓練與挑選。
4. **Notebook 圖表內嵌**：Notebook 開頭使用 `%matplotlib inline`，程式模組僅在非 Jupyter 環境才使用 `Agg` 後端，確保圖表能在 GitHub 上直接顯示（沿用 HW02 修正過的經驗）。
