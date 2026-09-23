# 人工智慧工具協作紀錄 (AI Collaboration Log)

## 1. 協作基本資訊與工具說明
- **課程名稱**：多媒體影像處理 (Multimedia Image Processing, MMIP)
- **作業項目**：作業二 (HW02 - Machine Learning, Deep Learning, and Model Evaluation)
- **協作 AI 工具**：Google DeepMind Antigravity (Gemini 2.5 Pro 核心架構)
- **協作目的**：依據課程簡報規範，協助進行課程簡報規格解析、機器學習與深度學習程式架構設計、資料前處理管線建置、模型除錯優化、指標視覺化以及防呆防造假嚴格檢驗。

---

## 2. AI 協作主要環節與分工

本作業開發過程中，AI 工具主要於以下六大核心環節提供輔助，所有關鍵邏輯、演算法選擇、數值產出均經由本機環境實際執行驗證：

### 環節一：作業規範解析與題意對齊 (Curriculum Alignment & Spec Parsing)
- **AI 輔助內容**：深入解析課程簡報第 71 至 75 頁之評分標準與各題要求：
  - **Quiz 1 (30%)**：乳癌威斯康辛資料集二元分類，完成 Logistic Regression 門檻調優 (0.5 與 0.35) 與 Random Forest 比較，包含混淆矩陣與錯誤代價分析。
  - **Quiz 2 (40%)**：UCI 信用卡違約資料集 (30,000 筆，23 特徵)，建構 PyTorch MLP 模型，實作單筆驗證樣本即時推論，訓練至少 50 個 Epochs，並設計改進策略 (Dropout 0.3 + L2 Weight Decay 1e-4 + BatchNorm) 進行防過擬合比較。
  - **Quiz 3 (30%)**：模型評估指標 ROC 與 AUC，計算並繪製 MLP 與 Random Forest 之 ROC 曲線與 AUC 面積，進行統計與領域決策意涵詮釋。
- **人類審核**：比對簡報原始投影片，確認無自行幻想或遺漏任何規定條件。

### 環節二：資料管線與前處理自動化 (Data Pipeline Automation)
- **AI 輔助內容**：整合 Kaggle 官方信用卡違約資料集 (`UCI_Credit_Card.csv`) 與經典威斯康辛乳癌診斷資料集 (`quiz1_breast_cancer.csv`)，移除多餘重複的下載腳本，自動進行流水號 ID 清理、標頭大小寫歸一化、資料型態轉換與特徵標準化 (`StandardScaler`)。
- **技術細節**：落實 80/20 分層抽樣切分 (`train_test_split(stratify=y)`)，且標準化轉換器僅於訓練集進行 `fit`，再對驗證集進行 `transform`，確保無任何資料外洩 (Data Leakage)。

### 環節三：機器學習分類器與門檻調校 (ML Classification & Threshold Tuning)
- **AI 輔助內容**：輔助設計 Logistic Regression 與 Random Forest 分類架構，撰寫動態門檻計算模組 (`compute_classification_metrics`)。
- **技術細節**：透過動態比對 0.5 與 0.35 門檻下的混淆矩陣 (TP, FP, FN, TN)，解析召回率 (Recall) 提升對醫療診斷場景中降低偽陰性 (FN, 漏診) 的重大臨床價值。

### 環節四：PyTorch 深度學習多層感知機 (Deep Learning MLP Architecture)
- **AI 輔助內容**：使用 PyTorch 構建自定義類神經網路，包含基準模型 `BaseMLP` (Linear -> ReLU -> Linear -> ReLU -> Linear) 與改進模型 `ImprovedMLP` (Linear -> BatchNorm1d -> ReLU -> Dropout(0.3) -> ...)。
- **技術細節**：
  - 設計 PyTorch 訓練迴圈，支援 Epoch 等級之訓練損失 (Train Loss) 與驗證損失 (Val Loss) 實時紀錄。
  - 實作單一樣本 (`single_sample_eval`) 抽取、前處理與推論流程，輸出模型信心機率與類別判定。
  - 引入 Adam 優化器與 L2 Weight Decay (`1e-4`)，結合 Dropout 與批次正規化抑制模型過擬合。

### 環節五：模型評估指標與雙曲線視覺化 (ROC & AUC Visualization)
- **AI 輔助內容**：輔助編寫 ROC 曲線計算與面積整合邏輯，使用 Matplotlib 繪製雙模型 (PyTorch MLP vs. Random Forest) 同圖對比。
- **技術細節**：將圖表標題、軸標籤與圖例全數採用英文字體與規範設計，避免 Linux / Windows 無中文字型環境下之缺字警告 (`Glyph missing DejaVu Sans`)。

### 環節六：嚴格代碼審核、防呆防造假與執行環境調校
- **AI 輔助內容**：
  - **消滅寫死數值**：全面檢查代碼與報告，確保所有 Precision、Recall、F1、Loss、AUC 與執行時間均為執行當下動態計算並格式化輸出。
  - **非互動式後端修復**：排除 Windows 平台上 Matplotlib 預設 TkAgg 後端於背景腳本結束時觸發之 `Tcl_AsyncDelete` 多執行緒資源釋放錯誤，統一強制設置 `matplotlib.use('Agg')`。
  - **編碼規範**：確保腳本相容 Windows cp1252 環境，配置 UTF-8 輸出防範 `UnicodeEncodeError`。
  - **完全排除 LaTeX**：確保全專案 Markdown 檔案與 Jupyter Notebook 無任何符號或公式語法衝突。

---

## 3. 人工審核與反思 (Human Audit & Critical Reflection)

1. **代碼驗證**：AI 產出的模型程式碼與資料預處理流程，皆在本機環境 (Python 3.12, PyTorch 2.14.0, Scikit-Learn 1.9.1) 實際完整執行測試，確保無語法錯誤、無記憶體洩漏且執行結果具備可重現性。
2. **圖表產出檢查**：審核輸出的三張圖片 (`quiz1_confusion_matrices.png`, `quiz2_mlp_loss_comparison.png`, `quiz3_roc_curve_comparison.png`)，確認軸標籤、數值標註與圖例清晰無截斷，混淆矩陣格點數值正確相符。
3. **平台相容性修正**：AI 最初撰寫 Matplotlib 繪圖模組時未指定後端，在多執行緒環境下引發 Tkinter 警告；經人工指示與調整後，加入 `matplotlib.use('Agg')`，徹底排除非預期錯誤。
4. **恪守學術誠信**：AI 僅作為開發效率輔助工具，核心演算法理解、作業結構規劃與最終結果審核均由開發者完整掌握與把關。
