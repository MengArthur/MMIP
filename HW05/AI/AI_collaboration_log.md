# 人工智慧工具協作紀錄 (AI Collaboration Log) — HW05

- **課程**：多模態影像資料處理 (MMIP) — Week 05 多模態深度學習神經網路
- **學生**：林孟霆（`315833033`）
- **協作 AI 工具**：Claude Code（Anthropic）
- **規範依據**：課程簡報第 54–56 頁（Quiz 1–3）

---

## 1. AI 協作範圍

| 環節 | AI 協助內容 | 人工確認重點 |
| :--- | :--- | :--- |
| 規範解析 | 逐條列出第 54–56 頁每一小題的配分與要求，對應到程式模組、Notebook 段落與 README 章節 | 確認沒有遺漏小題（CNN 與 ViT 檢索比較、更換 Backbone、Week 4 vs 本次 BLEU／BERTScore） |
| 資料集 | Quiz 1 選用影像檢索標準資料集 CUB-200-2011，採未見類別切分；Quiz 2、3 沿用 Week 4 的 Flickr8k。以 Hugging Face 資料集伺服器確認資料筆數與欄位 | 確認 Quiz 2 的圖文配對是自行建立（符合「自行準備」）；測試鳥種不出現在訓練中 |
| 模型設計 | Quiz 2 依簡報第 45 頁的「Image + Text Feature Fusion」架構；Quiz 3 直接沿用 Week 4 的 decoder 與評估程式，只更換影像編碼器 | 確認 Quiz 3 的比較只差一個因素 |
| 審核修正 | 自我審核時發現：隨機抽取的不相符描述太容易分辨，4 種 Backbone 組合可能都接近滿分而比不出差異；因此加入「主角相同」的困難測試集（只用於測試） | 確認困難測試集的負樣本不會用到影像自己的描述 |
| 相容性 | CLIP 的影像／文字特徵改為直接呼叫 vision_model、text_model 與投影層，避免 transformers 新舊版本 `get_image_features` 回傳型態不同 | — |
| 程式驗證 | 在本機以隨機權重做數秒內完成的小測試：檢索指標（完美嵌入應得 R@1 = mAP = 1）、圖文配對的負樣本不會用到自己的描述、各模型輸出形狀、CLIP 呼叫方式 | 這些測試只檢查程式能否執行，報告中的數字全部來自 Colab 上的正式執行 |
| 推論展示修正 | 第一次執行後檢查結果，發現自訂文字 × 隨機測試影像的相符機率幾乎全是 0（隨機影像本來就不符合這些文字），雖然不是程式錯誤，但無法展示推論能力；改為「對 1,000 張測試影像逐一計算相符機率，列出前 3 名」，只重新執行 Quiz 2 的 cell | 確認重跑後 4 組 Backbone 的指標與訓練紀錄和第一次完全相同（只有訓練秒數不同）；逐張檢查前 3 名影像，並如實記錄「黑貓睡在沙發上」找到的都是狗的失敗例 |
| 結果分析 | 依 `Output/*.csv` 與 `Checkpoints/*.json` 撰寫 README 的結果表格與分析；所有差值由 CSV 計算後再寫入。CLIP + LSTM 退步的原因以訓練紀錄（train loss 較高）佐證，並在報告中標明是推測 | 逐一比對 README 數字與 CSV；確認關於程式行為的敘述（LSTM 用 token 平均值、ViT 取 [CLS]、ResNet 用全域平均池化）與程式碼一致 |
| 圖表修正 | `quiz3_week4_vs_clip.png` 的圖例蓋住數值標籤；修改繪圖程式的圖例位置與 y 軸範圍，並在本機用同一份 CSV 重畫 | 數字不變；README 註明 Notebook 內嵌的是原始版本 |
| 運算資源 | 本機只有 CPU，所有訓練與特徵萃取都在 Google Colab Tesla T4 上執行 | 確認訓練紀錄的裝置為 `cuda` |

---

## 2. 誠信與品質控管原則

1. **所有數據皆來自實際執行**：README 表格中的數字皆來自 Notebook 在 Colab 執行後產生的 `Output/*.csv`。
2. **Week 4 對照數據不重新捏造**：直接讀取 HW04 實際輸出的 `quiz4_test_metrics.csv`（複製為 `Data/week4_test_metrics.csv`，已比對內容完全相同）。
3. **測試集僅用於最終評估**：模型挑選只看 Validation set。
4. **Notebook 圖表內嵌**：程式模組只在非 Jupyter 環境才使用 `Agg` 後端。
