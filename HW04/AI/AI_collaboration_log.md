# 人工智慧工具協作紀錄 (AI Collaboration Log) — HW04

- **課程**：多模態影像資料處理 (MMIP) — Week 04 時間序列模型與 Transformer
- **學生**：林孟霆（`315833033`）
- **協作 AI 工具**：Claude Code（Anthropic）；Google Gemini（Quiz 4 規範要求的描述評審，也作為被比較的描述方法之一）
- **規範依據**：課程簡報第 55–58 頁（Quiz 1–4）

---

## 1. AI 協作範圍

| 環節 | AI 協助內容 | 人工確認重點 |
| :--- | :--- | :--- |
| 規範解析 | 逐條列出第 55–58 頁每一小題的配分與要求，對應到程式模組、Notebook 段落與 README 章節 | 確認沒有遺漏小題（RNN 推論結果、LSTM 比較指標、ResNet 推論、Macro-AUC、Gemini 判斷、BLEU / BERTScore、回覆時間比較） |
| 資料集 | 選用 IMDB、Oxford-IIIT Pet、Flickr8k，並透過 Hugging Face 的資料集伺服器確認三者的切分大小與欄位 | 確認都是公開標準資料集，切分與官方一致 |
| 模型設計 | RNN / LSTM 只差循環單元；ViT 與 ResNet 使用相同訓練設定；描述模型採 CNN-LSTM 與 CNN-Transformer 兩種架構 | 確認比較是「一次只改一個因素」 |
| 運算資源 | 本機只有 CPU，訓練改在 Google Colab Tesla T4 上執行；程式自動偵測 GPU，並快取模型權重與 Gemini 回覆 | 確認訓練紀錄（`Checkpoints/*.json`）記錄的裝置為 `cuda` |
| 程式驗證 | 在本機以假資料、隨機權重、1–2 個 epoch 跑完整流程，檢查每個模組都能執行、輸出檔案完整、快取重跑正常；另以縮小版 BLIP 與 distilroberta 確認 BLIP 呼叫與 BERTScore 多參考答案的介面 | 這些測試只檢查程式能否執行，報告中的數字全部來自 Colab 上的正式執行 |
| Gemini | API Key 只從環境變數或 Colab Secrets 讀取；評審時把各方法的描述匿名並打亂順序，加入人工描述作為對照組 | 確認 Key 沒有出現在任何檔案中 |
| Colab 執行問題排除 | (1) 第一次執行時，Notebook 開頭讀得到 Key，但約一小時後 Quiz 4 再次從 Colab Secrets 讀取時逾時，Gemini 被跳過；改為第一次讀到 Key 就存入環境變數。(2) 程式自動選用的最新模型 `gemini-3.8-flash` 與 `gemini-3.5-flash` 持續回傳 503（伺服器忙碌）與 429（超過免費額度），先請使用者列出 Key 可用的模型，再改用 `gemini-3.5-flash-lite`，並把評估影像由 50 張降為 20 張；Gemini 寫描述與評審使用同一個模型，並刪除先前其他模型的快取，避免結果混用。(3) 第一次結果中 ViT 與 ResNet 的 Macro-AUC 四捨五入到小數 4 位後相同，改為顯示 6 位 | 確認快取檔中 40 筆回覆都來自 `gemini-3.5-flash-lite`，140 筆評審結果全部解析成功 |
| 結果檢查 | 逐張檢查 Colab 產出的圖表與 CSV。發現「速度 vs 品質」圖的標題寫死為 50 張（實際為 20 張），且圖例遮住長條；修正程式後，以已存下的 CSV 在本機重畫該圖（數據不變，不需重跑 Colab），並替換 Notebook 中的該張圖；刪除重跑時臨時加入的兩格輔助程式（解壓縮、列出可用模型）、兩條在 GitHub 上會顯示為 0% 的下載進度條與對應的 widget 中繼資料 | README 的每個數字與例子都對照 `Output/*.csv`、`Checkpoints/*.json` 與圖片；Notebook 通過 nbformat 格式驗證 |
| 提交前審核 | 對照規範逐題檢查，並逐句重讀 README：修正證據不足或不精確的敘述（例如 RNN 與 LSTM 的 Validation Loss 起伏其實相近、依長度分析改以未截斷的區間比較、ViT 的歸納偏置用語、計時方式的描述），並補上 RNN 與 LSTM 使用相同訓練預算的限制說明；以最終版程式重跑假資料測試（含 Gemini 分支） | 修改只涉及文字與圖表標題，所有數據不變 |

---

## 2. 誠信與品質控管原則

1. **所有數據皆來自實際執行**：README 表格中的數字皆來自 Notebook 在 Colab 執行後產生的 `Output/*.csv`。
2. **不寫死結論**：模型優劣等結論依實際結果撰寫。
3. **測試集僅用於最終評估**：模型挑選只看 Validation set。
4. **Notebook 圖表內嵌**：Notebook 開頭使用 `%matplotlib inline`，程式模組只在非 Jupyter 環境才使用 `Agg` 後端，確保圖表能在 GitHub 上直接顯示。
