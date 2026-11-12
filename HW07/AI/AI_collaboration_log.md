# 人工智慧工具協作紀錄 (AI Collaboration Log) — HW07

- **課程**：多模態影像資料處理 (MMIP) — Week 09 大型語言模型結構化輸出
- **學生**：林孟霆（`315833033`）
- **協作 AI 工具**：Claude Code（Anthropic）
- **規範依據**：課程簡報第 77–79 頁（作業一、作業二、作業三）

---

## 1. AI 協作範圍

| 環節 | AI 協助內容 | 人工確認重點 |
| :--- | :--- | :--- |
| 規範解析 | 逐條列出第 77–79 頁每一小題的配分與要求；閱讀第 11 頁（Context Keeper / Service Router / Hinter / Guider / Replier）、第 15–16 頁（SFT / DPO 資料格式）、第 55–58 頁（約束解碼）、第 61–70 頁（System One / Jev 與模型架構）作為設計依據 | 確認每一小題都有程式、結果檔與 README 段落 |
| 情境與 Spec | 設計 3C 線上商城客服情境、7 個部門、4 種處理狀態、7 欄位 JSON Spec、JSON Schema 與 9 條判斷規則 | 規則是否足以讓每一題的標準答案唯一（例如退換貨 / 維修 / 技術支援的界線、話題切換時不沿用舊訂單） |
| 資料 | 樣板式訓練對話產生器（720 / 100 筆）、合成影像（收據、錯誤畫面、損壞照片、注入圖片）、52 題手寫 Benchmark 與作業三三組測試資料 | 逐一檢視合成影像與 COCO 照片：自動挑選的評估照片有 3 張不合適（筆電被貓擋住、狗只出現在後照鏡、「蛋糕」其實是一盤菜），改為人工挑選並排除不清楚的照片；閒聊照片的文字依照片內容（寵物 / 食物）配對；抽查訓練樣本標註，修正一個有歧義的樣板（「取消剛剛下的訂單」） |
| 實驗 C | 自己實作 byte 層級 FSM 約束解碼 | 用真實 tokenizer 與隨機 logits 做模糊測試：第一版找到兩個錯誤（字串長度上限可被長 token 超過、intent 可為空字串；四個 missing_info 都用完後的逗號造成無合法 token），修正後 40 次全部合法 |
| 實驗 B、D | LoRA SFT、SFT + DPO（含 RPO 的 SFT 項，沿用 HW06 的教訓）、System One（依第 70 頁的 Joint Schema Head） | — |
| 作業三 | 聊天機器人、Context Keeper、Guard、Gradio UI、三個實驗與對照組 | Guard 的規則以訓練 / Benchmark / 可疑但正常的訊息檢查誤判：第一版會把「開發者模式要怎麼關掉」「列出退貨規則」這類正常訊息判為攻擊，改窄後 0 誤判（代價是只攔下 20 種攻擊中的 4 種，其餘交給模型） |
| 真實模型試跑 | 在本機 CPU 以 bf16 載入真實的 Qwen3-VL-2B，對 Benchmark 的 B16、B28 跑實驗 A、C：模型能從收據圖片讀出訂單編號；只靠 Prompt 已能輸出合法 JSON，但會違反跨欄位規則（route 卻填了 missing_info） | 發現模型照抄 Prompt 中唯一的範例（`missing_info:["order_id"]`），改為 clarify / route 各一個範例，讓實驗 A 是公平的「盡力而為的 Prompt」 |
| 程式驗證（執行前） | 本機只有 CPU：以「真實 processor / tokenizer + 2 層隨機權重的 Qwen3-VL」測試完整流程：多圖對話編碼、`logits_to_keep` 的 SFT loss 與完整前向傳播一致、DPO、System One 訓練 / 儲存 / 重新載入後機率一致、兩個 adapter 共用一個 base model、作業二所有階段、快取接續、作業三三個實驗與 Gradio 介面 | 這些測試只檢查程式能否正確執行；報告中的數字全部來自 Colab T4 上的正式執行 |
| 第一次 Colab 結果檢查 | 逐題檢查 `pred_*.json` 與訓練紀錄 | 發現兩個異常。（1）System One v1 的部門機率幾乎平均分配，驗證集部門正確率只有 41%，判斷為架構問題，依第 70 頁加入 state token（v2）。（2）SFT + DPO 會產生不存在的部門名稱，判斷是 DPO 壓低易混淆部門後的副作用，新增「SFT + DPO + 約束解碼」組驗證。第二次執行只跑新增的部分 |
| 結果撰寫 | README 的分析由 AI 依 `Output/` 的 CSV 與逐題紀錄撰寫 | 每個數字與題號都回到原始 JSON 核對。例如 DPO 經約束解碼後是 6 題中 5 題部門正確，不是全部；Keeper 實驗中 T10、T11 在兩種設定下都正確，不是 Keeper 修好的。沒有實測的部分（LoRA 合併後的延遲、改用 DPO 作 System 2）明確標示為推測 |

---

## 2. 誠信與品質控管原則

1. **所有數據皆來自實際執行**：README 的結果表格只填入 Colab 上產生的 `Output/*.csv`。
2. **評估資料不參與訓練**：52 題 Benchmark 與作業三測試資料都是手寫，商品、句型、影像版面與照片都和訓練資料分開。
3. **Gemini 結果固定**：所有回覆存在 `Output/gemini_*_cache.json`，重新執行直接讀取。
4. **API Key 不進程式碼**：只從環境變數或 Colab Secrets 讀取。
