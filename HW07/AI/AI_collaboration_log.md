# 人工智慧工具協作紀錄 (AI Collaboration Log) — HW07

- **課程**：多模態影像資料處理 (MMIP) — Week 07 多模態語言模型如何運作
- **學生**：林孟霆（`315833033`）
- **協作 AI 工具**：Claude Code（Anthropic）
- **規範依據**：課程簡報第 64–66 頁（作業一、作業 B、作業 C）

---

## 1. AI 協作範圍

| 環節 | AI 協助內容 | 人工確認重點 |
| :--- | :--- | :--- |
| 規範解析 | 逐條列出第 64–66 頁每一小題的配分與要求，對應到程式模組、Notebook 段落與 README 章節 | 確認沒有遺漏小題（GPU 平均記憶體、優化 Prompt × 量化、影像劣化） |
| 模型選擇 | 依簡報第 16、17、57 頁的建議與 T4 16 GB / 2080 Ti 11 GB 的限制，選用 Qwen3-4B-Instruct-2507（FP16 8 GB）、Qwen3-VL-2B-Instruct、SmolVLM2-2.2B-Instruct；全部非 gated | 確認兩張 Turing 卡都沒有原生 bf16，因此全部使用 FP16，並加上 FP16 溢位檢查 |
| Benchmark 設計 | 作業一 20 題各對應一種能力並附規則式檢查；作業 B 從 COCO 標註挑選候選影像後**逐張看過**再寫題目與答案，另用程式產生 5 張答案精確的合成影像 | 確認每題答案正確（例如斑馬照片左緣有半隻，COCO 標註為 3 隻） |
| 作業 C 優化面向 | 以「台灣繁體中文用語」作為可全自動量化的面向；簡體字以 OpenCC 字表判斷並排除台灣標準字形；兩岸用語表只收明確的用法 | 確認 170 題自寫問題本身全部被判定為乾淨 |
| DPO 實作 | 以 PEFT LoRA + 自行撰寫的 DPO loss（與 TRL `loss_type="sigmoid"` 相同），LLM 與 VLM 共用；參考模型 = 關閉 LoRA 的同一模型；`logits_to_keep` 只計算回答位置的 logits 以節省記憶體 | — |
| 程式驗證（執行前） | 本機只有 CPU，以「真實 tokenizer / processor + 縮小成 2 層的隨機權重模型」做小測試：批次與逐筆生成結果相同、`logits_to_keep` 的 log-prob 與完整前向傳播一致、DPO 第一步 loss = ln 2、LoRA 只加在語言模型、adapter 重新載入後 log-prob 相同、中斷後從快取接續；規則式檢查以 20 組正確 / 錯誤回答測試 | 這些測試只檢查程式能否正確執行；報告中的數字全部來自 Colab T4 上的正式執行 |
| 執行時的環境問題 | （1）Colab 預設版 transformers 的 SmolVLM2 有 bug → 固定 transformers 5.15.0；（2）Colab 預裝的舊版 torchao 讓新版 peft 報錯 → 在 `dpo.py` 讓 peft 忽略不相容的 torchao；（3）錯誤後殘留的模型佔用 GPU 造成 OOM → 重新啟動工作階段 | 在本機以模擬方式重現 torchao 錯誤並確認修正有效 |
| **結果檢查後的修正 1：規則誤判** | 逐題閱讀第一次的回答，發現作業一 A17 的規則把 6 個答對的回答判錯（計算過程中提到其他人名）；改為檢查「最後提到的人名」，以存檔的回答重新計分，不需重新生成 | 確認只有這 6 個判定改變（都是 Gemini 10 分的回答），其餘 234 個判定不變，原本的錯誤測例仍判錯；README 註明 Notebook 中作業一表格是修正前的數字 |
| **結果檢查後的修正 2：DPO 退化** | 第一次 DPO 後大陸用語歸零，但簡體字回答從 7.5% 增加到 77.5%，並出現日文、德文與亂碼。判斷為 likelihood displacement（chosen / rejected 只差幾個詞），新增 DPO + SFT（RPO）版本，資料與超參數不變，只重跑作業 C；並記錄 chosen / rejected 相對參考模型的 log-ratio | 補量確認標準 DPO 的 chosen log-ratio 為 −135（偏好回答的機率也被壓低），DPO+SFT 為 +0.03；README 同時報告兩個版本，包含 DPO+SFT 仍增加的簡體字 |
| 結果分析 | 依 `Output/*.csv`、生成紀錄與 Gemini 理由撰寫 README；所有差值由 CSV 計算後寫入，並逐項回頭核對（例如修正「INT8 回答大多與 FP16 相同」為實際的 6/20 題） | 逐一比對 README 數字與 CSV |
| 運算資源評估 | 依 T4 與 2080 Ti 的規格估計各部分執行時間；T4 以實際執行紀錄更新，2080 Ti 為推估 | README 標明 2080 Ti 沒有實測 |

---

## 2. 誠信與品質控管原則

1. **所有數據皆來自實際執行**：README 表格中的數字皆來自 Colab T4 上產生的 `Output/*.csv`；作業一 A17 的重新計分只改變判定規則，沒有改變任何模型回答。
2. **失敗結果如實報告**：標準 DPO 的退化結果保留並分析，沒有以修正後的版本取代。
3. **測試資料不參與訓練**：作業 C 的 40 題保留問題、80 張評估影像都不參與 DPO；DPO 的模型挑選只看驗證集。
4. **Gemini 結果固定**：所有回覆存於 `Output/*_gemini_cache.json`，重新執行直接讀取，不會再次呼叫 API。
5. **API Key 不進程式碼**：只從環境變數或 Colab Secrets 讀取。
