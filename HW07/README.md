# 國立陽明交通大學 多模態影像資料處理 (MMIP) 作業七技術報告
## 多模態語言模型：LLM 部署／評估／提示工程／量化、VLM 比較、DPO 微調 LLM 與 VLM

- **學生**：林孟霆（`315833033`）
- **規範依據**：課程投影片 `多模態影像資料處理（MMIP）第七堂_多模態語言模型如何運作` 第 64–66 頁
- **配分**：作業一（基礎 20 + 進階 10）、作業 B（基礎 20）、作業 C（基礎 20 + 進階 20）
- **執行環境**：Google Colab **Tesla T4 GPU**（15 GiB 可用）、PyTorch 2.11.0+cu130、transformers 5.15.0（作業一使用 Colab 預設版本，見[執行紀錄](#執行紀錄)）；商用 LLM 評審：`gemini-3.5-flash-lite`

> [!NOTE]
> **給助教的導覽 (For TA & Reviewers)**：
> - **文字報告與分析**：本 `README.md`。
> - **程式執行結果**：`HW07_notebook.ipynb`（已在 Colab T4 上執行，表格與圖表內嵌在輸出中）。
> - **圖表與結果檔**：`Output/`；**所有模型回答**：`Output/partA_answers.csv`、`partB_answers.csv`、`partC_llm_test_responses.csv`、`partC_vlm_descriptions.csv`；**Gemini 回覆紀錄**：`Output/*_gemini_cache.json`；**DPO 訓練紀錄**：`Checkpoints/*/history.json`。
> - **自訂 Benchmark 與資料**：`Data/`（作業一 20 題、作業 B 20 題與影像、作業 C 170 題繁中問題、100 題通用能力、COCO 子集）。
> - **AI 協作紀錄**：`AI/AI_collaboration_log.md`。
> - 所有數字皆來自實際執行結果。模型權重（Hugging Face 下載）、LoRA 權重與 COCO 影像沒有放進 git，重新執行會自動下載並重建。
>
> **建議閱讀順序**：
> 1. 下方「**評分項目對照表**」：每一小題對應的 README 段落與結果檔。
> 2. 各部分的「結果」小節（A-1～A-5、B-1～B-2、C-1～C-3、CV-1～CV-2）：表格 + 分析，數字都可在 `Output/` 的同名 CSV 查到。
> 3. 需要看實際執行過程時再開 `HW07_notebook.ipynb`。**Notebook 中作業一表格的「規則式通過」是修正前的數字**（A17 檢查規則有誤，見[執行紀錄](#執行紀錄)），修正後的數字在 Notebook 的說明格與本 README；以「說明」開頭的引用框是事後加入的文字說明。
> 4. 想看模型實際的回答：`Output/partA_answers.csv`（含 Gemini 給分理由）、`partB_answers.csv`、`partC_llm_test_responses.csv`、`partC_vlm_descriptions.csv`。

### 評分項目對照表（各小題在哪裡看）

| 題目（配分） | 規範要求 | 本作業做法 | 文字分析 | 圖表 / 結果檔 |
| :--- | :--- | :--- | :--- | :--- |
| 作業一 基礎（5） | 成功部署並執行一個開源 LLM | Qwen3-4B-Instruct-2507（FP16）聊天示範與速度 | [A-1](#a-1-部署基礎) | `Output/partA_deploy_demo.csv` |
| 作業一 基礎（5） | 20 道涵蓋不同能力面向的自定義 Instruction Benchmark | 20 題 × 20 種能力，每題附規則式檢查與參考答案 | [設計](#作業一大型語言模型基礎-20-分--進階-10-分) | `Data/partA_llm_benchmark.json` |
| 作業一 基礎（5） | 應用 Prompting Techniques，以量化指標或商用 LLM 評估 | 8 種 Prompt；規則式通過率 + Gemini 1–10 分 | [A-2](#a-2-prompt-技巧比較基礎) | `Output/partA_technique_summary.csv`、`partA_techniques.png` |
| 作業一 基礎（5） | 調整數值精度，用預設 Prompt，以量化指標與商用 LLM 評估 | FP16 / INT8 / NF4 × 預設 Prompt | [A-3](#a-3-模型量化基礎) | `Output/partA_quant_summary.csv`、`partA_quantization.png` |
| 作業一 進階（5） | 預設精度下推論的平均 GPU 記憶體 | 每 20 ms 取樣 allocated / reserved / NVML，平均與峰值 | [A-4](#a-4-預設精度的-gpu-記憶體進階) | `Output/partA_memory_profile.csv`、`partA_memory_timeline.png` |
| 作業一 進階（5） | 優化 Prompt + 低精度模型：記憶體與成效比較 | Zero-shot CoT × INT8 / NF4 vs FP16 | [A-5](#a-5-優化-prompt--低精度量化進階) | `Output/partA_quant_summary.csv`、`partA_memory_profile.csv` |
| 作業 B 基礎（10） | 部署並執行兩個開源 VLM | Qwen3-VL-2B-Instruct、SmolVLM2-2.2B-Instruct | [B-1](#b-1-整體比較) | `Output/partB_summary.csv` |
| 作業 B 基礎（5） | 自行設計 20 道視覺理解 Benchmark | 15 張 COCO 照片 + 5 張合成影像，14 種任務 | [設計](#作業-b視覺多模態大模型基礎-20-分) | `Data/partB_vlm_benchmark.json`、`Data/partB_images/` |
| 作業 B 基礎（5） | 同樣 Prompt 測兩個 VLM，量化指標 + 商用 LLM | 規則式準確率 + Gemini 看圖評分 + 延遲 / 記憶體 | [B-1、B-2](#b-2-逐題分析) | `Output/partB_answers.csv`、`partB_summary.png`、`partB_examples.png` |
| 作業 C 基礎（5） | 自訂優化面向、RLHF 訓練資料 ≥ 100 組 | 台灣繁體中文用語；130 組偏好資料 | [設計](#作業-c以-dpo-微調-llm基礎-20-分) | `Data/partC_tw_prompts.json`、`Output/partC_llm_dpo_pairs.json` |
| 作業 C 基礎（5） | 另 100 組與優化面向無關的 Benchmark | 60 題 MMLU + 40 題 GSM8K | [C-2](#c-2-通用能力) | `Data/partC_general_benchmark.json` |
| 作業 C 基礎（10） | 微調開源 LLM；量化優化面向與通用能力的變化 | DPO 與 DPO+SFT（LoRA）；40 題保留測試題 + 100 題通用能力 | [C-1～C-3](#c-1-優化面向) | `Output/partC_llm_aspect.csv`、`partC_llm_general.csv`、`partC_llm_results.png`、`partC_llm_training_*.png` |
| 作業 C 進階（10） | 自訂優化面向（如避免描述不存在的東西），資料 ≥ 100 組 | 減少物件幻覺；COCO 150 張影像、449 組 | [設計](#作業-c-進階以-dpo-微調-vlm進階-20-分) | `Data/partC_coco_subset.json`、`Output/partC_vlm_dpo_pairs.json` |
| 作業 C 進階（5） | 微調開源 VLM，量化優化面向 | DPO + LoRA；POPE、CHAIR | [CV-1](#cv-1-pope-與-chair) | `Output/partC_vlm_pope.csv`、`partC_vlm_chair.csv`、`partC_vlm_results.png`、`partC_vlm_examples.png` |
| 作業 C 進階（5） | 調整影像畫質、明暗等，觀測平均表現 | 7 種劣化 × Base / DPO 的 POPE | [CV-2](#cv-2-影像劣化) | `Output/partC_vlm_pope.csv`、`partC_vlm_degradations.png` |

---

## 目錄結構

```text
HW07/
├── HW07_main.py              # 一鍵執行（--parts A B C CV）
├── HW07_notebook.ipynb       # 已在 Colab T4 執行的 Notebook
├── gpu_benchmark.py          # 量測 GPU 速度（T4 / 2080 Ti 各跑一次比較）
├── README.md
├── requirements.txt
├── Code/
│   ├── common.py             # 裝置、GPU 記憶體取樣器、OOM 自動減半 batch
│   ├── llm.py                # LLM 載入（FP16 / INT8 / NF4）與批次生成
│   ├── scoring.py            # 作業一、B 的規則式檢查
│   ├── gemini.py             # Gemini 評審（回覆快取）
│   ├── partA_llm.py          # 作業一
│   ├── vlm.py                # 兩個 VLM 的載入、批次生成、Yes/No 機率
│   ├── partB_vlm.py          # 作業 B
│   ├── taiwan_terms.py       # 作業 C：簡體字 / 大陸用語偵測、偏好資料建立
│   ├── dpo.py                # DPO（可加 SFT 項）+ LoRA 訓練，LLM 與 VLM 共用
│   ├── partC_dpo_llm.py      # 作業 C 基礎
│   ├── coco_objects.py       # CHAIR 物件詞對照
│   ├── partC_dpo_vlm.py      # 作業 C 進階（POPE、CHAIR、影像劣化）
│   └── prepare_data.py       # 建立 Data/ 內的 Benchmark（已執行，輸出已提交）
├── Data/                     # 自訂 Benchmark、影像、偏好資料題目、COCO 子集（說明見 Data/README.md）
├── Checkpoints/
│   ├── partC_llm_dpo_lora/       # 作業 C 標準 DPO：history.json（每步 loss / reward / margin）、adapter_config.json
│   ├── partC_llm_dposft_lora/    # 作業 C DPO+SFT
│   └── partC_vlm_dpo_lora/       # 作業 C 進階 VLM DPO（LoRA 權重不提交）
├── Output/
│   ├── partA_*.csv / *.png       # 作業一：Prompt 技巧、量化、記憶體的彙整表與圖
│   ├── partA_answers.csv         # 作業一：12 種設定 × 20 題的回答、規則式判定、Gemini 分數與理由
│   ├── partB_*.csv / *.png       # 作業 B：兩個 VLM 的彙整、逐題回答與範例圖
│   ├── partC_llm_*.csv / *.png   # 作業 C：優化面向、通用能力、用語 / 簡體字統計、訓練曲線
│   ├── partC_llm_dpo_pairs.json  # 作業 C：130 組偏好資料（prompt / chosen / rejected / 類型）
│   ├── partC_vlm_*.csv / *.png   # 作業 C 進階：POPE、CHAIR、描述、劣化範例、訓練曲線
│   ├── partC_vlm_dpo_pairs.json  # 作業 C 進階：449 組偏好資料
│   ├── *_gemini_cache.json       # Gemini 原始回覆（重新執行直接讀取）
│   ├── *_meta.json               # GPU、模型、設定紀錄
│   └── *_runs/                   # 每次生成的原始紀錄（快取），重新執行時不需重算
└── AI/AI_collaboration_log.md
```

**執行方式**：Colab 上把 `HW07/` 放到 Google Drive `MyDrive/MMIP/HW07`（或壓縮成 `HW07.zip` 上傳）、選 T4 GPU、在 Secrets 加入 `GEMINI_API_KEY` 後全部執行（詳見 Notebook 第一格）。伺服器：`pip install -r requirements.txt` 後執行 `python HW07_main.py`。每個生成結果、Gemini 回覆與 LoRA adapter 都有快取，中斷後重新執行會從斷點繼續。

---

## 執行環境與速度評估（Colab 免費 T4 vs 實驗室 RTX 2080 Ti）

兩張卡都是 Turing 架構（sm_75）：都**沒有原生 bf16**（因此全部使用 FP16，並在載入時檢查 FP16 是否溢位；本次三個模型都沒有溢位），也都不支援 FlashAttention-2（使用 PyTorch SDPA）。

| 項目 | Colab 免費 T4 | 實驗室 RTX 2080 Ti |
| :--- | :--- | :--- |
| VRAM | 16 GB（可用 14.6 GiB） | 11 GB（可用約 10.7 GiB） |
| FP16 Tensor Core 峰值 | 65 TFLOPS，但 70 W 功耗上限，長時間運算會降頻 | 約 54 TFLOPS（FP32 累加），250 W |
| 記憶體頻寬 | 320 GB/s | 616 GB/s |
| 生成（decode，受頻寬限制） | 1× | 估計約 1.6–1.9× |
| 訓練（DPO，受運算限制） | 1× | 估計約 1.3–1.7× |
| 作業一 batch size | 10 | 5（11 GB 放不下 10 個長 Few-shot Prompt 的 KV cache） |

**T4 實測**（本作業的執行紀錄）與 2080 Ti 估計：

| 項目 | T4 實測 | 2080 Ti 估計 |
| :--- | ---: | ---: |
| Qwen3-4B FP16 單筆生成速度 | 15.1 tok/s | 約 25–28 tok/s |
| 作業一（12 組 × 20 題 + 6 組記憶體量測） | 約 27 分（生成 1,143 秒 + 記憶體量測 466 秒，不含模型載入） | 約 18 分 |
| 作業 C：Qwen3-4B DPO 訓練（351 組 × 2 次前向／反向） | 517 秒（DPO）、525 秒（DPO+SFT） | 約 6 分 |
| 作業 C 進階：Qwen3-VL-2B DPO 訓練（808 組 × 2，含參考模型 128 秒） | 1,106 秒 | 約 12 分 |
| 全部（含評估、下載約 21 GB 模型） | 約 1.5 小時 | 約 1 小時 |

- 2080 Ti 的數字是依頻寬與運算量比例推估，沒有實測；`gpu_benchmark.py` 可在兩邊實測（本次在 Colab 執行時受 torchao 問題影響沒有完成，見[執行紀錄](#執行紀錄)）。
- **Colab 免費版的限制**：不保證分配到 GPU、閒置會斷線、每日 GPU 時數有上限，斷線後本機磁碟清空。把 `HW07/` 放在 Google Drive 讓快取在斷線後保留，本次中途數次重新啟動都只重算尚未完成的部分。
- **2080 Ti 的限制**：作業一需改用 batch 5；作業 C 的 4B DPO 峰值約 9.5–10 GB，接近 11 GB 上限，共用伺服器需先確認 GPU 空閒。優點是沒有時數限制、不會斷線，速度約快 1.5 倍。

---

## 作業一：大型語言模型（基礎 20 分 + 進階 10 分）

### 設計
| 項目 | 設定 |
| :--- | :--- |
| 模型 | `Qwen/Qwen3-4B-Instruct-2507`（4.0 B，Apache-2.0，非 gated）；簡報第 17 頁推薦「Qwen3-4B / FP16」 |
| 生成 | Greedy decoding（結果可重現）、最多 768 個新 token、batch 10 |
| Benchmark | 20 題、20 種能力（下表），每題有規則式檢查（量化指標）與給 Gemini 的參考答案 |
| Prompt 技巧 | 8 種（下表），全部在 FP16 上執行 |
| 精度 | FP16（預設）；INT8 = bitsandbytes LLM.int8；NF4 = 4-bit NormalFloat + double quant（compute dtype FP16） |
| 評估 | 規則式通過率；Gemini 對同一題 12 個設定的回答（匿名、打亂順序）一次評 1–10 分 |
| GPU 記憶體 | 背景執行緒每 20 ms 取樣 `memory_allocated`（權重 + KV cache + 中間值）、`memory_reserved`、NVML 本程序用量；另做 5 題 × 256 token 的 batch = 1 記憶體曲線 |

**20 題 Benchmark**（`Data/partA_llm_benchmark.json`）：數學計算、多步驟數學推理、邏輯推理、日期推理（閏年）、字串處理（strawberry 有幾個 r）、程式撰寫（實際執行 6 個測資）、程式理解（list 參照與複製）、知識問答、抗幻覺（不存在的諾貝爾數學獎）、中→英翻譯、英→中翻譯、40 字以內摘要、JSON 格式、3 行條列、恰好 3 句、情緒分類、表格推理、安全拒答、每行含「風」的四行詩、單位換算。

規則式檢查只看「最終答案」：若回答中有「最終答案：」，只檢查其後的內容。檢查類型包括最後一個數字、數字序列、正規表示式、關鍵字、JSON 鍵值、條列行數、句數、每行含指定字、在子程序中執行程式碼測資等。

**Prompt 技巧**（簡報第 30–35 頁）

| 名稱 | 內容 |
| :--- | :--- |
| `baseline` | 直接輸入題目（預設 Prompt） |
| `zero_shot_cot` | 題目後加「請先一步一步思考，最後一行以「最終答案：」開頭寫出答案」 |
| `few_shot` | 3 組示範問答（計算、條列、翻譯），示範「最終答案：」格式 |
| `emotion` | 題目後加「這對我的職涯發展非常重要，請務必認真作答。」（EmotionPrompt） |
| `xml` | `<instruction>…</instruction><question>…</question>` |
| `markdown` | `## 任務` / `## 問題` |
| `combined_zh` | Markdown 系統提示（角色、規則、Emotional Stimuli）+ Few-shot + CoT + XML 題目 |
| `combined_en` | 同上，但系統提示改用英文（簡報第 35 頁「用中文 or 用英文」） |

「優化後的 Prompt」= FP16 上規則式通過率最高的非預設技巧（同分取平均 token 數較少者），結果為 `zero_shot_cot`。

### A-1 部署（基礎）
FP16 載入約 33 秒，權重佔 7,672 MiB；單筆生成約 14–15 tok/s。示範（`Output/partA_deploy_demo.csv`）：「請用三句話向高中生解釋什麼是大型語言模型」→ 回答恰好三句、內容正確；英文 haiku 也符合 5-7-5 的三行格式。示範檔中的峰值記憶體 12,083 MiB 受到前一次錯誤殘留的記憶體影響（見[執行紀錄](#執行紀錄)），記憶體以 A-4 的量測為準。

### A-2 Prompt 技巧比較（基礎）

| Prompt | 規則式通過 | Gemini 平均 | 平均輸出 token | 失敗題目 |
| :--- | ---: | ---: | ---: | :--- |
| **baseline（預設）** | **20 / 20** | **9.65** | 146 | — |
| zero_shot_cot | 20 / 20 | 9.30 | 233 | — |
| emotion | 19 / 20 | 8.85 | 230 | A05 |
| combined_zh | 18 / 20 | 8.65 | 61 | A04、A05 |
| combined_en | 17 / 20 | 8.65 | 36 | A04、A05、A06 |
| few_shot | 16 / 20 | 8.50 | 46 | A03、A04、A05、A19 |
| xml | 16 / 20 | 8.20 | 55 | A03、A04、A05、A07 |
| markdown | 16 / 20 | 8.00 | 87 | A03、A04、A05、A07 |

（`Output/partA_technique_summary.csv`、`partA_techniques.png`；逐題回答與 Gemini 理由在 `partA_answers.csv`）

- **預設 Prompt 已經最好**：Qwen3-4B-Instruct-2507 經過大量指令微調，直接提問時會自己逐步推導（平均 146 token），20 題全對，Gemini 平均 9.65。
- **會讓回答「變短」的技巧，都在推理題上失分**：Few-shot、XML、Markdown、兩種組合 Prompt 的平均輸出只有 36–87 token，模型直接給答案、不再推導，錯誤集中在同一組推理題：
  - A04 日期：回答「3 月 2 日」（忽略 2024 是閏年，5 種技巧都錯）；
  - A05 strawberry：回答「2」（7 種非 CoT 技巧中有 6 種錯）；
  - A03 邏輯排序：回答「丙、甲、丁、乙」；A07 程式理解：回答「6 10 6」或「4 10 4」。
  - Few-shot 的 3 組示範答案都很短，模型學到的是「示範的長度」而不是「答對的方法」；組合 Prompt 雖然在系統提示寫了「先逐步思考」，示範仍讓它跳過推理，可見 Few-shot 示範的影響大於文字規則。
- **Zero-shot CoT** 與預設同樣 20 題全對，但 Gemini 分數較低（9.30）：A09 在指出「沒有諾貝爾數學獎」後繼續推導，捏造了 2023 年阿貝爾獎得主（Gemini 4 分）；A01、A06 的推理文字也被 Gemini 小幅扣分。CoT 讓模型多寫，也就多了出錯與捏造的機會。
- **Emotional Stimuli** 讓回答變長且多了客套話，A10 翻譯甚至把「這對我的職涯發展非常重要」也一起翻成英文（Gemini 4 分）。
- **中文 vs 英文系統提示**：規則式 18 vs 17、Gemini 都是 8.65，差異只有 1 題（英文版的 A06 把程式碼寫成一行，無法執行），語言本身影響不大。
- **兩種評估的互補**：規則式只看「有沒有答對、格式對不對」，20 題中有 4 種技巧已達滿分；Gemini 能分辨「答對但多寫了捏造內容」（A09）與「答對但格式不理想」，因此用來區分技巧之間的細微差異。

### A-3 模型量化（基礎）
使用預設 Prompt：

| 精度 | 權重記憶體 | 規則式通過 | Gemini 平均 | batch 10 生成速度 | batch 1 生成速度 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| FP16（預設） | 7,672 MiB | 20 / 20 | 9.65 | 44.6 tok/s | 15.1 tok/s |
| INT8（LLM.int8） | 4,207 MiB（−45%） | 20 / 20 | 9.65 | 13.0 tok/s | 4.6 tok/s |
| NF4（4-bit） | 2,475 MiB（−68%） | 20 / 20 | 9.35 | 32.7 tok/s | 11.4 tok/s |

（`Output/partA_quant_summary.csv`、`partA_memory_profile.csv`、`partA_quantization.png`）

- **INT8 品質與 FP16 幾乎相同**：20 題全對、Gemini 同為 9.65；20 題中有 6 題與 FP16 的回答一字不差，其餘多為措辭差異。但速度只有 FP16 的 30%。LLM.int8 會把超過門檻的離群值拆出來用 FP16 計算，每一層多了拆分與合併，在 T4 上反而最慢。
- **NF4 記憶體最省、速度居中、品質小幅下降**：Gemini 9.35（−0.30），主要來自 A09（從 7 分降到 4 分：NF4 在說明阿貝爾獎時陷入反覆修正的循環）與 A11（「早起的鳥兒撈到蟲」，用字不自然）。NF4 的 4-bit 權重在生成時即時反量化，batch 1 時約為 FP16 的 75% 速度。
- **結論**：記憶體不夠時優先選 NF4（只用 FP16 約 1/3 的權重記憶體，品質損失很小）；INT8 雖然品質最接近 FP16，但在 T4 上太慢，不划算。

### A-4 預設精度的 GPU 記憶體（進階）
FP16、預設 Prompt，5 題 × 最多 256 token、batch = 1，每 20 ms 取樣（`Output/partA_memory_timeline.png`）：

| 指標 | FP16 | 說明 |
| :--- | ---: | :--- |
| 平均 allocated | **7,701 MiB** | 權重 7,672 MiB + KV cache 與中間值，只多約 30 MiB |
| 峰值 allocated | 7,742 MiB | 單筆生成時 KV cache 很小（每個 token 144 KiB） |
| 平均 reserved | 9,304 MiB | PyTorch 快取分配器保留的記憶體池（先前 batch 10 生成時擴張後不會縮回） |
| 平均 NVML 本程序用量 | 9,436 MiB | nvidia-smi 看到的數字 = reserved + CUDA context |

batch 10（20 題、最多 768 token）時平均 allocated 為 8,163 MiB、峰值 8,682 MiB：KV cache 隨 batch 與長度增加約 0.5–1 GB。也就是說，4B 模型單筆推論的記憶體幾乎全是權重；要省記憶體，量化權重的效果遠大於縮短輸出。

### A-5 優化 Prompt + 低精度量化（進階）
最佳 Prompt（Zero-shot CoT）× 三種精度：

| 設定 | 規則式通過 | Gemini 平均 | 平均輸出 token | batch 1 平均 allocated | batch 10 平均 allocated | batch 1 速度 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 + 預設 Prompt（基準） | 20 / 20 | 9.65 | 146 | 7,701 MiB | 8,163 MiB | 15.1 tok/s |
| FP16 + CoT | 20 / 20 | 9.30 | 233 | 7,711 MiB | 8,367 MiB | 15.1 tok/s |
| INT8 + CoT | 20 / 20 | 9.60 | 214 | 4,251 MiB | 4,707 MiB | 4.7 tok/s |
| **NF4 + CoT** | **20 / 20** | **9.45** | 259 | **2,569 MiB（−67%）** | **3,216 MiB（−61%）** | 11.4 tok/s |

- **NF4 + CoT 用 1/3 的記憶體維持同樣的正確率**：規則式 20/20，Gemini 9.45，比 NF4 + 預設 Prompt（9.35）高，也比 FP16 + CoT（9.30）高。CoT 讓 4-bit 模型逐步推導，部分抵銷了量化造成的精度損失。
- **代價是時間**：NF4 + CoT 的輸出比基準多約 77%（259 vs 146 token），加上 NF4 單筆速度只有 FP16 的 75%，每題的延遲約為基準的 2.3 倍；記憶體方面，CoT 較長的輸出在 batch 1 時只多了約 10 MiB 的 KV cache。
- INT8 + CoT 的 Gemini 分數最高（9.60），但速度只有 4.7 tok/s，實用性低。
- 整體而言，在 16 GB 以下的 GPU 上，「NF4 + CoT」是記憶體與品質最好的折衷；若延遲重要、記憶體足夠，FP16 + 預設 Prompt 仍是最佳選擇。

---

## 作業 B：視覺多模態大模型（基礎 20 分）

### 設計
| 項目 | Qwen3-VL-2B-Instruct | SmolVLM2-2.2B-Instruct |
| :--- | :--- | :--- |
| 參數 | 2.1 B | 2.2 B |
| 架構 | ViT + MLP merger 投影到 LLM 的輸入空間（LLaVA 式，簡報第 37 頁），DeepStack 多層視覺特徵 | SigLIP + pixel-shuffle connector + SmolLM2（Idefics3 式） |
| 影像處理 | 原生解析度（640×480 → 含文字共 271 個 token） | 切成 384×384 子圖 + 全圖（640×480 → 1,092 個 token） |
| 精度 | FP16 | FP16 |

- **Benchmark**（`Data/partB_vlm_benchmark.json`）：15 張 COCO val2017 照片（逐張看過後撰寫參考答案）+ 5 張程式產生的合成影像（答案精確）：B01 影像分類、B02–B05 計數（斑馬含邊緣半隻、馬、遙控器、人）、B06–B07 場景文字 OCR（STOP、球衣號碼，背景有干擾數字 182）、B08 顏色、B09 空間關係、B10 多物件偵測、B11 運動辨識、B12 情境推理、B13 細粒度辨識、B14 幻覺探測（長頸鹿照片問有沒有斑馬）、B15 形容照片、B16–B20 合成影像（招牌 OCR、紅色圓形計數、長條圖、左右關係、視覺數學）。
- **評估**：兩個模型使用**完全相同的英文 Prompt**（SmolVLM2 以英文為主）、greedy decoding、最多 128 token、batch = 1。量化指標為每題的規則式判定；商用 LLM 評估為 Gemini 看圖、參考答案與兩個匿名回答，各給 1–10 分。

### B-1 整體比較

| 模型 | 規則式準確率 | COCO 15 題 | 合成 5 題 | Gemini 平均 | 平均延遲 | 延遲中位數 | 平均輸出 token | GPU 記憶體（平均 / 峰值） |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Qwen3-VL-2B** | **18 / 20** | **15 / 15** | 3 / 5 | **8.75** | 2.02 s | **0.24 s** | 36.4 | 4,112 / 4,117 MiB |
| SmolVLM2-2.2B | 17 / 20 | 14 / 15 | 3 / 5 | 8.65 | **1.19 s** | 1.15 s | 7.4 | 4,394 / 4,509 MiB |

（`Output/partB_summary.csv`、`partB_summary.png`、`partB_examples.png`）

- **兩者整體接近，Qwen3-VL 略勝**：Qwen3-VL 的真實照片 15 題全對；SmolVLM2 只錯 B02（把 3 隻斑馬數成 4 隻）。
- **兩個模型的強項不同**：
  - Qwen3-VL 的計數比較準（B02 斑馬、B17 紅色圓形都對；SmolVLM2 把 3 隻斑馬數成 4 隻、7 個紅色圓形數成 10 個）。
  - SmolVLM2 的回答簡潔、格式遵循較好：Qwen3-VL 在開放式問題習慣以「Based on the image provided…」開頭並條列說明，B13、B15 被 Gemini 扣分（超過要求長度），B18 長條圖甚至在列到「Apr: 2…」時被 128 token 上限截斷，沒有答到結論；SmolVLM2 一句話就答對（April, 25）。
- **延遲**：SmolVLM2 每張影像要處理 1,092 個 token（Qwen3-VL 只有 271 個），所以短答題的延遲約 1.0 s，Qwen3-VL 只要 0.2 s；但 Qwen3-VL 開放式問題的長回答把平均延遲拉到 2.0 s。若要求簡短回答，Qwen3-VL 較快。
- **記憶體**：兩者 FP16 權重都約 4.1–4.3 GB，推論時幾乎沒有額外成本（SmolVLM2 的影像 token 多，峰值多約 100 MiB）。

### B-2 逐題分析
- **兩者都錯：B20 視覺數學（27 + 15 = ?）**。Qwen3-VL 回答「2」（完全錯誤）；SmolVLM2 回答「27 + 15 = 42」，計算正確，但題目要求只回答數字，規則取第一個數字判為錯誤，Gemini 也因格式給 7 分。
- **幻覺探測 B14**：兩者都正確回答沒有斑馬。
- **OCR**：STOP、球衣號碼 4（不受背景的 182 干擾）、招牌名稱與電話都正確。
- **Gemini 評審的雜訊**：B10 兩個模型都被扣分，理由是「not in the requested JSON format」，但題目沒有要求 JSON，這是評審本身的錯誤；兩者分數一樣，不影響比較。

---

## 作業 C：以 DPO 微調 LLM（基礎 20 分）

### 優化面向：台灣繁體中文用語
Qwen 系列以簡體中文資料為主，使用者用繁體中文提問時，回答會（1）夾雜簡體字、（2）用繁體字寫出中國大陸用語（軟件、視頻、打印、激光⋯）。本作業以 DPO 讓模型改用台灣用語。

| 項目 | 設定 |
| :--- | :--- |
| 指標（全自動） | 簡體字 = OpenCC 簡轉繁字表中「繁體字形不同」且不是台灣標準字形的字（不會把 台、吃、群 誤判為簡體）；大陸用語 = 99 組人工整理的兩岸用語對照（只收明確的用法，如 軟件／軟體、視頻／影片，不收 支持、項目 這類兩岸都用的詞）；**乾淨回答** = 無簡體字、無大陸用語、且至少 20 個中文字；另外記錄混入日文假名／韓文／帶重音拉丁字母的比例 |
| 偏好資料 | 130 題自行撰寫的繁中問題（科技、網路、交通、飲食、工作、學習）→ 基礎模型作答 → chosen = 把簡體字轉繁體、大陸用語換成台灣用語；rejected = 模型原本的回答（on-policy）。若原回答已經乾淨，rejected = 把台灣用語換回大陸用語，再不行則轉成簡體。共 130 組：on-policy 31 組、換回大陸用語 76 組、轉簡體 23 組；117 組訓練、13 組驗證 |
| 測試 | 另 40 題保留問題（不參與訓練） |
| 通用能力 | 100 題英文題：60 題 MMLU（57 科隨機抽樣）、40 題 GSM8K，與優化面向無關 |
| 訓練 | LoRA r = 16 / α = 32（所有 attention 與 MLP 投影層，33 M 參數）、β = 0.1、lr 1e-4、3 epochs、等效 batch 8、warm-up + cosine；參考模型 = 關閉 LoRA 的同一個模型；依驗證 DPO loss 保留最佳 epoch（兩個版本都是第 3 epoch） |
| 兩個版本 | **DPO**：標準 DPO loss。**DPO+SFT**：DPO loss + 1.0 × chosen 回答的平均每 token NLL（RPO，Pang et al., 2024；TRL 的 `rpo_alpha`）。資料與其他超參數完全相同 |

### C-1 優化面向
40 題保留測試題：

| 模型 | **乾淨回答** | 含大陸用語的回答 | 每題大陸用語數 | 含簡體字的回答 | 每千字簡體字 | 混入其他文字 | 平均中文字數 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 70.0% | 22.5%（9 題） | 0.375 | 7.5%（3 題） | 0.71 | 0% | 141 |
| DPO | 22.5% | **0%** | **0** | **77.5%（31 題）** | **12.22** | 7.5% | 168 |
| **DPO+SFT** | **75.0%** | 7.5%（3 題） | 0.075 | 17.5%（7 題） | 2.27 | 0% | 143 |

（`Output/partC_llm_aspect.csv`、`partC_llm_term_counts.csv`、`partC_llm_simplified_chars.csv`、`partC_llm_results.png`；三個模型對每一題的回答在 `partC_llm_test_responses.csv`）

**DPO：大陸用語完全消失，但模型退化**
- 40 題中再也沒有大陸用語（打印、土豆、激光⋯全部消失），但 31 題開始夾雜簡體字（体、萤、装、档⋯），3 題混入日文（`アプリ`）或德文（`Menü`），也出現無意義的英文片段（如「暫存作業中所需的 ansom」）。
- 訓練資料的 130 個 chosen 中**一個簡體字都沒有**，所以這不是資料錯誤，而是 DPO 本身的已知問題 **likelihood displacement**（Razin et al., 2024）：DPO 只要求 chosen 相對 rejected 的機率變高，當兩者非常相似時（本資料的 chosen 與 rejected 只差幾個詞），模型會同時壓低兩者，把機率推到其他相近的 token 上——例如 `體` 的機率移到字形、語意都相近的簡體 `体`。
- 訓練紀錄證實了這點：訓練後在驗證資料上，chosen 的 log π − log π_ref = **−135**、rejected = −201。偏好拉開了（margin 66、驗證準確率 100%），但 chosen 本身的機率也大幅下降。驗證 DPO loss 一路下降，**偵測不到這種退化**，只有看生成結果才發現。

**DPO+SFT：修正退化，乾淨回答最多**
- 加上 chosen 的 NLL 項後，chosen 的 log-ratio 維持在 **+0.03**（`partC_llm_training_dposft.png` 最右圖：chosen 在 0 附近、rejected 降到約 −45），只壓低 rejected、不再犧牲 chosen。
- 乾淨回答 70% → **75%**，大陸用語的回答 9 題 → 3 題（每題用語數 −80%），沒有任何日文或亂碼，回答長度與 Base 相同（143 vs 141 字）。
- **仍有取捨**：含簡體字的回答從 3 題增加到 7 題，都是零星的單字（例如同一段裡先寫「语气」、後面又寫「語氣」）。SFT 項大幅減輕了 displacement，但沒有完全消除。
- 40 題的樣本小（1 題 = 2.5 個百分點），乾淨回答 70% → 75% 只差 2 題；比較明確的改善是大陸用語的回答數（9 → 3 題）。

### C-2 通用能力
100 題與優化面向無關的英文題：

| 模型 | MMLU（60） | GSM8K（40） | 整體（100） | 原本對、變錯 | 原本錯、變對 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Base | 76.7% | 87.5% | 81% | — | — |
| DPO | 80.0% | 92.5% | 85% | 0 | 4 |
| DPO+SFT | 76.7% | 92.5% | 83% | 0 | 2 |

（`Output/partC_llm_general.csv`）

- 兩個版本都**沒有任何原本答對的題目變錯**，通用能力沒有退步。
- 小幅上升（+2～+4 題）在 100 題的規模下不顯著（約 ±4 個百分點的抽樣誤差）。LoRA 只訓練中文的用語偏好，英文推理能力不受影響；DPO 版雖然中文生成退化，英文題卻沒有受到影響，可見 displacement 集中在被訓練的中文 token 附近。

### C-3 小結
- 規範要求的「量化優化面向是否提升」：DPO+SFT 讓大陸用語的回答減少 2/3、乾淨回答增加，且通用能力不變；標準 DPO 則示範了偏好微調的典型失敗：指標上的目標（大陸用語）完全達成，但模型在指標沒有直接涵蓋的地方（簡體字、其他語言）退化。
- 經驗：偏好資料的 chosen / rejected 差異越小，越需要在 DPO 之外錨定 chosen 的機率（SFT 項），並且必須檢查實際生成結果，不能只看 DPO loss 或 reward accuracy。

---

## 作業 C 進階：以 DPO 微調 VLM（進階 20 分）

### 設計：描述照片時不提及不存在的物件
| 項目 | 設定 |
| :--- | :--- |
| 模型 | Qwen3-VL-2B-Instruct + LoRA（只加在語言模型，17.4 M 參數；視覺編碼器凍結），影像上限 448×448 像素（約 196 個視覺 token） |
| 資料來源 | COCO val2017：150 張訓練影像、80 張評估影像（互不重疊，也不含作業 B 的影像），以 COCO 物件標註判斷物件是否存在 |
| 描述類偏好資料（149 組） | 模型自己的「Describe this image in detail.」描述中，提到 COCO 標註沒有的物件 → rejected = 原描述、chosen = 刪掉幻覺句子的描述（43 組；另 1 組刪掉超過一半而不使用）；沒有幻覺 → chosen = 原描述、rejected = 在中間插入一句「常與畫面物件一起出現、但不存在」的物件（106 組） |
| 是非題偏好資料（300 組） | 每張影像一個存在、一個不存在（adversarial）物件：「Is there a … in the image?」chosen / rejected = 正確 / 錯誤的 Yes / No 回答 |
| DPO | 共 449 組（404 訓練 / 45 驗證）、β = 0.1、LoRA r = 16、lr 5e-5、2 epochs、等效 batch 8；T4 上訓練 18 分鐘 |
| 評估 | POPE：每張評估影像 3 個存在物件、2 個 adversarial 不存在物件、1 個隨機不存在物件，共 480 題，以 Yes / No 的下一個 token 機率判斷；CHAIR_s（含幻覺的描述比例）、CHAIR_i（幻覺物件 / 提到的物件）、物件召回率（描述提到的顯著物件比例） |
| 影像劣化 | 原圖、變暗 ×0.35、變亮 ×1.8、低對比 ×0.4、JPEG 品質 5、高斯模糊 r = 4、低解析度（長邊 64 px 再放大）、高斯雜訊 σ = 50（`Output/partC_vlm_degradations.png`），Base 與 DPO 模型都測 POPE |

基礎模型在 150 張訓練影像中有 43 張的描述出現 COCO 標註以外的物件（29%），所以描述類資料有不少是 on-policy 的真實幻覺。

### CV-1 POPE 與 CHAIR

| 模型 | POPE 準確率 | F1 | Precision | Recall | adversarial 負樣本準確率 | 隨機負樣本準確率 | Yes 比例 | AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.9146 | 0.9172 | 0.8902 | 0.9458 | 0.8313 | 0.9875 | 0.531 | 0.9783 |
| **DPO** | **0.9375** | **0.9370** | **0.9449** | 0.9292 | **0.9313** | 0.9750 | 0.492 | 0.9729 |

| 模型 | CHAIR_s ↓ | CHAIR_i ↓ | 物件召回率 ↑ | 每段描述提到的物件數 | 平均字數 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.250 | 0.118 | 0.624 | 2.75 | 147 |
| **DPO** | **0.238** | **0.091** | **0.670** | 3.01 | 94 |

（`Output/partC_vlm_pope.csv`、`partC_vlm_chair.csv`、`partC_vlm_results.png`；80 張影像的描述與判定在 `partC_vlm_descriptions.csv`、範例在 `partC_vlm_examples.png`）

- **POPE 準確率 +2.3 個百分點，主要來自最難的 adversarial 負樣本（+10 個百分點）**：adversarial 負樣本是「常與畫面物件一起出現、但實際不存在」的物件（例如餐桌場景問有沒有椅子），Base 只答對 83%；DPO 後提升到 93%，Precision 0.890 → 0.945。
- **沒有變成一律回答「沒有」**：Yes 比例 0.53 → 0.49（正負樣本各半時理想值為 0.5），存在物件的 Recall 只小降 1.7 個百分點，AUC 幾乎不變（0.978 → 0.973），表示模型的判別能力相同，主要是把「不確定時答 Yes」的傾向修正了。
- **描述的幻覺減少**：CHAIR_i 0.118 → 0.091（幻覺物件的比例相對減少 23%），CHAIR_s 0.250 → 0.238（80 段中含幻覺的從 20 段降到 19 段）。
- **不是只靠描述變短**：DPO 後描述變短（147 → 94 字），但物件召回率反而上升（0.62 → 0.67），每段提到的物件也更多（2.75 → 3.01）。Base 的描述常以大段氛圍、材質描寫開頭（多數寫滿 192 token 上限），DPO 後改成直接列出畫面中的物件，資訊密度較高。
- 範例（`partC_vlm_examples.png`）：辦公桌場景中 Base 提到不存在的 cell phone、mouse，DPO 版本沒有；但 DPO 版本改提到了 chair、dining table（COCO 標註沒有），幻覺沒有完全消失，只是變少。

### CV-2 影像劣化

| 條件 | Base 準確率 | DPO 準確率 | 差異 | Base AUC | DPO AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 原圖 | 0.9146 | 0.9375 | +2.3 | 0.978 | 0.973 |
| 變暗 ×0.35 | 0.9229 | 0.9271 | +0.4 | 0.975 | 0.971 |
| 變亮 ×1.8 | 0.9021 | 0.9187 | +1.7 | 0.967 | 0.961 |
| 低對比 ×0.4 | 0.9187 | 0.9271 | +0.8 | 0.974 | 0.970 |
| JPEG q = 5 | 0.9000 | 0.9062 | +0.6 | 0.968 | 0.959 |
| 模糊 r = 4 | 0.9062 | 0.9083 | +0.2 | 0.973 | 0.966 |
| 低解析度 64 px | **0.8729** | **0.8812** | +0.8 | **0.952** | **0.936** |
| 雜訊 σ = 50 | 0.9229 | 0.9333 | +1.0 | 0.975 | 0.968 |
| **7 種劣化平均** | **0.9065** | **0.9146** | **+0.8** | | |

（`Output/partC_vlm_pope.csv`、`partC_vlm_results.png` 右圖）

- **低解析度影響最大**：長邊縮到 64 px 再放大後，Base −4.2、DPO −5.6 個百分點，AUC 也最低；小物件在 64 px 下已無法辨識。JPEG q = 5 與模糊次之（約 −1 ～ −3 個百分點）。
- **明暗、對比與雜訊幾乎沒有影響**：Base 在變暗與加雜訊後的準確率甚至略高於原圖，但 480 題的抽樣誤差約 ±1.3 個百分點，這種小幅上升屬於誤差範圍；AUC 在所有劣化下都比原圖低，代表判別能力實際上都有輕微下降。
- **DPO 在 8 種條件下都高於 Base**（平均 +0.8 個百分點），但優勢在劣化時縮小（原圖 +2.3 → 劣化平均 +0.8），且 DPO 在低解析度下降得更多：DPO 學到的是「不確定時傾向回答沒有」，影像越模糊，存在的物件也越容易被判為沒有（低解析度的 Recall：Base 0.854、DPO 0.833）。

---

## 執行紀錄
1. **作業一**在 Colab 預設的 transformers 版本上執行完成。
2. **作業 B** 的 SmolVLM2 在 Colab 預設版本上報錯（`inputs_merger` 以 vision patch size 重塑影像特徵，與 SmolVLM2 每個子圖 81 個 token 不符，是該版 transformers 的問題）。改裝本機測試過的 transformers 5.15.0、重新啟動後完成作業 B（`requirements.txt` 已固定此版本）。重新啟動前曾不小心重跑一次作業一的 cell：12 組生成與 Gemini 評分都直接讀快取，結果不變，但部署示範重新生成時，前一次錯誤的 SmolVLM2 模型仍佔用 GPU，所以 `partA_deploy_demo.csv` 的峰值記憶體（12 GB）偏高，記憶體量測以 A-4 為準。
3. **作業 C** 建立 LoRA 時，新版 peft 因 Colab 預裝的舊版 torchao（0.10）報錯；本作業不使用 torchao，已在 `Code/dpo.py` 中讓 peft 忽略不相容的 torchao。同一個問題也讓 Notebook 中的 `gpu_benchmark.py` 沒有完成。
4. 檢查第一次的結果後做了兩項修正：
   - **作業一 A17（表格推理）的規則式檢查誤判**：原規則要求答案中只能出現「小強」，但沒有「最終答案：」標記的回答會在計算過程中列出三個人的分數，導致 6 個答對的回答（Gemini 都給 10 分）被判錯。改為檢查「最後提到的人名」是否為小強，並以已存的回答重新計分（生成內容不變，其餘 234 個判定都不變）。**Notebook 中作業一的表格是修正前的數字**（baseline 19/20 等），以 `Output/partA_*.csv` 為準。
   - **作業 C 的 DPO 退化**（見 C-1）：新增 DPO+SFT 版本，只重新執行作業 C 的 cell；標準 DPO 版本的結果沿用第一次的快取，並以存下的 adapter 補量 chosen / rejected 的 log-ratio。Notebook 中作業 C 的輸出來自第二次執行。

## 限制與說明
1. 作業一的「最佳 Prompt」是在同一組 20 題上挑選的；20 題中有 4 種設定的規則式指標已達滿分，1 題約等於 5 個百分點，Prompt 間的細部差異主要依賴 Gemini 分數。
2. Gemini 評審一次看同一題的所有回答（匿名、打亂順序）。評審有可觀察到的偏差：例如把沒有「最終答案：」標記的正確回答扣分（A02 baseline 7 分），以及 B10 誤以為題目要求 JSON；分數適合比較設定間的相對差異。
3. 作業 C 的大陸用語表與 CHAIR 物件詞表都是人工整理的保守版本，會漏掉表外的用語／物件，但不會把明確的台灣用語或標註內物件算錯。
4. CHAIR 依 COCO 標註判斷，標註遺漏的物件會被算成幻覺（文獻中的已知限制）；Base 的描述多數寫滿 192 token 上限，只保留完整的句子計算。
5. 作業 C 的測試只有 40 題、通用能力 100 題、POPE 480 題；每個實驗只執行一次（固定 seed、greedy decoding），小於約 2 題（LLM）或 1.3 個百分點（POPE）的差異不能視為顯著。
