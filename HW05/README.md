# 國立陽明交通大學 多模態影像資料處理 (MMIP) 作業五技術報告
## 多模態深度學習神經網路：影像檢索、圖文多模態分類、CLIP Image Captioning

- **學生**：林孟霆（`315833033`）
- **規範依據**：課程投影片 `多模態影像資料處理（MMIP）第五堂_多模態深度學習神經網路` 第 54–56 頁
- **配分**：Quiz 1（基礎 20 + 進階 10）、Quiz 2（基礎 20 + 進階 15）、Quiz 3（基礎 20 + 進階 10）
- **執行環境**：Google Colab **Tesla T4 GPU**、PyTorch 2.11.0+cu130

> [!NOTE]
> **給助教的導覽 (For TA & Reviewers)**：
> - **文字報告與分析**：本 `README.md`。
> - **程式執行結果**：`HW05_notebook.ipynb` 已在 Colab T4 上執行完成，所有圖表與表格都內嵌在輸出中。
> - **圖表與結果檔**：`Output/`；**每個 epoch 的訓練紀錄**：`Checkpoints/*.json`；**Week 4 對照資料**：`Data/week4_test_metrics.csv`。
> - **AI 協作紀錄**：`AI/AI_collaboration_log.md`。
> - 所有數字皆來自實際執行結果；Testing Dataset 只用於最終評估，每個模型保留的權重都是依 **Validation** 挑選的。
> - 模型權重（`Checkpoints/*.pt`）、抽好的特徵（`Data/*.pt`）與資料集沒有放進 git；重新執行會自動下載並重建（Colab T4 約 1 小時）。

### 評分項目對照表（各小題在哪裡看）

| 題目（配分） | 規範要求 | 文字說明與分析 | 圖表 / 結果檔 |
| :--- | :--- | :--- | :--- |
| Quiz 1 基礎（5） | 準備影像檢索資料集 | [Quiz 1 設計](#quiz-1影像檢索任務基礎-20-分--進階-10-分) | — |
| Quiz 1 基礎（10 + 5） | 訓練檢索模型；展示檢索結果 | [1-1、1-3](#1-1-檢索訓練的效果基礎) | `Output/quiz1_training_curves.png`、`quiz1_retrieval_examples_*.png` |
| Quiz 1 進階（5 + 5） | CNN 與 ViT 兩種 Backbone；比較檢索表現 | [1-2](#1-2-cnn-vs-vit進階) | `Output/quiz1_retrieval_results.csv`、`quiz1_recall_at_k.png` |
| Quiz 2 基礎（5） | 準備影像與文字的多模態資料集 | [Quiz 2 設計](#quiz-2多模態深度學習神經網路基礎-20-分--進階-15-分) | — |
| Quiz 2 基礎（10 + 5） | 影像與文字 Backbone 萃取特徵、融合後分類；訓練與推論 | [2-1、2-3](#2-1-訓練與推論基礎) | `Output/quiz2_training_curves.png`、`quiz2_inference_examples*.png`、`quiz2_custom_text_inference.*` |
| Quiz 2 進階（10 + 5） | 更換 Backbone；比較不同組合 | [2-2](#2-2-backbone-組合比較進階) | `Output/quiz2_backbone_comparison.csv` |
| Quiz 3 基礎（5 + 10 + 5） | 資料準備；CLIP Encoder ＋ 文字 Decoder；訓練並產生描述 | [3-1](#3-1-訓練與產生的描述基礎) | `Output/quiz3_training_curves.png`、`quiz3_sample_captions.png`、`quiz3_test_captions.csv` |
| Quiz 3 進階（10） | 比較 Week 4 與本次的 BLEU、BERTScore | [3-2](#3-2-week-4-vs-week-5-bleu--bertscore進階) | `Output/quiz3_week4_vs_clip*.csv`、`quiz3_week4_vs_clip.png` |

---

## 目錄結構

```text
HW05/
├── HW05_main.py              # 一鍵執行 Quiz 1–3
├── HW05_notebook.ipynb       # 已在 Colab T4 執行完成的 Notebook
├── README.md
├── requirements.txt
├── Code/
│   ├── common.py             # 裝置、快取、學習率排程
│   ├── data.py               # CUB-200-2011、Flickr8k、caption tokenizer／詞彙表
│   ├── encoders.py           # 凍結的預訓練編碼器：ResNet-50、ViT-B/16、CLIP、BERT
│   ├── quiz1_retrieval.py    # Quiz 1 影像檢索（Normalized Softmax、Recall@K、mAP）
│   ├── quiz2_multimodal.py   # Quiz 2 圖文相符分類（特徵融合 MLP、4 種 Backbone 組合）
│   ├── caption_models.py     # Week 4 的 LSTM／Transformer decoder 與 BLEU／BERTScore（只改特徵維度）
│   └── quiz3_clip_caption.py # Quiz 3 CLIP Image Captioning 與 Week 4 比較
├── Data/                     # 資料來源說明與 Week 4 對照結果
├── Checkpoints/              # 訓練紀錄 (*.json)
├── Output/                   # 圖表與 CSV
└── AI/AI_collaboration_log.md
```

---

## Quiz 1：影像檢索任務（基礎 20 分 + 進階 10 分）

### 設計
| 項目 | 設定 |
| :--- | :--- |
| 資料集 | CUB-200-2011（200 種鳥、11,788 張）。標準的「未見類別」檢索切法：第 1–90 種訓練（5,265 張）、第 91–100 種驗證（599 張）、第 101–200 種測試（5,924 張），**測試鳥種完全沒在訓練中出現** |
| 模型 | ImageNet 預訓練 Backbone（ResNet-50 或 ViT-B/16）＋ 512 維嵌入層，輸出 L2 正規化向量 |
| 損失函數 | Normalized Softmax（Zhai & Wu, 2019）：每個訓練類別一個可學習的代表向量，cosine 相似度除以溫度 0.05 後做 cross-entropy |
| 訓練 | AdamW（backbone lr 1e-4、嵌入層與代表向量 1e-3）、warm-up + cosine、12 epochs、batch 64、混合精度；依驗證 Recall@1 保留最佳 epoch |
| 檢索 | 每張測試影像當查詢，從其餘 5,923 張測試影像中依 cosine 相似度排序 |
| 評估 | Recall@1/2/4/8（前 K 名中是否有同種鳥）、mAP；以「沒經過檢索訓練的 ImageNet 特徵」作為 baseline |

### 結果（測試集：100 種沒看過的鳥、5,924 張）

| 模型 | 嵌入維度 | R@1 | R@2 | R@4 | R@8 | mAP | 參數量 | 訓練時間 | 推論速度 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-50 預訓練特徵（未訓練） | 2048 | 0.5029 | 0.6389 | 0.7557 | 0.8425 | 0.2015 | 23.5 M | — | 152 張/秒 |
| **ResNet-50 + 檢索訓練** | 512 | 0.6035 | 0.7265 | 0.8256 | 0.8980 | 0.3027 | 24.6 M | 7.9 分 | 166 張/秒 |
| ViT-B/16 預訓練特徵（未訓練） | 768 | 0.5663 | 0.6955 | 0.8003 | 0.8766 | 0.2206 | 85.8 M | — | 146 張/秒 |
| **ViT-B/16 + 檢索訓練** | 512 | **0.6403** | **0.7591** | **0.8406** | **0.9024** | **0.3317** | 86.2 M | 12.6 分 | 149 張/秒 |

（`Output/quiz1_retrieval_results.csv`、`Output/quiz1_recall_at_k.png`）

#### 1-1 檢索訓練的效果（基礎）
- 兩種 Backbone 經過檢索訓練後都明顯進步：ResNet-50 的 R@1 從 0.5029 → 0.6035（**+10.1 個百分點**）、mAP 從 0.2015 → 0.3027；ViT-B/16 的 R@1 從 0.5663 → 0.6403（**+7.4 個百分點**）、mAP 從 0.2206 → 0.3317。
- 測試鳥種在訓練中完全沒出現過，所以進步代表模型學到的是「怎麼分辨鳥種的外觀特徵」，而不是記住特定類別。
- mAP 比 R@1 低很多（約 0.30 vs 0.60）是正常的：R@1 只看第 1 名，mAP 要求同種的其餘約 58 張影像全部排在前面。
- 訓練曲線（`quiz1_training_curves.png`）：訓練 loss 在 12 epochs 內從約 2 降到 0.02，幾乎完全擬合 90 個訓練鳥種。驗證 R@1 在 ResNet-50 第 10 epoch、ViT-B/16 **第 4 epoch** 達到最高，之後 ViT 的驗證 R@1 下降，表示對訓練鳥種過擬合；依驗證挑選 epoch 可避免用到過擬合的權重。
- 驗證 R@1（約 0.89）遠高於測試 R@1（約 0.60），原因是驗證集只有 10 種鳥、599 張，每次查詢的干擾影像少很多，兩者不能直接比較；驗證集只用來挑 epoch。

#### 1-2 CNN vs ViT（進階）
- **ViT-B/16 在所有指標都比 ResNet-50 好**：訓練後 R@1 高 3.7 個百分點（0.6403 vs 0.6035）、mAP 高 2.9 個百分點；沒訓練時也高 6.3 個百分點（0.5663 vs 0.5029），表示 ViT 的預訓練特徵本身就比較能分辨細部外觀。
- 差距隨 K 變大而縮小：R@8 兩者幾乎相同（0.9024 vs 0.8980）。ViT 的優勢主要在「第 1 名就找對」，也就是最難分辨的相近鳥種。
- 檢索訓練對 ResNet-50 的幫助較大（+10.1 vs +7.4 個百分點），縮小了兩者的差距。
- 代價：ViT-B/16 的參數量是 ResNet-50 的 3.5 倍（86.2 M vs 24.6 M），訓練時間 1.6 倍（12.6 vs 7.9 分鐘），推論每秒少約 17 張；ViT 也比較早開始過擬合（第 4 epoch）。
- 結論：如果精確度最重要，選 ViT-B/16；如果要兼顧速度與記憶體，ResNet-50 經過檢索訓練後的 R@4／R@8 已經很接近 ViT。

#### 1-3 檢索結果展示
`Output/quiz1_retrieval_examples_vit_b_16.png`、`quiz1_retrieval_examples_resnet50.png`：每列一張查詢影像與前 5 名檢索結果（綠框為同種、紅框為不同種）。

- 以 ViT-B/16 的範例為例。成功例：Blue winged Warbler 前 5 名有 4 張同種；Mourning Warbler 第 1、3 名正確。
- 錯誤都發生在外觀非常相似的鳥：Seaside Sparrow 的錯誤結果是 White throated Sparrow、Savannah Sparrow 等其他麻雀；Red eyed Vireo 被找成 Warbling Vireo、White eyed Vireo；Blue winged Warbler 的第 2 名是外型很像的 Golden winged Warbler。
- Western Wood Pewee 的 5 張結果全錯，都是「藍天背景、停在樹枝或電線上的小型鳥」，表示模型有時會被相同的背景與姿勢影響。
- 同樣 5 張查詢影像，ResNet-50 的結果有幾張反而比 ViT-B/16 好：Western Wood Pewee 前 5 名有 3 張正確（ViT 是 0 張），Seaside Sparrow 第 1 名就正確（ViT 是第 2 名）；但 ResNet-50 的 Blue winged Warbler 前 5 名中有 2 張是同樣全身黃色的 Prothonotary Warbler。單張範例會有個別差異，兩種 Backbone 的優劣以 5,924 張查詢的平均指標（上表）為準。

---

## Quiz 2：多模態深度學習神經網路（基礎 20 分 + 進階 15 分）

### 設計（對應簡報第 44–45 頁）
| 項目 | 設定 |
| :--- | :--- |
| 資料集 | 由 Flickr8k 自行建立圖文配對：每張影像配自己的 5 句描述（相符，1）與 5 句其他隨機影像的描述（不相符，0）。訓練 60,000 對、驗證 10,000 對、測試 10,000 對，正負各半；影像切分與 Week 4 相同 |
| 特徵萃取 | 凍結的影像 Backbone 與文字 Backbone，每張影像、每句描述只萃取一次並快取 |
| 融合與分類 | 兩種特徵各自 LayerNorm → 串接 → MLP（512 → 128 → 1）→ 相符機率 |
| 訓練 | AdamW（lr 1e-3）、20 epochs、BCE；依驗證 ROC-AUC 保留最佳 epoch |
| Backbone 組合 | ResNet-50 + BERT（基準）、ViT-B/16 + BERT（換影像 Backbone）、CLIP 影像 + BERT、CLIP 影像 + CLIP 文字 |
| 困難測試集 | 另建 10,000 對測試題：不相符描述改從「主角相同」的其他影像挑（例如狗的照片配上另一張狗照片的描述），必須看懂動作與場景才能分辨；只用於測試 |
| 推論展示 | (1) 測試影像配上正確與錯誤描述的相符機率；(2) 自訂文字查詢：對 1,000 張測試影像逐一計算相符機率，列出最高的 3 張 |

### 結果（測試集 10,000 對；困難測試集 10,000 對）

| Backbone 組合 | 影像／文字維度 | Accuracy | Precision | Recall | F1 | ROC-AUC | 困難測試 Accuracy | 困難測試 ROC-AUC | 最佳 epoch | 融合模型參數 |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-50 + BERT（基準） | 2048 / 768 | 0.8999 | 0.9115 | 0.8858 | 0.8985 | 0.9585 | 0.8040 | 0.8709 | 19 | 1.51 M |
| ViT-B/16 + BERT | 768 / 768 | 0.8833 | 0.8995 | 0.8630 | 0.8809 | 0.9542 | 0.7898 | 0.8618 | 11 | 0.86 M |
| CLIP 影像 + BERT | 512 / 768 | 0.9325 | 0.9359 | 0.9286 | 0.9322 | 0.9776 | 0.8365 | 0.9056 | 18 | 0.72 M |
| **CLIP 影像 + CLIP 文字** | 512 / 512 | **0.9579** | **0.9556** | **0.9604** | **0.9580** | **0.9901** | **0.8904** | **0.9542** | 19 | 0.59 M |

（`Output/quiz2_backbone_comparison.csv`；特徵已快取，每組融合模型訓練 20 epochs 只需 5–7 秒）

#### 2-1 訓練與推論（基礎）
- 四組模型的驗證 ROC-AUC 都在前 5 個 epoch 內快速上升、之後趨於平穩（`quiz2_training_curves.png`），沒有明顯過擬合。
- **配對推論**（`quiz2_inference_examples.png`，CLIP + CLIP）：4 張測試影像的正確描述相符機率都是 1.00，錯誤描述都是 0.00。`quiz2_inference_examples_resnet_bert.png` 是基準組合在相同影像上的結果。
- **自訂文字查詢**（`quiz2_custom_text_inference.png`／`.csv`，CLIP + CLIP）：對 1,000 張測試影像逐一計算相符機率，列出最高的 3 張與該影像的人工描述：

| 自訂文字 | 相符機率 > 0.5 的影像數 | 前 3 名影像（人工描述） | 判讀 |
| :--- | ---: | :--- | :--- |
| a dog runs across the grass | 60 | 在長草中奔跑的狗；在草地上跑的黑狗；叼著球在草地上跑的狗 | 正確 |
| a child is playing in the water | 43 | 戴泳鏡的小孩；在灑水中玩的男孩；甩頭髮上水的女孩 | 正確（第 3 名是成年女性，但場景相符） |
| a man rides a bicycle on the street | 16 | 三張都是騎腳踏車的人 | 正確 |
| a black cat sleeping on a sofa | **4** | 躺在墊子上的狗；趴在地上的狗；坐在石板路上的狗 | **失敗例** |

- **失敗例分析**：「黑貓睡在沙發上」的前 3 名都是**躺著的狗**，第 1 名的相符機率甚至是 1.00。原因：
  1. 測試集幾乎沒有貓：1,000 張測試影像的第一句人工描述中，提到 cat／kitten 的只有 2 句，提到 dog／dogs 的有 228 句。
  2. 訓練時的不相符描述是隨機抽的，模型很少需要分辨「貓 vs 狗」這種細節，學到的是「躺著的動物＋室內／墊子」這類較粗的對應，因此會對相似場景給出過度自信的機率。
  3. 相符機率 > 0.5 的影像只有 4 張（其他三個查詢有 16–60 張），模型對這個查詢整體的信心確實比較低。

#### 2-2 Backbone 組合比較（進階）
- **最佳組合是 CLIP 影像 + CLIP 文字**：ROC-AUC 0.9901、Accuracy 0.9579，比基準 ResNet-50 + BERT 高 3.2 個百分點 AUC、5.8 個百分點 Accuracy；融合模型的參數量反而最少（0.59 M）。原因是 CLIP 的影像與文字編碼器經過大量圖文配對的對比式學習，兩種特徵本來就在同一個空間對齊（簡報第 47 頁），融合 MLP 只需學很簡單的比較。
- **只換影像 Backbone**（文字固定為 BERT）：
  - ResNet-50 → CLIP 影像：AUC +0.0191、困難測試 AUC +0.0347。CLIP 影像特徵是用文字描述訓練的，包含較多「描述得出來」的內容（動作、場景、人數）。
  - ResNet-50 → ViT-B/16：AUC −0.0043、困難測試 AUC −0.0091，ViT 略差。torchvision 的 ViT-B/16 只用 ImageNet 分類訓練，取 [CLS] token 作為特徵；我們推測這個特徵主要保留「是什麼物體」，而 ResNet-50 全域平均池化的 2048 維特徵保留較多場景資訊。不過差距很小，且只有一個 seed，不能視為顯著差異。
- **只換文字 Backbone**（影像固定為 CLIP）：BERT → CLIP 文字，AUC +0.0125、**困難測試 AUC +0.0486**。在困難測試中，換文字編碼器的幫助比換影像編碼器更大，表示「兩邊特徵是否對齊」是分辨細節的關鍵。
- **困難測試集拉開差距**：一般測試四組的 AUC 只差 0.036（0.9542–0.9901），困難測試差到 0.092（0.8618–0.9542）。從一般測試到困難測試的 AUC 下降幅度：ResNet-50 + BERT −0.088、ViT-B/16 + BERT −0.092、CLIP + BERT −0.072、CLIP + CLIP 只有 −0.036，對齊的特徵空間最能分辨「主角相同但動作或場景不同」的描述。

---

## Quiz 3：CLIP Image Captioning（基礎 20 分 + 進階 10 分）

### 設計
| 項目 | Week 4 | Week 5（本次） |
| :--- | :--- | :--- |
| 資料 | Flickr8k 6,000 / 1,000 / 1,000 | 相同 |
| Image Encoder（凍結） | ImageNet ResNet-50，49 × 2048 | **CLIP ViT-B/32，50 × 768**（[CLS] + 7×7 patch） |
| Text Decoder | LSTM、3 層 Transformer | 相同架構，從頭訓練 |
| 詞彙表、超參數 | 出現 ≥ 5 次（2,533 字）、15 epochs、AdamW lr 3e-4、batch 128 | 相同 |
| 解碼與評估 | Greedy、Beam (k = 3)；BLEU-1~4、BERTScore（RoBERTa-large rescaled） | 相同 |

除了影像編碼器之外全部相同，因此 BLEU、BERTScore 的差異可以直接歸因於「CLIP 編碼器 vs ImageNet ResNet-50 編碼器」。兩種 decoder 使用影像特徵的方式也和 Week 4 相同：LSTM 以 50 個 token 的**平均值**初始化隱藏狀態；Transformer 以 cross-attention 讀取全部 50 個 token。

#### 3-1 訓練與產生的描述（基礎）
| 模型 | 最佳 epoch（驗證 BLEU-4） | 驗證 BLEU-4 | 第 15 epoch train / val loss | 訓練時間 |
| :--- | ---: | ---: | :---: | ---: |
| CLIP + LSTM | 13 | 0.1704 | 2.699 / 2.836 | 1.9 分 |
| CLIP + Transformer | 8 | 0.2008 | 2.141 / 2.724 | 9.9 分 |
| *Week 4 ResNet-50 + LSTM* | *13* | *0.1729* | *2.543 / 2.815* | *1.8 分* |
| *Week 4 ResNet-50 + Transformer* | *8* | *0.1886* | *1.864 / 2.929* | *9.9 分* |

（`Checkpoints/quiz3_clip_*.json`、`Data/quiz4_cnn_*.json`、`Output/quiz3_training_curves.png`）

- 兩個模型都順利收斂，在 1,000 張測試影像上產生描述（`Output/quiz3_test_captions.csv`，含 4 種解碼方式）。
- 範例（`Output/quiz3_sample_captions.png`）：
  - #5（棕色狗在草地上跑）：Week 4 與 CLIP 的兩個模型都正確描述「a brown dog is running in/through the grass」。
  - #2（一男一女穿著節慶服裝跳舞）：CLIP Transformer 描述出「a man ... and a woman ... in front of a crowd」，比 Week 4 的「a group of people are walking down a street」更貼近內容；CLIP LSTM 則出現重複字句（「a man in a white shirt and a woman in a white shirt and a man in a white shirt」）。
  - #8（黑狗叼著白色物體從水中走上沙灘）：CLIP Transformer 誤認為「swimming in the water」，Week 4 的「running through the sand」反而較正確。
  - 單張範例好壞參半，整體比較以下面 1,000 張的指標為準。

#### 3-2 Week 4 vs Week 5 BLEU / BERTScore（進階）

| 解碼方式 | 編碼器 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | BERTScore F1 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| LSTM greedy | Week 4 ResNet-50 | 0.5866 | 0.4127 | 0.2787 | 0.1866 | 0.5680 |
| | Week 5 CLIP | 0.5828 | 0.4031 | 0.2636 | 0.1737 | 0.5604 |
| LSTM beam3 | Week 4 ResNet-50 | 0.6183 | 0.4414 | 0.3077 | 0.2119 | 0.5838 |
| | Week 5 CLIP | 0.5980 | 0.4190 | 0.2853 | 0.1931 | 0.5697 |
| Transformer greedy | Week 4 ResNet-50 | 0.5847 | 0.4091 | 0.2784 | 0.1882 | 0.5621 |
| | Week 5 CLIP | 0.6260 | 0.4448 | 0.3038 | 0.2038 | 0.5828 |
| Transformer beam3 | Week 4 ResNet-50 | 0.6164 | 0.4373 | 0.3053 | 0.2087 | 0.5803 |
| | **Week 5 CLIP** | **0.6411** | **0.4637** | **0.3240** | **0.2243** | **0.5933** |
| （參考）BLIP-base 預訓練模型 | Week 4 | 0.6420 | 0.4910 | 0.3636 | 0.2672 | 0.6067 |

| 解碼方式 | BLEU-4 變化 | BERTScore 變化 |
| :--- | ---: | ---: |
| LSTM greedy | −0.0129 | −0.0076 |
| LSTM beam3 | −0.0188 | −0.0141 |
| Transformer greedy | **+0.0156** | **+0.0207** |
| Transformer beam3 | **+0.0156** | **+0.0130** |

（`Output/quiz3_week4_vs_clip.csv`、`quiz3_week4_vs_clip_delta.csv`、`quiz3_week4_vs_clip.png`）

- **CLIP + Transformer 全面優於 Week 4**：BLEU-1~4 與 BERTScore 在 greedy 與 beam3 都進步，BLEU-4 +1.56 個百分點，BERTScore +0.013～+0.021。本次最佳（Transformer beam3，BLEU-4 0.2243）也超過 Week 4 最佳（LSTM beam3，0.2119）。
  - 訓練紀錄也支持這點：CLIP + Transformer 第 15 epoch 的 val loss 2.724 低於 Week 4 的 2.929，train／val loss 差距也較小（0.58 vs 1.07），表示 CLIP 的 token 讓 decoder 較不容易過擬合。CLIP 的影像特徵是和文字一起訓練的，比 ImageNet 分類特徵包含更多「人、動作、場景」這類描述會用到的資訊。
- **CLIP + LSTM 反而比 Week 4 差**（BLEU-4 −1.3～−1.9 個百分點，BERTScore −0.008～−0.014）。LSTM 只看 50 個 token 的平均值，我們推測原因是：
  - Week 4 的 49 個 ResNet 區域特徵取平均，剛好等於 ResNet-50 原本訓練時使用的全域平均池化特徵；
  - CLIP ViT 只有 [CLS] token（經過最後的 LayerNorm 與投影層）是和文字對齊訓練的，50 個 token 的平均值並不是 CLIP 訓練出來的表示。
  - 佐證：CLIP + LSTM 的 train loss 明顯較高（2.699 vs 2.543），表示從這個平均向量能取得的資訊較少；Transformer 用 cross-attention 自行挑選 token，沒有這個問題。
  - 這個推測沒有另外做實驗驗證（例如改用 [CLS] 或 CLIP 投影後的影像向量初始化 LSTM）；為了讓比較只差影像編碼器一個因素，本次的 decoder 程式維持與 Week 4 相同。
- **與預訓練 BLIP 仍有差距**：CLIP + Transformer 的 BLEU-4 0.2243 仍低於 BLIP-base 的 0.2672。BLIP 的 decoder 也經過上億張圖文配對預訓練；本次的 decoder 只用 6,000 張影像從頭訓練。
- **結論**：換成 CLIP 編碼器的效果取決於 decoder 怎麼使用它的特徵。能讀取所有 token 的 Transformer decoder 可以充分利用 CLIP；只用平均向量的 LSTM 反而退步。

---

## 執行紀錄
- 本機只有 CPU，所有訓練與特徵萃取都在 Google Colab Tesla T4 上執行（PyTorch 2.11.0+cu130）。
- Notebook 先完整執行一次；之後修改 Quiz 2 的自訂文字推論（由「自訂文字 × 隨機測試影像」改為「從 1,000 張測試影像找前 3 名」），**只重新執行 Quiz 2 的 cell**。所以 Notebook 中 Quiz 1、Quiz 3 的輸出來自第一次執行（沒有執行編號）。Quiz 2 兩次的指標與訓練紀錄完全相同，只有訓練秒數不同。
- `Output/quiz3_week4_vs_clip.png` 在本機用同一份 CSV 重畫，只調整圖例位置，避免圖例蓋住數值標籤；數字沒有改變。Notebook 內嵌的是原始版本。

## 限制與說明
1. Quiz 2 訓練用的不相符描述是隨機抽取的，多半和影像差異很大；因此另外以「主角相同」的困難測試集評估。困難測試集以描述中的主角詞（dog、man、woman 等）配對，沒有主角詞的描述仍隨機抽取。
2. 每組實驗只訓練一次（seed = 42），差距很小的比較（例如 ResNet-50 + BERT vs ViT-B/16 + BERT 的 AUC 差 0.004）不能視為顯著差異。
3. Quiz 1 的驗證集只有 10 種鳥、599 張，驗證 R@1 差 0.005 約等於 3 張影像，用來挑 epoch 時有一定雜訊。
4. Quiz 3 中 CLIP + LSTM 退步的原因是根據訓練紀錄的推測，沒有另外做對照實驗驗證。
