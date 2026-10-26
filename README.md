# Multi-Modality Image Processing (MMIP) - 2026 Fall

NYCU College of Artificial Intelligence
- **Course**: Multi-Modality Image Processing (MMIP)
- **Student**: 林孟霆 (Student ID: 315833033)
- **Instructors**: Prof. Chih-Chung Hsu (ACVLab), Eric Cheng (EriXNet)

---

## 作業目錄 (Assignments Index)

- **[HW01: Computer Vision Analysis & OpenCV Fundamentals (點此查閱完整作業報告)](HW01/README.md)**
  - **Quiz 1**: RGB to Grayscale Conversion (OpenCV vs. Vectorized NumPy)
  - **Quiz 2**: Histogram Equalization & Intensity Distribution (OpenCV vs. Pure NumPy LUT)
  - **Quiz 3**: Perspective Transformation & Automated Keystone Rectification (5-Angle Evaluation)
  - **Quiz 4**: SIFT Feature Matching, Seamless Mosaic & Multi-Condition Failure Analysis

- **[HW02: Machine Learning, Deep Learning & Model Evaluation (點此查閱完整作業報告)](HW02/README.md)**
  - **Quiz 1**: Machine Learning Binary Classification (Logistic Regression vs. Random Forest, Threshold Tuning)
  - **Quiz 2**: Deep Learning Credit Default Prediction (PyTorch MLP, Overfitting & Regularization)
  - **Quiz 3**: Model Performance Evaluation (ROC Curve & AUC Comparison)

- **[HW03: Convolutional Neural Networks, Transfer Learning & Explainability (點此查閱完整作業報告)](HW03/README.md)**
  - **Quiz 1**: Image Dataset Preparation (CIFAR-10, Stratified Train / Validation / Test Split)
  - **Quiz 2**: Plain CNN vs. ResNet-18 (Top-1 / Top-5, Per-Class ROC & Macro-AUC, Hyperparameter & Transfer-Learning Experiments)
  - **Quiz 3**: Generalization (Data Augmentation), Kernel Visualization & Grad-CAM

- **[HW04: Time-Series Models & Transformers (點此查閱完整作業報告)](HW04/README.md)**
  - **Quiz 1**: Dataset Preparation (IMDB, Oxford-IIIT Pet, Flickr8k)
  - **Quiz 2**: Sentiment Analysis with RNN vs. LSTM (Accuracy / F1 / ROC-AUC, Accuracy by Review Length)
  - **Quiz 3**: Fine-tuning Vision Transformer (ViT-B/16) vs. ResNet-50 (Macro-AUC Comparison)
  - **Quiz 4**: Image Captioning (CNN-LSTM & CNN-Transformer, Gemini Judge, BLEU / BERTScore, Response-Time Comparison)

- **[HW05: Multimodal Deep Neural Networks (點此查閱完整作業報告)](HW05/README.md)**
  - **Quiz 1**: Image Retrieval on Unseen Species (CUB-200-2011, ResNet-50 vs. ViT-B/16, Recall@K & mAP)
  - **Quiz 2**: Image-Text Multimodal Classification (Flickr8k Match / Not-Match, Feature Fusion, 4 Backbone Combinations incl. CLIP)
  - **Quiz 3**: CLIP Image Captioning (CLIP Encoder + LSTM / Transformer Decoder, BLEU / BERTScore vs. Week 4)

- **[Midterm: RouteVision — 晶片版圖繞線壅塞熱點預測 (點此查閱完整專案報告)](Midterm/README.md)**
  - **任務定義**：鑑別式 AI 系統，輸入 placement 特徵圖與設計參數，預測 global routing 壅塞熱點。
  - **比較方法**：EDA 基準（RUDY, eGR）、OpenCV + Random Forest、MLP、CNN、Multimodal、ViT-B/16、Ensemble、U-Net（逐像素回歸）。
  - **核心成果**：跨設計驗證（LODO）達到 0.984 ROC-AUC、不確定度分析（分歧度 AUROC 0.911）、Grad-CAM 可解釋性與完整 Flask Web Demo。

- **[HW07: Multimodal Language Models (點此查閱完整作業報告)](HW07/README.md)**
  - **作業一**: Open-source LLM (Qwen3-4B) — 20-Question Instruction Benchmark, Prompting Techniques, FP16 / INT8 / NF4, GPU Memory
  - **作業 B**: Two Open-source VLMs (Qwen3-VL-2B vs. SmolVLM2-2.2B) on a 20-Question Visual Benchmark, Gemini Judge
  - **作業 C**: DPO Fine-tuning — LLM (Taiwan Traditional Chinese) and VLM (Object Hallucination, POPE / CHAIR, Image Degradations)

---

## 環境建置說明（Miniconda / Anaconda）

> [!NOTE]
> **開發環境說明**：
> 本課程作業開發與測試環境採用 **Miniconda3 (Python 3.10+)** 輕量化虛擬環境（相較於完整版 Anaconda 更加輕巧純淨，且指令與套件完全相容）。
> 每週作業各自獨立管理相依套件（見各資料夾內的 `requirements.txt`），助教可依下列步驟快速復現與執行：

```bash
# 1. 建立並啟用專用虛擬環境
conda create -n mmip python=3.10 -y
conda activate mmip

# 2. 執行 HW01（電腦視覺）
cd HW01
pip install -r requirements.txt
python HW01_main.py
# 或啟動互動介面：jupyter notebook HW01_notebook.ipynb

# 3. 執行 HW02（機器學習／深度學習）
cd ../HW02
pip install -r requirements.txt
python HW02_main.py
# 或啟動互動介面：jupyter notebook HW02_notebook.ipynb

# 4. 執行 HW03（CNN；建議使用 GPU，Notebook 已在 Colab T4 執行完成並內嵌所有輸出）
cd ../HW03
pip install -r requirements.txt
python HW03_main.py
# 或啟動互動介面：jupyter notebook HW03_notebook.ipynb

# 5. 執行 HW04（RNN / LSTM、ViT、Image Captioning；建議使用 GPU，Notebook 已在 Colab T4 執行完成並內嵌所有輸出）
#    Quiz 4 的 Gemini 部分需要先設定環境變數 GEMINI_API_KEY
cd ../HW04
pip install -r requirements.txt
python HW04_main.py
# 或啟動互動介面：jupyter notebook HW04_notebook.ipynb

# 6. 執行 HW05（影像檢索、圖文多模態分類、CLIP Image Captioning；建議使用 GPU，Notebook 已在 Colab T4 執行完成並內嵌所有輸出）
cd ../HW05
pip install -r requirements.txt
python HW05_main.py
# 或啟動互動介面：jupyter notebook HW05_notebook.ipynb

# 7. 執行 Midterm（RouteVision Web Demo；本機 CPU 即可快速啟動）
cd ../Midterm
pip install -r requirements.txt
python app/server.py
# 瀏覽器開啟 http://127.0.0.1:5000 即可體驗四步驟互動分析

# 8. 執行 HW07（LLM / VLM 評估與 DPO 微調；需要 GPU：Colab T4 或 RTX 2080 Ti）
#    Gemini 評審需要環境變數 GEMINI_API_KEY（沒有則只跳過評審）
cd ../HW07
pip install -r requirements.txt
python HW07_main.py
# 或啟動互動介面：jupyter notebook HW07_notebook.ipynb
```
