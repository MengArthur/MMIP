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
```
