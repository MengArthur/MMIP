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

---

## 環境建置說明（Miniconda / Anaconda）

> [!NOTE]
> **開發環境說明**：
> 本作業開發與測試環境採用 **Miniconda3 (Python 3.10+)** 輕量化虛擬環境（相較於完整版 Anaconda 更加輕巧純淨，且指令與套件完全相容）。
> 助教無論使用 **Miniconda** 或 **Anaconda**，皆可依下列步驟快速復現與執行作業：

`ash
# 1. 建立並啟用專用虛擬環境
conda create -n mmip python=3.10 -y
conda activate mmip

# 2. 安裝核心相依套件 (opencv-python, numpy, matplotlib, Pillow)
pip install -r requirements.txt

# 3. 執行 HW01 主程式評測
cd HW01
python HW01_main.py

# 4. 或啟動 Jupyter Notebook 互動介面
pip install notebook
jupyter notebook HW01_notebook.ipynb
`
