"""
Quiz 1: Machine Learning Binary Classification Task
Basic (18%):
  - Binary classification dataset (30 continuous features, Breast Cancer).
  - Train/Validation split (80/20 stratified).
  - Feature Scaling via StandardScaler.
  - Model 1: Logistic Regression.
  - Probability output, threshold selection (0.5), Confusion Matrix, Acc/Prec/Recall/F1.
  - Threshold fine-tuning (0.35), recompute Confusion Matrix and Acc/Prec/Recall/F1.
Advanced (12%):
  - Model 2: Random Forest Classifier.
  - Threshold optimization.
  - Comparative evaluation table on identical Validation set.
  - In-depth error trade-off analysis (Type I False Positive vs Type II False Negative).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .utils import compute_classification_metrics, render_confusion_matrix_axis


def load_and_preprocess_quiz1_data(data_path, test_size=0.2, random_state=42):
    """
    Load Quiz 1 dataset, perform stratified train-validation split,
    and apply StandardScaler on feature matrix X.
    """
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != 'target']
    X = df[feature_cols].values
    y = df['target'].values

    # Stratified 80/20 train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature Scaling: Fit strictly on training data, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'feature_names': feature_cols,
        'scaler': scaler
    }


def find_optimal_f1_threshold(y_true, y_prob, thresholds=None):
    """
    Search for the threshold that maximizes F1-score on validation data.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)

    best_thresh = 0.5
    best_f1 = -1.0
    best_metrics = None

    for th in thresholds:
        m = compute_classification_metrics(y_true, y_prob, threshold=th)
        if m['f1_score'] > best_f1:
            best_f1 = m['f1_score']
            best_thresh = th
            best_metrics = m

    return best_thresh, best_metrics


def run_quiz1(data_dir, output_dir):
    """
    Execute Quiz 1 end-to-end pipeline and save visualization plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "quiz1_breast_cancer.csv")
    if not os.path.exists(csv_path):
        from sklearn.datasets import load_breast_cancer
        ds = load_breast_cancer(as_frame=True)
        df_bc = ds.frame
        df_bc.to_csv(csv_path, index=False)

    data = load_and_preprocess_quiz1_data(csv_path, test_size=0.2, random_state=42)
    X_train, X_val = data['X_train'], data['X_val']
    y_train, y_val = data['y_train'], data['y_val']

    print("=======================================================")
    print("      QUIZ 1: MACHINE LEARNING CLASSIFICATION TASK     ")
    print("=======================================================")
    print(f"Dataset: Breast Cancer Wisconsin Diagnostic ({len(X_train)+len(X_val)} samples)")
    print(f"Features: {len(data['feature_names'])} continuous dimensions (scaled with StandardScaler)")
    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
    print(f"Class distribution (Val): Positive (Malignant)={sum(y_val==1)}, Negative (Benign)={sum(y_val==0)}")

    # -------------------------------------------------------------
    # 1. Basic Task: Model 1 - Logistic Regression
    # -------------------------------------------------------------
    lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_model.fit(X_train, y_train)
    lr_val_probs = lr_model.predict_proba(X_val)[:, 1]

    # Baseline Threshold: 0.5
    m1_base = compute_classification_metrics(y_val, lr_val_probs, threshold=0.5)

    # Fine-tuned Threshold: 0.35 (Medical context: lower threshold to catch more malignant cases)
    m1_tuned = compute_classification_metrics(y_val, lr_val_probs, threshold=0.35)

    print("\n[Basic Task: Model 1 - Logistic Regression]")
    print(f"  Baseline (Threshold=0.50): Acc={m1_base['accuracy']:.4f}, Prec={m1_base['precision']:.4f}, Rec={m1_base['recall']:.4f}, F1={m1_base['f1_score']:.4f}")
    print(f"    Confusion Matrix: TN={m1_base['tn']}, FP={m1_base['fp']}, FN={m1_base['fn']}, TP={m1_base['tp']}")
    print(f"  Fine-tuned (Threshold=0.35): Acc={m1_tuned['accuracy']:.4f}, Prec={m1_tuned['precision']:.4f}, Rec={m1_tuned['recall']:.4f}, F1={m1_tuned['f1_score']:.4f}")
    print(f"    Confusion Matrix: TN={m1_tuned['tn']}, FP={m1_tuned['fp']}, FN={m1_tuned['fn']}, TP={m1_tuned['tp']}")

    # -------------------------------------------------------------
    # 2. Advanced Task: Model 2 - Random Forest Classifier
    # -------------------------------------------------------------
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_val_probs = rf_model.predict_proba(X_val)[:, 1]

    m2_base = compute_classification_metrics(y_val, rf_val_probs, threshold=0.5)
    opt_th, m2_tuned = find_optimal_f1_threshold(y_val, rf_val_probs)

    print("\n[Advanced Task: Model 2 - Random Forest Classifier]")
    print(f"  Baseline (Threshold=0.50): Acc={m2_base['accuracy']:.4f}, Prec={m2_base['precision']:.4f}, Rec={m2_base['recall']:.4f}, F1={m2_base['f1_score']:.4f}")
    print(f"    Confusion Matrix: TN={m2_base['tn']}, FP={m2_base['fp']}, FN={m2_base['fn']}, TP={m2_base['tp']}")
    print(f"  Optimized (Threshold={opt_th:.2f}): Acc={m2_tuned['accuracy']:.4f}, Prec={m2_tuned['precision']:.4f}, Rec={m2_tuned['recall']:.4f}, F1={m2_tuned['f1_score']:.4f}")
    print(f"    Confusion Matrix: TN={m2_tuned['tn']}, FP={m2_tuned['fp']}, FN={m2_tuned['fn']}, TP={m2_tuned['tp']}")

    # -------------------------------------------------------------
    # 3. Comparative Summary & Error Analysis
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"QUIZ 1 MODEL PERFORMANCE COMPARISON TABLE (Validation Set, N={len(y_val)})")
    print("=" * 80)
    header = f"{'Model & Condition':35s} | {'Thresh':>6s} | {'Accuracy':>8s} | {'Precision':>9s} | {'Recall':>8s} | {'F1-Score':>8s} | {'Errors (FP/FN)':>14s}"
    print(header)
    print("-" * len(header))
    rows = [
        ("M1: Logistic Regression (Default)", 0.50, m1_base),
        ("M1: Logistic Regression (Fine-tuned)", 0.35, m1_tuned),
        ("M2: Random Forest (Default)", 0.50, m2_base),
        ("M2: Random Forest (Optimized)", opt_th, m2_tuned),
    ]
    for name, th, m in rows:
        err_str = f"FP={m['fp']} / FN={m['fn']}"
        print(f"{name:35s} | {th:6.2f} | {m['accuracy']:8.4f} | {m['precision']:9.4f} | {m['recall']:8.4f} | {m['f1_score']:8.4f} | {err_str:>14s}")
    print("=" * 80)

    # -------------------------------------------------------------
    # 4. Visualization: 2x2 Clean Confusion Matrix Grid
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), facecolor='white')
    class_names = ['Benign (0)', 'Malignant (1)']

    render_confusion_matrix_axis(
        axes[0, 0], m1_base['confusion_matrix'],
        title="Model 1: Logistic Regression (Threshold = 0.50)",
        class_names=class_names,
        subtitle=f"Acc: {m1_base['accuracy']:.3f} | Prec: {m1_base['precision']:.3f} | Rec: {m1_base['recall']:.3f} | F1: {m1_base['f1_score']:.3f}"
    )

    render_confusion_matrix_axis(
        axes[0, 1], m1_tuned['confusion_matrix'],
        title="Model 1: Logistic Regression (Fine-tuned = 0.35)",
        class_names=class_names,
        subtitle=f"Acc: {m1_tuned['accuracy']:.3f} | Prec: {m1_tuned['precision']:.3f} | Rec: {m1_tuned['recall']:.3f} | F1: {m1_tuned['f1_score']:.3f}"
    )

    render_confusion_matrix_axis(
        axes[1, 0], m2_base['confusion_matrix'],
        title="Model 2: Random Forest (Threshold = 0.50)",
        class_names=class_names,
        subtitle=f"Acc: {m2_base['accuracy']:.3f} | Prec: {m2_base['precision']:.3f} | Rec: {m2_base['recall']:.3f} | F1: {m2_base['f1_score']:.3f}"
    )

    render_confusion_matrix_axis(
        axes[1, 1], m2_tuned['confusion_matrix'],
        title=f"Model 2: Random Forest (Optimized = {opt_th:.2f})",
        class_names=class_names,
        subtitle=f"Acc: {m2_tuned['accuracy']:.3f} | Prec: {m2_tuned['precision']:.3f} | Rec: {m2_tuned['recall']:.3f} | F1: {m2_tuned['f1_score']:.3f}"
    )

    plt.suptitle("Quiz 1: Confusion Matrices & Threshold Tuning Comparison", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "quiz1_confusion_matrices.png")
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved Output] Confusion matrix comparison plot saved to: {plot_path}")

    return {
        'm1_base': m1_base,
        'm1_tuned': m1_tuned,
        'm2_base': m2_base,
        'm2_tuned': m2_tuned,
        'opt_threshold': opt_th,
        'plot_path': plot_path
    }
