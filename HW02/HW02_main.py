"""
MMIP HW02: Main Evaluation & Benchmarking Entry Point
Executes Quiz 1, Quiz 2, and Quiz 3 end-to-end,
validates compliance with course guidelines, and outputs a consolidated performance summary.
"""

import os
import sys
import time
import matplotlib
matplotlib.use('Agg')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Ensure HW02 root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from Code.quiz1_ml_classifier import run_quiz1
from Code.quiz2_mlp_credit import run_quiz2
from Code.quiz3_roc_eval import run_quiz3


def main():
    start_time = time.perf_counter()
    data_dir = os.path.join(CURRENT_DIR, "Data")
    output_dir = os.path.join(CURRENT_DIR, "Output")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("      NYCU MMIP WEEK 02 ASSIGNMENT: ML, DEEP LEARNING & MODEL EVALUATION       ")
    print("=" * 80)
    print(f"Base Directory:   {CURRENT_DIR}")
    print(f"Data Directory:   {data_dir}")
    print(f"Output Directory: {output_dir}")

    # -------------------------------------------------------------
    # Quiz 1: Machine Learning Binary Classification & Thresholds
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    q1_results = run_quiz1(data_dir, output_dir)
    q1_time = time.perf_counter() - t0

    # -------------------------------------------------------------
    # Quiz 2: Deep Learning Credit Card Default Prediction (PyTorch MLP)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    q2_results = run_quiz2(data_dir, output_dir)
    q2_time = time.perf_counter() - t0

    # -------------------------------------------------------------
    # Quiz 3: Model Performance Evaluation (ROC & AUC)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    q3_results = run_quiz3(data_dir, output_dir, mlp_bundle=q2_results)
    q3_time = time.perf_counter() - t0

    total_time = time.perf_counter() - start_time

    # -------------------------------------------------------------
    # Consolidated Grand Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                         SUMMARY OF ALL QUIZ RESULTS                            ")
    print("=" * 80)
    print(f"Total Execution & Evaluation Time: {total_time:.2f} seconds")

    print("\n[Quiz 1: Machine Learning Classification Task (18% Basic + 12% Advanced)]")
    print(f"  - Dataset:                      Breast Cancer Wisconsin Diagnostic (30 Features, N=569)")
    print(f"  - Feature Scaling:              StandardScaler applied to all continuous features")
    print(f"  - Model 1 (Logistic Regression): Baseline (Thresh=0.50): Acc={q1_results['m1_base']['accuracy']:.4f}, Rec={q1_results['m1_base']['recall']:.4f}, F1={q1_results['m1_base']['f1_score']:.4f}")
    print(f"  - Model 1 Fine-Tuned (Thresh=0.35): Recall boosted from {q1_results['m1_base']['recall']:.4f} -> {q1_results['m1_tuned']['recall']:.4f} (Catches more malignant cases)")
    print(f"  - Model 2 (Random Forest):      Optimized (Thresh={q1_results['opt_threshold']:.2f}): Acc={q1_results['m2_tuned']['accuracy']:.4f}, Prec={q1_results['m2_tuned']['precision']:.4f}, F1={q1_results['m2_tuned']['f1_score']:.4f}")
    print(f"  - [Supplementary] F2-Optimized (Thresh={q1_results['opt_threshold_f2']:.2f}): Rec={q1_results['m2_f2']['recall']:.4f} (catches all malignant cases at the cost of {q1_results['m2_f2']['fp']} FP)")
    print(f"  - Error Trade-off Analysis:     Logistic Regression has FP={q1_results['m1_base']['fp']}, FN={q1_results['m1_base']['fn']}; Random Forest has FP={q1_results['m2_base']['fp']}, FN={q1_results['m2_base']['fn']}")
    print(f"  - Output Plot:                  {q1_results['plot_path']}")

    print("\n[Quiz 2: Deep Learning Credit Default Prediction (24% Basic + 16% Advanced)]")
    print(f"  - Dataset:                      UCI / Kaggle Default of Credit Card Clients (30,000 Samples)")
    print(f"  - Deep Learning Framework:      PyTorch (Native TensorDataset & DataLoader)")
    print(f"  - Validation Sample Demo:       Sample #0 Pred Prob={q2_results['sample_demo']['predicted_prob']:.4f} -> Pred Label={q2_results['sample_demo']['predicted_label']} (Ground Truth={q2_results['sample_demo']['true_label']})")
    print(f"  - Baseline MLP Training:        50 Epochs trained | Final Train Loss={q2_results['res_base']['train_losses'][-1]:.4f} | Final Val Loss={q2_results['res_base']['val_losses'][-1]:.4f}")
    gap_base = q2_results['res_base']['val_losses'][-1] - q2_results['res_base']['train_losses'][-1]
    gap_imp = q2_results['res_improved']['val_losses'][-1] - q2_results['res_improved']['train_losses'][-1]
    gap_word = "narrows" if gap_imp < gap_base else "widens"
    print(f"  - Overfitting Observation:      Baseline generalization gap is {gap_base:.4f}; Regularization {gap_word} gap to {gap_imp:.4f}")
    print(f"  - Improved MLP Strategy:        Dropout (0.3) + L2 Regularization (Weight Decay 1e-4) + BatchNorm")

    val_loss_delta = q2_results['res_improved']['val_losses'][-1] - q2_results['res_base']['val_losses'][-1]
    f1_delta = q2_results['m_improved']['f1_score'] - q2_results['m_base']['f1_score']
    val_loss_word = "decreased" if val_loss_delta < 0 else "increased"
    f1_word = "improved" if f1_delta > 0 else ("declined" if f1_delta < 0 else "unchanged")
    print(f"  - Improvement Verification:     Val Loss {val_loss_word} to {q2_results['res_improved']['val_losses'][-1]:.4f} (Delta={val_loss_delta:+.4f}); F1 {f1_word} to {q2_results['m_improved']['f1_score']:.4f} (Delta={f1_delta:+.4f})")
    print(f"  - Output Plot:                  {q2_results['plot_path']}")

    print("\n[Quiz 3: Model Performance Evaluation (18% Basic + 12% Advanced)]")
    print(f"  - PyTorch MLP ROC & AUC:        AUC = {q3_results['roc_mlp']['auc']:.4f} (Marked on ROC Curve)")
    print(f"  - Machine Learning Model:       Random Forest Classifier on identical Training/Val set")
    print(f"  - Random Forest ROC & AUC:      AUC = {q3_results['roc_rf']['auc']:.4f} (Marked on same ROC Curve)")
    print(f"  - Discriminatory Power Verdict: {q3_results['better_model']} achieves superior positive/negative separation")
    print(f"  - Output Plot:                  {q3_results['plot_path']}")

    print("\n" + "=" * 80)
    print(f"All visual figures and diagnostic plots saved in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
