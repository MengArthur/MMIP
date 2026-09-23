"""
Quiz 3: Model Performance Evaluation (ROC Curve & AUC Analysis)
Basic (18%):
  - Continue using Quiz 2 Credit Card Default dataset and trained MLP model.
  - Obtain validation prediction probabilities.
  - Plot ROC Curve (X: FPR, Y: TPR) with annotated AUC score.
  - Explain statistical & domain significance of ROC & AUC.
Advanced (12%):
  - Train second model using Machine Learning algorithm from class (Random Forest Classifier).
  - Obtain validation prediction probabilities on identical validation split.
  - Plot both MLP and Random Forest ROC curves on the same plot.
  - Compute and annotate both AUC scores.
  - Comparative analysis of positive/negative sample discriminatory capacity.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from .utils import compute_roc_auc
from .quiz2_mlp_credit import load_and_preprocess_credit_data, ImprovedCreditMLP, train_mlp_model, set_seed


def run_quiz3(data_dir, output_dir, mlp_bundle=None):
    """
    Execute Quiz 3 end-to-end:
    1. Retrieve or train PyTorch MLP on Credit Card dataset
    2. Train Random Forest Classifier on identical training set
    3. Compute ROC & AUC for both models
    4. Render combined comparative ROC Curve plot
    5. Output detailed discriminative capacity analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "UCI_Credit_Card.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_dir, "default_of_credit_card_clients.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Credit Card dataset not found in {data_dir}. Expected 'UCI_Credit_Card.csv'.")

    set_seed(42)
    data = load_and_preprocess_credit_data(csv_path, test_size=0.2, random_state=42)
    X_train, X_val = data['X_train'], data['X_val']
    y_train, y_val = data['y_train'], data['y_val']

    print("=======================================================")
    print("      QUIZ 3: MODEL PERFORMANCE EVALUATION (ROC & AUC)  ")
    print("=======================================================")
    print(f"Validation Sample Size: {len(y_val)} (Positive Default={int(sum(y_val==1))}, Negative={int(sum(y_val==0))})")

    # -------------------------------------------------------------
    # 1. Basic Task: MLP Predictions & ROC/AUC
    # -------------------------------------------------------------
    if mlp_bundle is not None and 'res_improved' in mlp_bundle:
        mlp_probs = mlp_bundle['res_improved']['val_probs']
        print("[Basic Task] Reusing trained Improved MLP model from Quiz 2...")
    else:
        print("[Basic Task] Training Improved MLP model (50 Epochs)...")
        mlp_model = ImprovedCreditMLP(in_features=X_train.shape[1], dropout_rate=0.3)
        res_mlp = train_mlp_model(
            mlp_model, X_train, y_train, X_val, y_val,
            epochs=50, batch_size=128, lr=0.001, weight_decay=1e-4
        )
        mlp_probs = res_mlp['val_probs']

    roc_mlp = compute_roc_auc(y_val, mlp_probs)
    print(f"  PyTorch MLP AUC Score: {roc_mlp['auc']:.4f}")

    # -------------------------------------------------------------
    # 2. Advanced Task: Machine Learning Model (Random Forest)
    # -------------------------------------------------------------
    print("\n[Advanced Task: Training Machine Learning Model (Random Forest)]")
    rf_model = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_probs = rf_model.predict_proba(X_val)[:, 1]

    roc_rf = compute_roc_auc(y_val, rf_probs)
    print(f"  Random Forest Classifier AUC Score: {roc_rf['auc']:.4f}")

    # -------------------------------------------------------------
    # 3. Model Comparison Table
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("QUIZ 3 ROC & AUC COMPARATIVE SUMMARY (Validation Set, N=6,000)")
    print("=" * 75)
    header = f"{'Model':30s} | {'Algorithm Type':20s} | {'AUC Score':>10s} | {'Discriminatory Rank':>19s}"
    print(header)
    print("-" * len(header))
    better_model = "Random Forest" if roc_rf['auc'] >= roc_mlp['auc'] else "PyTorch MLP"
    print(f"{'PyTorch Improved MLP':30s} | {'Deep Neural Network':20s} | {roc_mlp['auc']:10.4f} | {'Baseline / Competitor':>19s}")
    print(f"{'Random Forest Classifier':30s} | {'Ensemble Decision Trees':20s} | {roc_rf['auc']:10.4f} | {'Superior AUC (Winner)' if better_model=='Random Forest' else 'Runner-up':>19s}")
    print("=" * 75)

    diff_auc = abs(roc_rf['auc'] - roc_mlp['auc'])
    print(f"\n[Comparative Insights]")
    print(f"1. AUC Comparison: Random Forest ({roc_rf['auc']:.4f}) vs MLP ({roc_mlp['auc']:.4f}) -> Delta = {diff_auc:.4f}")
    print(f"2. Superior Discriminatory Power: {better_model} achieves a higher AUC, demonstrating better separation between defaulting and non-defaulting cardholders across all classification thresholds.")
    print("3. Why Tree Ensemble Excels on Tabular Data: Random Forest naturally captures non-linear tabular feature interactions and step-function relationships without requiring smooth continuous manifold assumptions.")

    # -------------------------------------------------------------
    # 4. Visualization: Combined ROC Curve Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8), facecolor='white')

    # Random Guess Baseline
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, label='Random Chance Baseline (AUC = 0.5000)')

    # Model 1: PyTorch MLP
    ax.plot(
        roc_mlp['fpr'], roc_mlp['tpr'],
        color='#1f77b4', linewidth=2.5,
        label=f"PyTorch Improved MLP (AUC = {roc_mlp['auc']:.4f})"
    )

    # Model 2: Random Forest
    ax.plot(
        roc_rf['fpr'], roc_rf['tpr'],
        color='#2ca02c', linewidth=2.5,
        label=f"Random Forest Classifier (AUC = {roc_rf['auc']:.4f})"
    )

    ax.set_title("Quiz 3: ROC Curve & AUC Performance Comparison\n(PyTorch Deep MLP vs. Random Forest on Credit Card Default)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("False Positive Rate (FPR = 1 - Specificity)", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Positive Rate (TPR = Sensitivity / Recall)", fontsize=11, fontweight='bold')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=10.5, frameon=True, framealpha=0.95, edgecolor='#cccccc')

    # Annotate best operating region
    ax.scatter([0.15], [0.55], color='red', s=40, zorder=5)
    ax.annotate(
        "Practical Banking Operating Zone\n(Low False Alarms, High True Catch)",
        xy=(0.15, 0.55), xytext=(0.28, 0.42),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
        fontsize=9.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.3)
    )

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "quiz3_roc_curve_comparison.png")
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved Output] ROC curve comparison plot saved to: {plot_path}")

    return {
        'roc_mlp': roc_mlp,
        'roc_rf': roc_rf,
        'better_model': better_model,
        'plot_path': plot_path
    }
