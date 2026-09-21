"""
Data Downloader & Preprocessor for MMIP HW02
Downloads the official UCI / Kaggle Default of Credit Card Clients Dataset,
cleans the column headers, and saves as standard CSV.
Also prepares the Quiz 1 Breast Cancer Wisconsin dataset (30 features, binary target).
"""

import os
import sys
import urllib.request
import zipfile
import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(DATA_DIR, exist_ok=True)


def download_credit_card_dataset():
    """
    Download the official UCI Default of Credit Card Clients dataset (xls inside zip),
    convert to clean CSV with standardized column names.
    Dataset: 30,000 observations, 24 attributes (23 features + 1 target).
    Target: 'default payment next month' (0: no default, 1: default).
    """
    csv_path = os.path.join(DATA_DIR, "default_of_credit_card_clients.csv")
    if os.path.exists(csv_path):
        print(f"[Credit Card Dataset] Found existing: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Shape: {df.shape}, Default rate: {df['default_payment_next_month'].mean():.2%}")
        return csv_path

    url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    zip_path = os.path.join(DATA_DIR, "credit_card.zip")

    print(f"[Credit Card Dataset] Downloading from UCI: {url}")
    urllib.request.urlretrieve(url, zip_path)
    print(f"  Downloaded zip size: {os.path.getsize(zip_path)} bytes")

    with zipfile.ZipFile(zip_path, 'r') as z:
        xls_filename = [name for name in z.namelist() if name.endswith(('.xls', '.xlsx'))][0]
        z.extract(xls_filename, DATA_DIR)
        raw_xls_path = os.path.join(DATA_DIR, xls_filename)

    # Read the Excel file (Row 1 contains feature names, row 0 is UCI metadata header)
    print(f"  Parsing raw Excel file: {raw_xls_path}")
    df_raw = pd.read_excel(raw_xls_path, header=1)

    # Standardize column names (lowercase, underscores, rename target)
    rename_dict = {
        'ID': 'id',
        'LIMIT_BAL': 'limit_bal',
        'SEX': 'sex',
        'EDUCATION': 'education',
        'MARRIAGE': 'marriage',
        'AGE': 'age',
        'PAY_0': 'pay_1',  # UCI PAY_0 is actually month 1 (September)
        'PAY_2': 'pay_2',
        'PAY_3': 'pay_3',
        'PAY_4': 'pay_4',
        'PAY_5': 'pay_5',
        'PAY_6': 'pay_6',
        'BILL_AMT1': 'bill_amt1',
        'BILL_AMT2': 'bill_amt2',
        'BILL_AMT3': 'bill_amt3',
        'BILL_AMT4': 'bill_amt4',
        'BILL_AMT5': 'bill_amt5',
        'BILL_AMT6': 'bill_amt6',
        'PAY_AMT1': 'pay_amt1',
        'PAY_AMT2': 'pay_amt2',
        'PAY_AMT3': 'pay_amt3',
        'PAY_AMT4': 'pay_amt4',
        'PAY_AMT5': 'pay_amt5',
        'PAY_AMT6': 'pay_amt6',
        'default payment next month': 'default_payment_next_month'
    }
    df = df_raw.rename(columns=rename_dict)

    # Drop the ID column as it has no predictive value
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Save clean CSV
    df.to_csv(csv_path, index=False)
    print(f"[Credit Card Dataset] Successfully created: {csv_path}")
    print(f"  Total samples: {len(df)}, Total features: {df.shape[1] - 1}")
    print(f"  Default rate: {df['default_payment_next_month'].mean():.2%}")

    # Clean up temporary raw files
    if os.path.exists(zip_path):
        os.remove(zip_path)
    if os.path.exists(raw_xls_path):
        os.remove(raw_xls_path)

    return csv_path


def prepare_quiz1_dataset():
    """
    Prepare the Quiz 1 binary classification dataset.
    Uses Breast Cancer Wisconsin (Diagnostic) Dataset from Scikit-Learn:
    - 569 instances
    - 30 continuous clinical features (mean, standard error, worst value of radius, texture, perimeter, etc.)
    - Binary target: 1 = Malignant (陽性/惡性), 0 = Benign (陰性/良性)
    - Fully satisfies the requirement: >= 5 features, real medical binary classification.
    """
    csv_path = os.path.join(DATA_DIR, "quiz1_breast_cancer.csv")
    if os.path.exists(csv_path):
        print(f"[Quiz 1 Dataset] Found existing: {csv_path}")
        return csv_path

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=[c.replace(' ', '_') for c in data.feature_names])
    # Target in sklearn: 0 = malignant, 1 = benign. We invert so 1 = Malignant (positive class), 0 = Benign
    df['target'] = (data.target == 0).astype(int)

    df.to_csv(csv_path, index=False)
    print(f"[Quiz 1 Dataset] Created: {csv_path}")
    print(f"  Shape: {df.shape}, Positive rate (Malignant): {df['target'].mean():.2%}")
    return csv_path


if __name__ == "__main__":
    print("=== MMIP HW02 Data Preparation ===")
    download_credit_card_dataset()
    prepare_quiz1_dataset()
    print("=== All Datasets Ready! ===")
