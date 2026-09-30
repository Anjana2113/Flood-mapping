import os
import sys
import joblib
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, 
    recall_score, cohen_kappa_score, classification_report, 
    confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE

def train_and_evaluate():
    print("=" * 60)
    print("🤖 PHASE 5 & 6: MACHINE LEARNING MODEL TRAINING & COMPARISON")
    print("=" * 60)

    dataset_path = os.path.join("data", "kuttanad_ml_dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset CSV not found! Please run 01_build_dataset.py first.")

    # 1. Load Dataset
    print(f"\n1. Loading ML dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    feature_cols = ['elevation', 'slope', 'ndvi', 'lulc', 'rainfall', 'dist_water']
    X = df[feature_cols]
    y = df['flood_label'].astype(int)

    print(f"   - Features ({len(feature_cols)}): {feature_cols}")
    print(f"   - Total Grid Cell Samples: {len(X):,}")

    # 2. Stratified Train-Test Split (80% Train, 20% Test)
    print("\n2. Splitting data into 80% Train and 20% Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   - Training Samples: {len(X_train):,}")
    print(f"   - Testing Samples : {len(X_test):,}")

    # 3. Handle Class Imbalance using SMOTE
    print("\n3. Applying SMOTE (Synthetic Minority Over-sampling) on Training Set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"   - Resampled Training Samples: {len(X_train_res):,}")
    print(f"   - Flood Class Count after SMOTE: {sum(y_train_res==1):,}")
    print(f"   - Non-Flood Class Count after SMOTE: {sum(y_train_res==0):,}")

    # 4. Train Random Forest Classifier
    print("\n4. Training Random Forest Classifier (Bagging Ensemble)...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_res, y_train_res)

    # 5. Train XGBoost Classifier
    print("\n5. Training XGBoost Classifier (Gradient Boosting Ensemble)...")
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train_res, y_train_res)

    # 6. Model Evaluation Strategy
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE EVALUATION RESULTS (TEST SET)")
    print("=" * 60)

    models = {'Random Forest': rf_model, 'XGBoost': xgb_model}
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        results[name] = {
            'Accuracy': acc, 'AUC-ROC': auc, 'F1-Score': f1,
            'Precision': prec, 'Recall': rec, 'Kappa': kappa,
            'y_pred': y_pred, 'y_prob': y_prob
        }

        print(f"\n--- {name} Performance ---")
        print(f" - Accuracy   : {acc*100:.2f}%")
        print(f" - AUC-ROC    : {auc:.4f}")
        print(f" - F1-Score   : {f1:.4f}")
        print(f" - Precision  : {prec:.4f}")
        print(f" - Recall     : {rec:.4f}")
        print(f" - Cohen Kappa: {kappa:.4f}")

    # 7. Generate & Save Comparison Plots
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)

    # ROC Curves Plot
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['AUC-ROC']:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.title('ROC Curve Comparison - Kuttanad Flood Susceptibility', fontsize=12, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate (Recall)', fontsize=10)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "figures", "roc_curve_comparison.png"), dpi=300)
    plt.close()

    # Confusion Matrices Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Non-Flood', 'Flood'], yticklabels=['Non-Flood', 'Flood'])
        axes[i].set_title(f"{name} Confusion Matrix", fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "figures", "confusion_matrices.png"), dpi=300)
    plt.close()

    # 8. Save Trained Models
    rf_path = os.path.join("outputs", "models", "random_forest_model.pkl")
    xgb_path = os.path.join("outputs", "models", "xgboost_model.pkl")
    
    joblib.dump(rf_model, rf_path)
    joblib.dump(xgb_model, xgb_path)
    print(f"\n✅ Models saved to:")
    print(f"   - {rf_path}")
    print(f"   - {xgb_path}")
    print(f"✅ Comparison figures saved to outputs/figures/")

    # Determine Best Model
    best_model_name = 'XGBoost' if results['XGBoost']['AUC-ROC'] >= results['Random Forest']['AUC-ROC'] else 'Random Forest'
    print(f"\n🏆 Best Performing Model: {best_model_name} (AUC-ROC: {results[best_model_name]['AUC-ROC']:.4f})")
    
    return best_model_name

if __name__ == "__main__":
    train_and_evaluate()
