import os
import sys
import joblib
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import matplotlib.pyplot as plt
import shap

def run_shap_analysis():
    print("=" * 60)
    print("🔍 PHASE 7: EXPLAINABLE AI (SHAP ANALYSIS)")
    print("=" * 60)

    dataset_path = os.path.join("data", "kuttanad_ml_dataset.csv")
    xgb_path = os.path.join("outputs", "models", "xgboost_model.pkl")

    if not os.path.exists(dataset_path) or not os.path.exists(xgb_path):
        raise FileNotFoundError("Dataset or trained XGBoost model missing! Run 01 and 02 first.")

    # 1. Load Data & Model
    print(f"\n1. Loading dataset & XGBoost model...")
    df = pd.read_csv(dataset_path)
    feature_cols = ['elevation', 'slope', 'ndvi', 'lulc', 'rainfall', 'dist_water']
    X = df[feature_cols]

    xgb_model = joblib.load(xgb_path)

    # Sample for fast SHAP computation (1,000 background samples)
    sample_size = min(1000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)

    # 2. Compute SHAP Values
    print(f"\n2. Computing SHAP values using TreeExplainer ({sample_size} grid samples)...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_sample)

    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)

    # 3. Save Summary Beeswarm Plot
    print("\n3. Generating SHAP Summary Beeswarm Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Feature Importance & Impact on Kuttanad Flood Susceptibility", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    summary_plot_path = os.path.join("outputs", "figures", "shap_summary_plot.png")
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()

    # 4. Save Feature Importance Bar Chart
    print("4. Generating SHAP Bar Chart...")
    plt.figure(figsize=(9, 5))
    shap.plots.bar(shap_values, show=False)
    plt.title("Global Feature Ranking (Mean |SHAP Value|)", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    bar_plot_path = os.path.join("outputs", "figures", "shap_bar_plot.png")
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()

    # 5. Print Feature Importance Ranking
    vals = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame(list(zip(X.columns, vals)), columns=['Feature', 'Mean |SHAP|'])
    importance_df = importance_df.sort_values(by=['Mean |SHAP|'], ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("📊 FEATURE IMPORTANCE RANKING (SHAP VALUES)")
    print("=" * 60)
    for idx, row in importance_df.iterrows():
        print(f"  {idx+1}. {row['Feature']:<12} : {row['Mean |SHAP|']:.4f}")

    print(f"\n✅ SHAP analysis plots saved to:")
    print(f"   - {summary_plot_path}")
    print(f"   - {bar_plot_path}")

if __name__ == "__main__":
    run_shap_analysis()
