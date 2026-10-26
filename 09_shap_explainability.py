"""
=============================================================================
09_shap_explainability.py
Phase 1: Explainable AI (XAI) Engine — Global & Village-Level SHAP
Reference: Li et al. (Remote Sensing, 2025, Section 3.5)
=============================================================================
Computes:
1. Global SHAP Interpretability: Beeswarm summary plot & feature ranking
2. Local Village-Level Interpretability: Exact SHAP waterfall plots and
   natural-language risk factor drivers for all 10 Kuttanad villages.
3. Exports village_shap_explanations.json for real-time web dashboard serving.
=============================================================================
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

FEATURE_NAMES = ['elevation', 'slope', 'ndvi', 'lulc', 'rainfall', 'dist_water']
FEATURE_DISPLAY = {
    'elevation': 'Elevation (m ASL)',
    'slope': 'Slope Gradient (°)',
    'ndvi': 'Vegetation Index (NDVI)',
    'lulc': 'Land Use / Land Cover',
    'rainfall': 'Monsoon Rainfall (mm/day)',
    'dist_water': 'Distance to River/Lake (m)'
}

def generate_plain_english_summary(village_name, risk_pct, pos_drivers, neg_drivers):
    """Generates an intuitive, non-black-box explanation for disaster management officers."""
    if risk_pct >= 70:
        severity = "HIGH RISK"
    elif risk_pct >= 40:
        severity = "MODERATE RISK"
    else:
        severity = "LOW RISK"
        
    driver_texts = [f"{FEATURE_DISPLAY.get(k, k)} (+{v:.1f}%)" for k, v in pos_drivers[:2]]
    drivers_str = " and ".join(driver_texts) if driver_texts else "general regional baseline conditions"
    
    summary = (
        f"{village_name} is currently flagged as {severity} ({risk_pct:.1f}% flood probability). "
        f"The primary hydrological drivers elevating inundation danger are {drivers_str}."
    )
    if neg_drivers:
        mitigator = f"{FEATURE_DISPLAY.get(neg_drivers[0][0], neg_drivers[0][0])} (-{abs(neg_drivers[0][1]):.1f}%)"
        summary += f" Risk is partially mitigated by {mitigator}."
        
    return summary

def main():
    print("=" * 65)
    print("🔍 PHASE 1: EXPLAINABLE AI (SHAP) ENGINE — GLOBAL & LOCAL")
    print("   Reference: Li et al. (Remote Sensing, 2025, Section 3.5)")
    print("=" * 65)

    dataset_path = os.path.join("data", "kuttanad_ml_dataset.csv")
    graph_path = os.path.join("data", "kuttanad_graph.json")
    xgb_path = os.path.join("outputs", "models", "xgboost_model.pkl")

    if not os.path.exists(dataset_path) or not os.path.exists(xgb_path):
        raise FileNotFoundError("Missing dataset CSV or trained XGBoost model!")

    # 1. Load Model and Background Dataset
    print("\n1. Loading pre-trained Tree model & background samples...")
    xgb_model = joblib.load(xgb_path)
    df = pd.read_csv(dataset_path)
    X = df[FEATURE_NAMES]

    sample_size = min(1500, len(X))
    X_background = X.sample(n=sample_size, random_state=42)

    # 2. Compute Global TreeSHAP
    print(f"\n2. Computing Global SHAP values across {sample_size} background samples...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_background)

    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)

    # 3. Global Beeswarm Summary Plot
    print("\n3. Generating Global SHAP Beeswarm Summary Plot...")
    plt.figure(figsize=(9, 6), dpi=300)
    shap.summary_plot(shap_values, X_background, feature_names=[FEATURE_DISPLAY[f] for f in FEATURE_NAMES], show=False)
    plt.title("Global SHAP Feature Attribution — Kuttanad Inundation Factors\n(Li et al., Remote Sensing 2025)", 
              fontsize=11, fontweight='bold', pad=14)
    plt.tight_layout()
    beeswarm_path = os.path.join("outputs", "figures", "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, bbox_inches='tight')
    plt.close()
    print(f"   - Saved to: {beeswarm_path}")

    # 4. Global Feature Importance Bar Chart
    print("\n4. Generating Global Feature Importance Bar Chart...")
    plt.figure(figsize=(9, 5), dpi=300)
    shap.plots.bar(shap_values, show=False)
    plt.title("Mean Absolute SHAP Value (Global Impact on Model Output Magnitude)", fontsize=11, fontweight='bold', pad=14)
    plt.tight_layout()
    bar_path = os.path.join("outputs", "figures", "shap_global_bar.png")
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()
    print(f"   - Saved to: {bar_path}")

    # Global Ranking Table
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': [FEATURE_DISPLAY[f] for f in FEATURE_NAMES],
        'Mean |SHAP|': mean_abs_shap,
        'Relative Impact (%)': (mean_abs_shap / mean_abs_shap.sum()) * 100
    }).sort_values(by='Mean |SHAP|', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("📊 GLOBAL FEATURE IMPORTANCE SUMMARY")
    print("=" * 60)
    for idx, row in importance_df.iterrows():
        print(f"  {idx+1}. {row['Feature']:<28} | |SHAP|: {row['Mean |SHAP|']:.4f} ({row['Relative Impact (%)']:.1f}%)")

    # 5. Local Village-Level SHAP Interpretability
    print("\n5. Computing Local Village-Level SHAP Explanations...")
    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    village_explanations = {}

    for node in graph_data["nodes"]:
        v_id = node["id"]
        v_name = node["name"]
        
        # Build node feature DataFrame in exact column order
        feat_dict = {f: [node["features"][f]] for f in FEATURE_NAMES}
        v_df = pd.DataFrame(feat_dict)
        
        # Predict probability
        prob = float(xgb_model.predict_proba(v_df)[0, 1])
        risk_pct = round(prob * 100, 1)

        # Compute exact local SHAP values for this village
        v_shap = explainer(v_df)
        v_shap_vals = v_shap.values[0] # array of shape (6,)
        base_val = float(v_shap.base_values[0])

        # Separate positive (aggravating) and negative (mitigating) risk contributors
        pos_drivers = []
        neg_drivers = []
        feature_breakdown = {}

        for k, f in enumerate(FEATURE_NAMES):
            impact = float(v_shap_vals[k] * 100) # scale to percentage contribution
            val = float(node["features"][f])
            feature_breakdown[f] = {
                "display_name": FEATURE_DISPLAY[f],
                "measured_value": val,
                "shap_impact_pct": round(impact, 2)
            }
            if impact > 0:
                pos_drivers.append((f, impact))
            else:
                neg_drivers.append((f, impact))

        pos_drivers.sort(key=lambda x: x[1], reverse=True)
        neg_drivers.sort(key=lambda x: x[1]) # most negative first

        # Generate Plain-English Narrative
        narrative = generate_plain_english_summary(v_name, risk_pct, pos_drivers, neg_drivers)

        # Generate Individual Village Waterfall Plot
        plt.figure(figsize=(8, 4.5), dpi=250)
        shap.plots.waterfall(v_shap[0], show=False, max_display=6)
        plt.title(f"SHAP Waterfall Explanation: {v_name} (Risk: {risk_pct}%)", fontsize=11, fontweight='bold', pad=12)
        plt.tight_layout()
        waterfall_path = os.path.join("outputs", "figures", f"shap_waterfall_{v_id}.png")
        plt.savefig(waterfall_path, bbox_inches='tight')
        plt.close()

        village_explanations[v_id] = {
            "name": v_name,
            "lat": node["lat"],
            "lon": node["lon"],
            "flood_risk_pct": risk_pct,
            "base_rate_pct": round(base_val * 100, 1),
            "plain_english_summary": narrative,
            "waterfall_plot_path": f"/outputs/figures/shap_waterfall_{v_id}.png",
            "top_aggravating_drivers": [{"feature": FEATURE_DISPLAY[f], "impact": round(imp, 1)} for f, imp in pos_drivers[:3]],
            "top_mitigating_factors": [{"feature": FEATURE_DISPLAY[f], "impact": round(imp, 1)} for f, imp in neg_drivers[:2]],
            "features_breakdown": feature_breakdown
        }

        print(f"   ✓ {v_name:<13} | Risk: {risk_pct:>5.1f}% | Waterfall plot saved.")

    # 6. Save JSON Payload for Web Dashboard
    output_json = os.path.join("outputs", "models", "village_shap_explanations.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(village_explanations, f, indent=2)

    print(f"\n   - Full village explanations JSON saved to: {output_json}")
    print("\n" + "=" * 65)
    print("✅ Step 3 (Option A: SHAP Explainability Engine) Complete!")
    print("=" * 65)

if __name__ == "__main__":
    main()
