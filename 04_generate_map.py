import os
import sys
import joblib
import rasterio
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

def generate_susceptibility_map():
    print("=" * 60)
    print("🗺️ PHASE 8 & 9: GENERATING FINAL GIS FLOOD SUSCEPTIBILITY MAP")
    print("=" * 60)

    factors_path = os.path.join("data", "Kuttanad_Environmental_Factors_30m.tif")
    xgb_path = os.path.join("outputs", "models", "xgboost_model.pkl")
    rf_path = os.path.join("outputs", "models", "random_forest_model.pkl")

    if not os.path.exists(factors_path):
        raise FileNotFoundError(f"Missing environmental factors raster: {factors_path}")

    # Use XGBoost if available, else Random Forest
    if os.path.exists(xgb_path):
        model_path = xgb_path
        model_name = "XGBoost"
    elif os.path.exists(rf_path):
        model_path = rf_path
        model_name = "Random Forest"
    else:
        raise FileNotFoundError("No trained model found! Please run 02_train_models.py first.")

    print(f"\n1. Loading trained {model_name} model from: {model_path}")
    model = joblib.load(model_path)

    # 2. Read Environmental Factors Raster
    print(f"\n2. Reading environmental raster stack: {factors_path}")
    with rasterio.open(factors_path) as src:
        meta = src.meta.copy()
        height = src.height
        width = src.width
        factors_data = src.read() # Shape: (6, height, width)
        feature_names = ['elevation', 'slope', 'ndvi', 'lulc', 'rainfall', 'dist_water']

    # 3. Predict Susceptibility Probabilities for Grid Cells
    print("\n3. Predicting flood susceptibility probabilities across Kuttanad grid...")
    flat_data = {name: factors_data[i].flatten() for i, name in enumerate(feature_names)}
    df = pd.DataFrame(flat_data)

    # Mask invalid / NoData cells
    valid_mask = ~df.isin([np.inf, -np.inf, np.nan]).any(axis=1) & (df['elevation'] >= -10) & (df['elevation'] <= 200)

    prob_flat = np.full(len(df), np.nan, dtype=np.float32)

    if valid_mask.sum() > 0:
        X_valid = df[valid_mask]
        print(f"   - Valid grid cells to evaluate: {len(X_valid):,}")
        probs = model.predict_proba(X_valid)[:, 1] # Probability of flood class (1)
        prob_flat[valid_mask] = probs

    prob_grid = prob_flat.reshape(height, width)

    # 4. Classify Probabilities into 4 Susceptibility Levels
    # 1: Low (< 0.25), 2: Moderate (0.25 - 0.50), 3: High (0.50 - 0.75), 4: Very High (> 0.75)
    print("\n4. Classifying into 4 NDMA/NRSC Hazard Levels (Low -> Very High)...")
    class_grid = np.zeros((height, width), dtype=np.uint8)
    class_grid[prob_grid < 0.25] = 1
    class_grid[(prob_grid >= 0.25) & (prob_grid < 0.50)] = 2
    class_grid[(prob_grid >= 0.50) & (prob_grid < 0.75)] = 3
    class_grid[prob_grid >= 0.75] = 4
    class_grid[np.isnan(prob_grid)] = 0 # NoData

    # 5. Export Susceptibility Raster GeoTIFF
    os.makedirs(os.path.join("outputs", "maps"), exist_ok=True)
    out_tif = os.path.join("outputs", "maps", "Kuttanad_Flood_Susceptibility_Map.tif")

    meta.update(dtype='uint8', count=1, nodata=0)
    with rasterio.open(out_tif, 'w', **meta) as dst:
        dst.write(class_grid, 1)

    print(f"✅ Classified GeoTIFF exported to: {out_tif}")

    # 5. Render transparent PNG map overlay
    print("\n5. Rendering transparent publication-quality susceptibility map...")
    # 0=Transparent, 1=Low (Blue 40% alpha), 2=Moderate (Green), 3=High (Orange), 4=Very High (Red)
    cmap = ListedColormap(['#00000000', '#2b83ba77', '#abdda4cc', '#fdae61ee', '#d7191cff'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_alpha(0.0) # Transparent background
    ax.patch.set_alpha(0.0)
    
    im = ax.imshow(class_grid, cmap=cmap, norm=norm)
    ax.axis('off')

    out_png = os.path.join("outputs", "maps", "Kuttanad_Flood_Susceptibility_Map.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    print(f"✅ Map image successfully saved to: {out_png}")
    print("\n" + "=" * 60)
    print("🎉 FULL KUTTANAD FLOOD SUSCEPTIBILITY PIPELINE READY!")
    print("=" * 60)

if __name__ == "__main__":
    generate_susceptibility_map()
