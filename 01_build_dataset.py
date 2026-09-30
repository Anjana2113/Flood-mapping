import os
import sys
import rasterio
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def build_kuttanad_dataset():
    print("=" * 60)
    print("🌊 PHASE 4: BUILDING KUTTANAD ML DATASET FROM RASTERS")
    print("=" * 60)

    factors_path = os.path.join("data", "Kuttanad_Environmental_Factors_30m.tif")
    label_path = os.path.join("data", "Kuttanad_Master_Flood_Inventory_2018_2021.tif")
    output_csv = os.path.join("data", "kuttanad_ml_dataset.csv")

    if not os.path.exists(factors_path) or not os.path.exists(label_path):
        raise FileNotFoundError("Raster files not found in data/ folder! Please check file names.")

    # 1. Read Environmental Conditioning Factors (6 Bands)
    print(f"\n1. Reading Environmental Factors from: {factors_path}")
    with rasterio.open(factors_path) as src_factors:
        factors_meta = src_factors.meta
        factors_data = src_factors.read() # Shape: (6, height, width)
        feature_names = ['elevation', 'slope', 'ndvi', 'lulc', 'rainfall', 'dist_water']
        print(f"   - Dimensions: {src_factors.height} x {src_factors.width} pixels")
        print(f"   - Bands found ({src_factors.count}): {feature_names}")

    # 2. Read Target Flood Inventory Label (1 Band)
    print(f"\n2. Reading Master Flood Inventory Label from: {label_path}")
    with rasterio.open(label_path) as src_label:
        label_data = src_label.read(1) # Shape: (height, width)
        print(f"   - Dimensions: {src_label.height} x {src_label.width} pixels")

    # Automatically align raster spatial dimensions to minimum overlap
    min_rows = min(factors_data.shape[1], label_data.shape[0])
    min_cols = min(factors_data.shape[2], label_data.shape[1])
    
    factors_data = factors_data[:, :min_rows, :min_cols]
    label_data = label_data[:min_rows, :min_cols]
    
    print(f"   - Aligned Spatial Grid: {min_rows} x {min_cols} pixels")

    # 3. Extract Grid Cells into Dataframe
    print("\n3. Processing raster grid cells into tabular dataset...")
    n_bands, height, width = factors_data.shape
    total_pixels = height * width

    # Flatten arrays
    flat_features = {name: factors_data[i].flatten() for i, name in enumerate(feature_names)}
    flat_features['flood_label'] = label_data.flatten()

    df = pd.DataFrame(flat_features)
    print(f"   - Total Grid Cells Extracted: {len(df):,}")

    # 4. Filter Invalid / NoData Cells
    print("\n4. Cleaning and filtering invalid cells...")
    # Remove NaN, Inf values
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    
    # Remove unrealistic artifact values (e.g. elevation < -10m or > 200m in Kuttanad)
    df_clean = df_clean[(df_clean['elevation'] >= -10) & (df_clean['elevation'] <= 200)].copy()

    # 5. Dataset Summary & Class Imbalance Analysis
    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY & CLASS DISTRIBUTION")
    print("=" * 60)
    print(f"Total Valid ML Samples (Grid Cells): {len(df_clean):,}")
    
    flood_count = int(df_clean['flood_label'].sum())
    non_flood_count = len(df_clean) - flood_count
    imbalance_ratio = non_flood_count / max(flood_count, 1)

    print(f" - Non-Flood Cells (Label 0): {non_flood_count:,} ({non_flood_count/len(df_clean)*100:.2f}%)")
    print(f" - Flooded Cells   (Label 1): {flood_count:,} ({flood_count/len(df_clean)*100:.2f}%)")
    print(f" - Class Imbalance Ratio     : 1:{imbalance_ratio:.2f}")

    print("\nFeature Summary Statistics:")
    print(df_clean.describe().round(2).T[['mean', 'std', 'min', '50%', 'max']])

    # 6. Export Cleaned Dataset to CSV
    df_clean.to_csv(output_csv, index=False)
    print(f"\n✅ Clean dataset successfully saved to: {output_csv}")
    return df_clean

if __name__ == "__main__":
    build_kuttanad_dataset()
