import os
import sys
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def generate_flood_risk_map():
    print("=" * 60)
    print("🛡️ BONUS FEATURE 2: GENERATING FLOOD RISK MAP (HAZARD × EXPOSURE)")
    print("=" * 60)

    susc_path = os.path.join("outputs", "maps", "Kuttanad_Flood_Susceptibility_Map.tif")
    factors_path = os.path.join("data", "Kuttanad_Environmental_Factors_30m.tif")

    if not os.path.exists(susc_path) or not os.path.exists(factors_path):
        raise FileNotFoundError("Missing susceptibility or factor raster!")

    # 1. Read Flood Susceptibility Map (Hazard)
    print(f"\n1. Reading Flood Susceptibility Raster: {susc_path}")
    with rasterio.open(susc_path) as src_susc:
        susc_meta = src_susc.meta.copy()
        susc_grid = src_susc.read(1)

    # 2. Read Land Use / Land Cover (Exposure Layer)
    print(f"\n2. Extracting Exposure Layer (LULC) from: {factors_path}")
    with rasterio.open(factors_path) as src_factors:
        factors_data = src_factors.read()
        lulc_grid = factors_data[3] # Band 4: LULC

    # Crop/align dimensions if needed
    min_rows = min(susc_grid.shape[0], lulc_grid.shape[0])
    min_cols = min(susc_grid.shape[1], lulc_grid.shape[1])

    susc_grid = susc_grid[:min_rows, :min_cols]
    lulc_grid = lulc_grid[:min_rows, :min_cols]

    # 3. Compute Exposure Scores (1=Low Exposure, 2=Moderate, 3=High Exposure for Paddy/Settlement)
    print("\n3. Calculating Land Cover Exposure Index...")
    exposure_grid = np.ones_like(lulc_grid, dtype=np.uint8)
    # LULC Codes: 10=Tree Cover, 40=Cropland/Paddy, 50=Built-up, 80=Water
    exposure_grid[lulc_grid == 40] = 3  # High exposure (Paddy fields / Cropland)
    exposure_grid[lulc_grid == 50] = 3  # High exposure (Human settlement / Built-up)
    exposure_grid[lulc_grid == 10] = 2  # Moderate exposure (Tree cover)

    # 4. Compute Flood Risk = Susceptibility Index x Exposure Index
    print("\n4. Multiplying Flood Hazard x Exposure to derive Final Risk Index...")
    risk_raw = susc_grid.astype(np.float32) * exposure_grid.astype(np.float32)

    risk_grid = np.zeros_like(susc_grid, dtype=np.uint8)
    risk_grid[susc_grid == 0] = 0 # NoData
    risk_grid[(risk_raw > 0) & (risk_raw <= 3)] = 1   # Low Risk
    risk_grid[(risk_raw > 3) & (risk_raw <= 6)] = 2   # Moderate Risk
    risk_grid[(risk_raw > 6) & (risk_raw <= 9)] = 3   # High Risk
    risk_grid[risk_raw > 9] = 4                        # Very High Risk (Critical Danger Zone)

    # 5. Export GeoTIFF Raster
    out_tif = os.path.join("outputs", "maps", "Kuttanad_Flood_Risk_Map.tif")
    susc_meta.update(dtype='uint8', count=1, nodata=0)
    with rasterio.open(out_tif, 'w', **susc_meta) as dst:
        dst.write(risk_grid, 1)

    print(f"✅ Flood Risk GeoTIFF exported to: {out_tif}")

    # 6. Render transparent PNG map overlay
    print("\n5. Rendering transparent Risk Map image...")
    # 0=Transparent, 1=Low (Yellow), 2=Moderate, 3=High (Orange), 4=Critical (Dark Red)
    cmap = ListedColormap(['#00000000', '#ffffb277', '#fecc5ccc', '#fd8d3cee', '#bd0026ff'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_alpha(0.0) # Transparent background
    ax.patch.set_alpha(0.0)

    im = ax.imshow(risk_grid, cmap=cmap, norm=norm)
    ax.axis('off')

    out_png = os.path.join("outputs", "maps", "Kuttanad_Flood_Risk_Map.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    print(f"✅ Map image successfully saved to: {out_png}")

if __name__ == "__main__":
    generate_flood_risk_map()
