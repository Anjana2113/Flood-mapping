# 🌊 Flood Susceptibility & Risk Mapping — Kuttanad, Kerala

> A GIS and Machine Learning based flood susceptibility and risk assessment system for the Kuttanad region of Kerala, India.

---

## 📌 Project Overview

This project uses **Sentinel-1 SAR imagery**, **Google Earth Engine (GEE)**, and **Machine Learning** to:
1. Extract historical flood inundation maps (2018, 2019, 2021 Kerala floods)
2. Collect environmental factors (DEM, Slope, NDVI, LULC, Rainfall, Distance to Water)
3. Train flood susceptibility models using **Random Forest** and **XGBoost**
4. Generate a **Flood Susceptibility Map** and a **Flood Risk Map** (Hazard x Exposure)
5. Serve an interactive web dashboard for flood exploration

---

## 🗂️ Project Structure

```
final_project/
│
├── 01_build_dataset.py        # Extract features + build ML dataset
├── 02_train_models.py         # Train RF & XGBoost with SMOTE balancing
├── 03_shap_analysis.py        # SHAP feature importance analysis
├── 04_generate_map.py         # Generate Flood Susceptibility Map
├── 05_interactive_web_map.py  # Folium-based standalone map
├── 06_flood_risk_map.py       # Generate Flood Risk Map (Hazard x Exposure)
│
├── app.py                     # Flask web app (API + dashboard server)
│
├── templates/
│   └── index.html             # Interactive Leaflet dashboard UI
│
├── data/                      # (Not tracked — large GEE rasters & CSV)
│   ├── Kuttanad_Environmental_Factors_30m.tif
│   ├── Kuttanad_Master_Flood_Inventory_2018_2021.tif
│   └── kuttanad_ml_dataset.csv
│
└── outputs/
    ├── maps/                  # Flood Susceptibility & Risk PNGs + TIFs
    ├── models/                # Trained ML model PKL files (not tracked)
    └── figures/               # SHAP plots and analysis figures
```

---

## 🛰️ Data & Factor Collection (Google Earth Engine)

All spatial data was extracted via **Google Earth Engine (GEE)** at **30m resolution** to ensure spatial consistency:

| Factor | Source | Why GEE? |
|---|---|---|
| Flood Inventory | Sentinel-1 SAR (2018, 2019, 2021) | Multi-event SAR processing via GEE |
| DEM (Elevation) | SRTM 30m | Clipped to exact study boundary |
| Slope | Derived from SRTM | Computed on-the-fly in GEE |
| NDVI | Sentinel-2 (pre-flood) | Cloud-masked composites |
| LULC | ESA WorldCover 2020 | Resampled to match grid |
| Rainfall | CHIRPS | Seasonal accumulation |
| Distance to Water | JRC Surface Water | Euclidean distance raster |

---

## 🤖 Machine Learning Pipeline

```
Raw Rasters → Feature Stack → ML Dataset → SMOTE Balancing
                                                ↓
                              Random Forest + XGBoost Training
                                                ↓
                         SHAP Analysis → Feature Importance
                                                ↓
                    Flood Susceptibility Map (pixel-wise prediction)
                                                ↓
                    Flood Risk Map = Susceptibility x Exposure (LULC)
```

---

## 🌐 Interactive Web Dashboard

Run the Flask app locally:

```bash
pip install flask rasterio numpy scikit-learn xgboost shap
python app.py
```

Then open: **http://127.0.0.1:5000**

### Features:
- Interactive Leaflet map with satellite base layers (Google, Esri)
- Village-level flood susceptibility & risk lookup
- Live SHAP explanation chart per location
- Flood Risk overlay showing endangered human settlements
- Toggle between Flood Susceptibility and Risk layers

---

## 📦 Requirements

```bash
pip install flask rasterio numpy pandas scikit-learn xgboost shap imbalanced-learn matplotlib
```

---

## 📁 Note on Large Files

The following files are **not tracked** in this repository due to size constraints:
- `data/*.tif` — GEE rasters (30-300 MB)
- `data/*.csv` — ML dataset (~37 MB)
- `outputs/models/*.pkl` — Trained model files (~175 MB)

To reproduce: run scripts `01_build_dataset.py` → `02_train_models.py` → `04_generate_map.py` → `06_flood_risk_map.py` → `app.py`

---

## 📍 Study Area

**Kuttanad, Alappuzha District, Kerala, India** — Known as the "Rice Bowl of Kerala," it is one of the most flood-prone regions in South Asia, lying 0.6–2.2m below sea level.

---

## Author

Anjana | Final Year Project | 2025-2026
