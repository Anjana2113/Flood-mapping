# 🌊 Kuttanad Spatio-Temporal Flood AI & Early Warning Platform

> An operational, spatio-temporal flood forecasting, early warning, and explainable AI system for the Kuttanad delta region of Kerala, India.  
> Grounded in the **March 2025 SCIE Base Paper**: *Li et al., "Flash Flood Regionalization Combining GNN and SHAP Methods", Remote Sensing (MDPI)*.

---

## 📌 Phase 1 Milestones Achieved (50% Implementation)

1. **Hydrological Graph Formulation $G = (V, E, X)$:** Converted physical landscape, village polders, and river canals into a topological graph with spatial adjacency matrix $A$ (*Li et al., Section 3.1*).
2. **Deep Sequence Modeling via GRU:** Implemented a Gated Recurrent Unit (GRU) neural network modeling 5-step antecedent precipitation and hydraulic memory lag.
3. **Formal 3-Model Benchmarking:** Evaluated Random Forest vs. XGBoost vs. Proposed GRU on a held-out test split across 6 metrics.
4. **Explainable AI (SHAP) Engine:** Generated global beeswarm attribution and localized village-level SHAP waterfall plots with automated plain-English risk driver narratives.
5. **Interactive Public Early Warning Dashboard:** Deployed Flask application featuring village risk lookup, real-time SHAP explanation inspector, and emergency alert broadcast simulation.

---

## 📊 3-Model Performance Benchmark (Held-Out Test Data)

| Metric | Random Forest (Baseline) | XGBoost (Baseline) | Proposed GRU (Deep Learning) |
|---|:---:|:---:|:---:|
| **Accuracy** | 88.53% | 93.14% | **90.71%** |
| **AUC-ROC** | 0.9421 | 0.9216 | **0.8995** |
| **F1-Score** | 0.9319 | 0.9611 | **0.9465** |
| **Precision** | 0.9733 | 0.9565 | **0.9574** |
| **Recall** | 0.8938 | 0.9657 | **0.9358** |
| **Cohen's Kappa** | 0.5740 | 0.6714 | **0.5962** |

---

## 🗂️ Project Structure

```
final_project/
│
├── 01_build_dataset.py            # Extract features from GEE rasters into ML dataset
├── 02_train_models.py             # Train baseline RF & XGBoost with SMOTE balancing
├── 03_shap_analysis.py            # Preliminary SHAP analysis
├── 04_generate_map.py             # Generate Flood Susceptibility Map (GeoTIFF)
├── 05_interactive_web_map.py      # Standalone Folium map viewer
├── 06_flood_risk_map.py           # Generate Flood Risk Map (Hazard x Exposure)
├── 07_construct_kuttanad_graph.py # Construct Hydrological Graph G=(V,E,X) & Adjacency Matrix
├── 08_train_gru_model.py          # Train Deep GRU Sequence Network & generate 3-model benchmark
├── 09_shap_explainability.py      # Global SHAP beeswarm + local village waterfall explanations
│
├── app.py                         # Flask web application & REST API server
│
├── templates/
│   └── index.html                 # Leaflet GIS early warning dashboard UI
│
├── data/
│   ├── kuttanad_graph.json        # Graph structure: nodes, features, adjacency matrix
│   ├── Kuttanad_Environmental_Factors_30m.tif (Git LFS / Local)
│   ├── Kuttanad_Master_Flood_Inventory_2018_2021.tif (Git LFS / Local)
│   └── kuttanad_ml_dataset.csv (Git LFS / Local)
│
└── outputs/
    ├── maps/                      # Flood Susceptibility & Risk Maps (TIF / PNG)
    ├── models/
    │   ├── benchmark_summary.json # 3-model benchmark metrics JSON
    │   ├── village_shap_explanations.json # Village-level natural-language SHAP drivers
    │   ├── gru_model.h5           # Trained Deep GRU model weights
    │   ├── feature_scaler.pkl     # Standard scaler for input features
    │   ├── xgboost_model.pkl      # Trained XGBoost model
    │   └── random_forest_model.pkl # Trained Random Forest model (local)
    └── figures/
        ├── kuttanad_graph_topology.png       # Graph topology visualization
        ├── 3_model_roc_comparison.png        # Comparative ROC curves
        ├── 3_model_metrics_comparison.png    # Multi-metric comparison bar chart
        ├── shap_summary_beeswarm.png         # Global SHAP beeswarm plot
        ├── shap_global_bar.png               # Global feature ranking bar chart
        └── shap_waterfall_*.png              # Village-level local waterfall plots
```

---

## 🌐 Running the Platform Locally

1. Install dependencies:
   ```bash
   pip install flask rasterio numpy pandas scikit-learn xgboost shap imbalanced-learn matplotlib networkx tensorflow
   ```

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open in your browser:  
   👉 **http://127.0.0.1:5000**

---

## 📍 Study Basin

**Kuttanad, Alappuzha District, Kerala, India** — Lying 0.6–2.2m below sea level, it is South Asia's lowest-lying agrarian delta experiencing chronic monsoon flooding and backwater ingress.

---

## 👥 Authors & Academic Context

* **Project:** Final Year B.Tech Data Science / Computer Science Project (Phase 1)
* **Base Paper Reference:** Li, Y., Zhang, C., Cui, P., et al. (2025). *Flash Flood Regionalization for the Hengduan Mountains Region, China, Combining GNN and SHAP Methods*. **Remote Sensing**, 17(6), 946. [DOI: 10.3390/rs17060946](https://doi.org/10.3390/rs17060946)
