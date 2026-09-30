import os
import sys
import joblib
import rasterio
import numpy as np
import pandas as pd
import shap
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/outputs/maps/<path:filename>')
def serve_map_images(filename):
    return send_from_directory(os.path.join("outputs", "maps"), filename)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

# Paths
FACTORS_PATH = os.path.join("data", "Kuttanad_Environmental_Factors_30m.tif")
SUSC_PATH = os.path.join("outputs", "maps", "Kuttanad_Flood_Susceptibility_Map.tif")
RISK_PATH = os.path.join("outputs", "maps", "Kuttanad_Flood_Risk_Map.tif")
XGB_PATH = os.path.join("outputs", "models", "xgboost_model.pkl")
RF_PATH = os.path.join("outputs", "models", "random_forest_model.pkl")

# Global Variables for Models & Explainer
model = None
explainer = None
feature_names = ['elevation', 'slope', 'ndvi', 'lulc', 'rainfall', 'dist_water']
default_medians = [0.59, 1.30, 0.23, 66.5, 11.03, 702.0]

# Kuttanad Key Villages / Places Database (Lat, Lon, Name, Description)
KUTTANAD_VILLAGES = [
    {"id": "kainakary", "name": "Kainakary", "lat": 9.4920, "lon": 76.3680, "desc": "Paddy polder surrounded by Vembanad Lake"},
    {"id": "pulinkunnoo", "name": "Pulinkunnoo", "lat": 9.4550, "lon": 76.4520, "desc": "Central Kuttanad lowlands along Pamba river"},
    {"id": "champakulam", "name": "Champakulam", "lat": 9.4120, "lon": 76.4250, "desc": "Low-lying agricultural polder zone"},
    {"id": "edathua", "name": "Edathua", "lat": 9.3640, "lon": 76.4670, "desc": "Southern Kuttanad delta region"},
    {"id": "ramankary", "name": "Ramankary", "lat": 9.4280, "lon": 76.4800, "desc": "Manimala-Pamba river intersection plain"},
    {"id": "nedumudi", "name": "Nedumudi", "lat": 9.4350, "lon": 76.4020, "desc": "Backwater channel and river mouth zone"},
    {"id": "thalavady", "name": "Thalavady", "lat": 9.3510, "lon": 76.4420, "desc": "Agricultural interior floodplain"},
    {"id": "kavalam", "name": "Kavalam", "lat": 9.4950, "lon": 76.4480, "desc": "Northern Vembanad backwater village"},
    {"id": "neelamperoor", "name": "Neelamperoor", "lat": 9.5120, "lon": 76.4750, "desc": "Low elevation wetland agricultural boundary"},
    {"id": "veliyanad", "name": "Veliyanad", "lat": 9.4980, "lon": 76.5100, "desc": "North-eastern river corridor zone"}
]

def load_ml_components():
    global model, explainer
    if os.path.exists(XGB_PATH):
        print(f"Loading XGBoost model from {XGB_PATH}...")
        model = joblib.load(XGB_PATH)
    elif os.path.exists(RF_PATH):
        print(f"Loading Random Forest model from {RF_PATH}...")
        model = joblib.load(RF_PATH)
    else:
        print("Warning: No pre-trained model found!")

    if model is not None:
        try:
            explainer = shap.TreeExplainer(model)
            print("SHAP TreeExplainer initialized successfully!")
        except Exception as e:
            print(f"SHAP Explainer warning: {e}")

@app.route('/')
def index():
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/api/stats')
def get_stats():
    return jsonify({
        "study_area": "Kuttanad Region, Kerala",
        "resolution": "30m x 30m Grid Cells",
        "total_samples": 836786,
        "best_model": "Random Forest / XGBoost Ensemble",
        "auc_roc": 0.9289,
        "accuracy": 82.82,
        "f1_score": 0.7538,
        "recall": 87.87,
        "kappa": 0.6259
    })

def extract_valid_pixel_features(src, lon, lat):
    row, col = src.index(lon, lat)
    row = min(max(0, row), src.height - 1)
    col = min(max(0, col), src.width - 1)
    
    vals = src.read()[:, row, col].astype(np.float64)
    
    # If cell is NaN or NoData, search 3x3 window for nearest valid pixel
    if np.isnan(vals).any() or vals[0] < -50:
        window_data = src.read(window=((max(0, row-2), min(src.height, row+3)), 
                                       (max(0, col-2), min(src.width, col+3))))
        valid_mask = ~np.isnan(window_data[0]) & (window_data[0] >= -10)
        if valid_mask.any():
            r_idx, c_idx = np.where(valid_mask)
            vals = window_data[:, r_idx[0], c_idx[0]].astype(np.float64)
        else:
            # Fallback to feature medians
            vals = np.array(default_medians, dtype=np.float64)

    # Clean individual NaNs if any
    for i in range(len(vals)):
        if np.isnan(vals[i]):
            vals[i] = default_medians[i]

    return vals

@app.route('/api/villages')
def get_villages():
    results = []
    if not os.path.exists(FACTORS_PATH) or model is None:
        return jsonify(KUTTANAD_VILLAGES)

    with rasterio.open(FACTORS_PATH) as src:
        for v in KUTTANAD_VILLAGES:
            vals = extract_valid_pixel_features(src, v["lon"], v["lat"])
            input_df = pd.DataFrame([vals], columns=feature_names)
            prob = float(model.predict_proba(input_df)[0, 1])

            if prob < 0.25:
                class_label = "Low"
                risk_label = "Low Risk"
            elif prob < 0.50:
                class_label = "Moderate"
                risk_label = "Moderate Risk"
            elif prob < 0.75:
                class_label = "High"
                risk_label = "High Risk"
            else:
                class_label = "Very High"
                risk_label = "Critical Risk"

            v_copy = v.copy()
            v_copy["probability"] = round(prob * 100, 1)
            v_copy["class_label"] = class_label
            v_copy["risk_label"] = risk_label
            results.append(v_copy)

    return jsonify(results)

@app.route('/api/explain')
def explain_point():
    lat = float(request.args.get('lat', 9.45))
    lon = float(request.args.get('lon', 76.43))

    if not os.path.exists(FACTORS_PATH) or model is None or explainer is None:
        return jsonify({"error": "Model or environmental raster not loaded"})

    with rasterio.open(FACTORS_PATH) as src:
        vals = extract_valid_pixel_features(src, lon, lat)

    input_df = pd.DataFrame([vals], columns=feature_names)
    prob = float(model.predict_proba(input_df)[0, 1])

    # SHAP feature contributions
    shap_vals = explainer(input_df)
    
    if hasattr(shap_vals, 'values') and len(shap_vals.values.shape) == 3:
        contributions = shap_vals.values[0, :, 1]
    elif hasattr(shap_vals, 'values'):
        contributions = shap_vals.values[0]
    else:
        contributions = shap_vals[0]

    feature_impacts = {}
    for name, val, contrib in zip(feature_names, vals, contributions):
        feature_impacts[name] = {
            "value": round(float(val), 2),
            "shap_contribution": round(float(contrib), 4)
        }

    if prob < 0.25:
        category = "Low Susceptibility (<25%)"
        risk_level = "Low Risk"
    elif prob < 0.50:
        category = "Moderate Susceptibility (25%-50%)"
        risk_level = "Moderate Risk"
    elif prob < 0.75:
        category = "High Susceptibility (50%-75%)"
        risk_level = "High Risk"
    else:
        category = "Very High Susceptibility (>75%)"
        risk_level = "Critical Risk"

    return jsonify({
        "lat": lat,
        "lon": lon,
        "probability_percent": round(prob * 100, 1),
        "susceptibility_category": category,
        "risk_level": risk_level,
        "factors": feature_impacts
    })

if __name__ == "__main__":
    load_ml_components()
    print("=" * 60)
    print("🚀 KUTTANAD FLOOD SUSCEPTIBILITY & RISK LOCALHOST APP RUNNING!")
    print("👉 Open your browser and go to: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
