from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

# -------------------------------
# LOAD MODELS
# -------------------------------
BRAND_MODEL_PATH = "milk_models/final_brand_pipeline_optimized.pkl"
WATER_MODEL_PATH = "milk_models/final_water_pipeline.pkl"
BRAND_ENCODER_PATH = "milk_models/brand_encoder.pkl"
SPECTRAL_COLS_PATH = "milk_models/spectral_cols.pkl"
BRAND_INPUT_COLS_PATH = "milk_models/final_brand_input_cols.pkl"
FEATURE_COLS_PATH = "milk_models/feature_cols.pkl"
METADATA_PATH = "milk_models/final_model_metadata.json"

for path in [
    BRAND_MODEL_PATH,
    WATER_MODEL_PATH,
    BRAND_ENCODER_PATH,
    SPECTRAL_COLS_PATH,
    BRAND_INPUT_COLS_PATH,
    FEATURE_COLS_PATH
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required model file: {path}")

brand_pipeline = joblib.load(BRAND_MODEL_PATH)
water_pipeline = joblib.load(WATER_MODEL_PATH)
brand_encoder = joblib.load(BRAND_ENCODER_PATH)
spectral_cols = joblib.load(SPECTRAL_COLS_PATH)
brand_input_cols = joblib.load(BRAND_INPUT_COLS_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)

metadata = {}
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def get_status_from_water(w):
    if w <= 5:
        return "PURE"
    elif w <= 15:
        return "LOW_ADULTERATION"
    elif w <= 30:
        return "MODERATE_ADULTERATION"
    else:
        return "HIGH_ADULTERATION"

def get_quality_score(w):
    score = 100 - (1.8 * w)
    return max(0, min(100, score))

def add_engineered_features(data):
    x = data.copy()
    eps = 1e-8

    visible_cols = ["A410", "A435", "A460", "A485", "A510", "A535", "B560", "B585", "B610"]
    nir_cols = ["B645", "B680", "B705", "C730", "C760", "C810", "C860", "C900", "C940"]

    x["SpecSum"] = x[spectral_cols].sum(axis=1)

    for c in spectral_cols:
        x[f"{c}_norm"] = x[c] / (x["SpecSum"] + eps)

    x["R_410_940"] = x["A410"] / (x["C940"] + eps)
    x["R_435_900"] = x["A435"] / (x["C900"] + eps)
    x["R_460_860"] = x["A460"] / (x["C860"] + eps)
    x["R_510_760"] = x["A510"] / (x["C760"] + eps)
    x["R_560_810"] = x["B560"] / (x["C810"] + eps)
    x["R_610_940"] = x["B610"] / (x["C940"] + eps)

    x["VisibleMean"] = x[visible_cols].mean(axis=1)
    x["NIRMean"] = x[nir_cols].mean(axis=1)
    x["VisibleNIRRatio"] = x["VisibleMean"] / (x["NIRMean"] + eps)

    x["VisibleSum"] = x[visible_cols].sum(axis=1)
    x["NIRSum"] = x[nir_cols].sum(axis=1)

    x["Diff_410_435"] = x["A410"] - x["A435"]
    x["Diff_435_460"] = x["A435"] - x["A460"]
    x["Diff_760_810"] = x["C760"] - x["C810"]
    x["Diff_860_940"] = x["C860"] - x["C940"]

    x["Log_410"] = np.log1p(x["A410"])
    x["Log_940"] = np.log1p(x["C940"])

    x["SpecMax"] = x[spectral_cols].max(axis=1)
    x["SpecMin"] = x[spectral_cols].min(axis=1)
    x["SpecStd"] = x[spectral_cols].std(axis=1)
    x["SpecRange"] = x["SpecMax"] - x["SpecMin"]

    x["Slope_410_940"] = (x["C940"] - x["A410"]) / (940 - 410)
    x["Slope_435_860"] = (x["C860"] - x["A435"]) / (860 - 435)
    x["Slope_510_760"] = (x["C760"] - x["A510"]) / (760 - 510)

    return x

def validate_channels(channels_dict):
    required = set(spectral_cols)
    received = set(channels_dict.keys())

    missing = required - received
    extra = received - required

    if missing:
        raise ValueError(f"Missing channels: {sorted(list(missing))}")
    if extra:
        raise ValueError(f"Unexpected channels: {sorted(list(extra))}")

def predict_milk_from_channels(channels_dict):
    validate_channels(channels_dict)

    sample = pd.DataFrame([channels_dict])

    # Ensure numeric
    for c in spectral_cols:
        sample[c] = pd.to_numeric(sample[c], errors="coerce")

    if sample[spectral_cols].isna().sum().sum() > 0:
        raise ValueError("One or more channel values are invalid or non-numeric.")

    sample_feat = add_engineered_features(sample)

    # Brand prediction using selected RFECV features
    brand_input = sample_feat[brand_input_cols]
    brand_pred_enc = brand_pipeline.predict(brand_input)[0]
    brand_pred = brand_encoder.inverse_transform([brand_pred_enc])[0]

    # Water prediction using full feature set
    water_input = sample_feat[feature_cols]
    water_pred = float(water_pipeline.predict(water_input)[0])
    water_pred = max(0.0, min(50.0, water_pred))

    # Snap to nearest 5% because dataset levels are in 5% increments
    water_pred = round(water_pred / 5) * 5
    water_pred = max(0.0, min(50.0, water_pred))

    status_pred = get_status_from_water(water_pred)
    quality_score = get_quality_score(water_pred)

    # based on your current MAE regime
    uncertainty = "±7%"

    return {
        "predictedBrand": str(brand_pred),
        "waterPercent": float(water_pred),
        "uncertainty": uncertainty,
        "status": str(status_pred),
        "qualityScore": round(float(quality_score), 2)
    }

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Milk Adulteration Prediction API",
        "status": "running",
        "health_endpoint": "/health",
        "prediction_endpoint": "/predict/milk"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": True,
        "brand_classes": metadata.get("brands", []),
        "brand_feature_mode": metadata.get("brand_feature_mode", "unknown")
    })

@app.route("/predict/milk", methods=["POST"])
def predict_milk():
    try:
        payload = request.get_json(force=True)

        if payload is None:
            return jsonify({
                "success": False,
                "error": "No JSON payload received."
            }), 400

        # Support two formats:
        # 1) {"channels": {"A410":..., ..., "C940":...}}
        # 2) {"channels": [18 values in spectral_cols order]}
        if "channels" not in payload:
            return jsonify({
                "success": False,
                "error": "Missing 'channels' in request body."
            }), 400

        channels = payload["channels"]

        if isinstance(channels, dict):
            channels_dict = channels

        elif isinstance(channels, list):
            if len(channels) != len(spectral_cols):
                return jsonify({
                    "success": False,
                    "error": f"Expected {len(spectral_cols)} channel values, got {len(channels)}."
                }), 400
            channels_dict = dict(zip(spectral_cols, channels))

        else:
            return jsonify({
                "success": False,
                "error": "'channels' must be either a dictionary or a list."
            }), 400

        result = predict_milk_from_channels(channels_dict)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/test", methods=["GET"])
def test():
    # optional hardcoded sample for quick deployment testing
    sample = {c: 1.0 for c in spectral_cols}
    try:
        result = predict_milk_from_channels(sample)
        return jsonify({
            "success": True,
            "message": "Test route working",
            "result": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    port = 5000
    print(f"Milk API running -> http://localhost:{port}/health")
    app.run(host="0.0.0.0", port=port, debug=False)