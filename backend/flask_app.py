from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from pathlib import Path

# Resolve artifacts at project root
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "random_forest_model.pkl"
SCALER_PATH = ROOT / "minmax_scaler.pkl"

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load scaler at {SCALER_PATH}: {e}")

app = Flask(__name__)
# Allow all origins to avoid dev-time CORS issues
CORS(app, resources={r"/*": {"origins": "*"}})


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def index():
    return {"service": "SGEMM Predictor (Flask)", "status": "ok"}

@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True)
        # Validate required keys
        required = [
            "MWG","NWG","KWG","MDIMC","NDIMC",
            "MDIMA","NDIMB","KWI","VWM","VWN",
            "STRM","STRN","SA","SB"
        ]
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing keys: {missing}"}), 400

        # First 10 -> scaler
        first10 = np.array([
            data["MWG"], data["NWG"], data["KWG"], data["MDIMC"], data["NDIMC"],
            data["MDIMA"], data["NDIMB"], data["KWI"], data["VWM"], data["VWN"]
        ], dtype=float).reshape(1, -1)

        try:
            first10_scaled = scaler.transform(first10)
        except Exception as e:
            return jsonify({"error": f"Scaler transform failed: {e}"}), 500

        # Append 4 binary features
        last4 = np.array([[data["STRM"], data["STRN"], data["SA"], data["SB"]]], dtype=float)
        X = np.concatenate([first10_scaled, last4], axis=1)

        # Predict log1p(ms)
        y_log = model.predict(X)
        y_ms = float(np.expm1(y_log[0]))
        y_ms = max(y_ms, 0.0)
        return jsonify({"predicted_time_ms": round(y_ms, 5)})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


if __name__ == "__main__":
    # Run on port 8000 to match frontend fetch
    print("Starting Flask SGEMM Predictor on http://127.0.0.1:8000 ...")
    app.run(host="127.0.0.1", port=8000, debug=True)
