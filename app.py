import sys
import os
import json
import threading
import base64
import subprocess
import io

# Add ml_pipeline dir to path for imports
PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "S7 new", "ml_pipeline"))
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from flask import Flask, request, jsonify, send_from_directory, send_file
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder=PIPELINE_DIR, static_url_path="")

# =====================================================
# PIPELINE STATE
# =====================================================
pipeline_state = {
    "status": "idle",
    "current_step": "",
    "logs": "",
    "active_step_index": -1
}
lock = threading.Lock()
running_process = None
stopped_by_user = False

# =====================================================
# SERVE DASHBOARD HTML (index)
# =====================================================
@app.route("/")
def index():
    return send_from_directory(PIPELINE_DIR, "dashboard.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(PIPELINE_DIR, filename)

# =====================================================
# ESP32 PPG DATA RECEIVER
# =====================================================
@app.route("/api/ppg", methods=["POST", "OPTIONS"])
@app.route("/api/esp32/dataset", methods=["POST", "OPTIONS"])
def receive_ppg():
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Device-ID"
        return resp

    device_id = request.headers.get("X-Device-ID", "PVD_PATCH_001")

    # Try JSON first, then raw CSV
    csv_text = ""
    subject_id = device_id
    label = "normal"
    try:
        data = request.get_json(silent=True)
        if data:
            subject_id = data.get("subject_id", subject_id)
            label = data.get("label", label)
            csv_text = data.get("csv_data", data.get("csv", ""))
    except Exception:
        pass

    if not csv_text:
        csv_text = request.data.decode("utf-8", errors="ignore")

    if not label or label.lower() not in ("normal", "moderate", "high"):
        label = "normal"

    safe_subject = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in subject_id)
    filename = f"{safe_subject}_{label.lower()}.csv"

    raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    dest = os.path.join(raw_dir, filename)
    try:
        with open(dest, "w", encoding="utf-8") as f:
            f.write(csv_text)
        # Count samples
        lines = [l for l in csv_text.strip().split("\n") if l]
        samples = max(0, len(lines) - 1)  # subtract header
        print(f"[PPG] Received {samples} samples from {device_id} -> {filename}")
        return jsonify({
            "status": "success",
            "device_id": device_id,
            "samples_received": samples,
            "filename": filename
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =====================================================
# PIPELINE STATUS
# =====================================================
@app.route("/api/pipeline/status", methods=["GET"])
def pipeline_status():
    with lock:
        return jsonify(dict(pipeline_state))

# =====================================================
# PIPELINE STOP
# =====================================================
@app.route("/api/pipeline/stop", methods=["POST"])
def pipeline_stop():
    global stopped_by_user, running_process
    with lock:
        if pipeline_state["status"] != "running" or not running_process:
            return jsonify({"error": "No pipeline task is currently running."}), 400
        stopped_by_user = True
        try:
            running_process.terminate()
        except Exception:
            pass
    return jsonify({"status": "stopping"})

# =====================================================
# PIPELINE RUN
# =====================================================
def run_script_thread(script_name, step_key, index):
    global pipeline_state, running_process, stopped_by_user
    script_path = os.path.join(PIPELINE_DIR, script_name)
    python_exe = sys.executable
    with lock:
        pipeline_state["status"] = "running"
        pipeline_state["current_step"] = step_key
        pipeline_state["active_step_index"] = index
        pipeline_state["logs"] = f"=== Starting Step: {step_key} ===\n"
        stopped_by_user = False
    try:
        process = subprocess.Popen(
            [python_exe, script_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8"
        )
        with lock:
            running_process = process
        for line in process.stdout:
            with lock:
                pipeline_state["logs"] += line
        process.wait()
        with lock:
            running_process = None
            if stopped_by_user:
                pipeline_state["status"] = "failed"
            elif process.returncode == 0:
                pipeline_state["status"] = "success"
            else:
                pipeline_state["status"] = "failed"
    except Exception as e:
        with lock:
            running_process = None
            pipeline_state["status"] = "failed"
            pipeline_state["logs"] += f"\nError: {e}\n"

@app.route("/api/pipeline/run", methods=["POST"])
def pipeline_run():
    global pipeline_state
    data = request.get_json(silent=True) or {}
    step = data.get("step", "")

    with lock:
        if pipeline_state["status"] == "running":
            return jsonify({"error": "A pipeline task is already running."}), 400

    steps_map = {
        "generate_synthetic": ("generate_synthetic_data.py", "Generate Synthetic Data", 0),
        "preprocess":         ("data_preprocessing.py",      "Preprocess Raw Data",     1),
        "extract_features":   ("feature_extraction.py",      "Extract Features",        2),
        "train_model":        ("train_model.py",              "Train Risk Classifier",   3),
        "convert_tflite":     ("convert_to_tflite.py",       "Export TFLite Model",     4),
    }

    if step in steps_map:
        script, name, idx = steps_map[step]
        t = threading.Thread(target=run_script_thread, args=(script, name, idx))
        t.daemon = True
        t.start()
        return jsonify({"status": "started", "step": step})
    else:
        return jsonify({"error": f"Unknown step: {step}"}), 400

# =====================================================
# DATASET UPLOAD
# =====================================================
@app.route("/api/dataset/upload", methods=["POST"])
def dataset_upload():
    params = request.get_json(silent=True) or {}
    clear_existing = params.get("clear_existing", False)
    files = params.get("files", [])

    raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
    if clear_existing and os.path.exists(raw_dir):
        for f in os.listdir(raw_dir):
            if f.endswith(".csv"):
                try:
                    os.remove(os.path.join(raw_dir, f))
                except Exception:
                    pass

    os.makedirs(raw_dir, exist_ok=True)
    saved, errors = [], []

    for fi in files:
        fname = fi.get("filename", "")
        content_b64 = fi.get("content", "")
        if not fname or not content_b64:
            continue
        try:
            data = base64.b64decode(content_b64)
            ext = os.path.splitext(fname)[1].lower()
            if ext == ".csv":
                with open(os.path.join(raw_dir, fname), "wb") as f:
                    f.write(data)
                saved.append(fname)
            elif ext == ".zip":
                import zipfile
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for zi in z.infolist():
                        n = os.path.basename(zi.filename)
                        if n and zi.filename.endswith(".csv") and not zi.is_dir():
                            with open(os.path.join(raw_dir, n), "wb") as f:
                                f.write(z.read(zi.filename))
                            saved.append(n)
        except Exception as e:
            errors.append(f"{fname}: {e}")

    return jsonify({"status": "success" if not errors else "partial_error", "uploaded": saved, "errors": errors})

# =====================================================
# PREDICT
# =====================================================
_model = None
_scaler_mean = None
_scaler_scale = None
_model_lock = threading.Lock()

@app.route("/api/predict", methods=["POST"])
def predict():
    global _model, _scaler_mean, _scaler_scale
    params = request.get_json(silent=True) or {}
    csv_data = params.get("csv_data", "")
    subject_id = params.get("subject_id", "esp32_subject")
    clinician_label = params.get("clinician_label", "normal")
    save_dataset = params.get("save_dataset", False)

    if not csv_data:
        return jsonify({"error": "No CSV data provided"}), 400

    try:
        import data_preprocessing as dp
        import feature_extraction as fe
    except Exception as e:
        return jsonify({"error": f"Pipeline modules not available: {e}"}), 500

    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {e}"}), 400

    if len(df) < 1600:
        return jsonify({"error": "Need at least 1600 samples (8 sec at 200 Hz)"}), 400

    if save_dataset:
        raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in subject_id)
        lbl = clinician_label.lower() if clinician_label.lower() in ("normal", "moderate", "high") else "normal"
        df.to_csv(os.path.join(raw_dir, f"{safe}_{lbl}.csv"), index=False)

    try:
        sos = dp.design_bandpass()
        windows = dp.segment_windows(df, sos, label=0, subject_id=subject_id)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {e}"}), 500

    if not windows:
        return jsonify({"error": "No windows extracted"}), 400

    features_list = []
    window_start_times = []
    for win in windows:
        try:
            feats = fe.extract_features_row(win)
            features_list.append(feats)
            window_start_times.append(win["window_start_ms"])
        except Exception as e:
            return jsonify({"error": f"Feature extraction failed: {e}"}), 500

    with _model_lock:
        if _model is None:
            try:
                import tensorflow as tf
                model_path = os.path.join(PIPELINE_DIR, "model", "model.h5")
                if not os.path.exists(model_path):
                    return jsonify({"error": "Model not trained yet. Run the ML pipeline first."}), 400
                _model = tf.keras.models.load_model(model_path)
            except Exception as e:
                return jsonify({"error": f"Model load failed: {e}"}), 500
        if _scaler_mean is None:
            try:
                _scaler_mean = np.load(os.path.join(PIPELINE_DIR, "model", "scaler_mean.npy"))
                _scaler_scale = np.load(os.path.join(PIPELINE_DIR, "model", "scaler_scale.npy"))
            except Exception as e:
                return jsonify({"error": f"Scaler load failed: {e}"}), 500

    X = np.array(features_list, dtype=np.float32)
    X_scaled = (X - _scaler_mean) / _scaler_scale
    probs = _model.predict(X_scaled)

    label_names = ["Normal", "Moderate", "High"]
    results = []
    for i, wp in enumerate(probs):
        results.append({
            "window_index": i,
            "window_start_ms": float(window_start_times[i]),
            "probabilities": [float(p) for p in wp],
            "risk_level": int(np.argmax(wp)),
            "risk_name": label_names[int(np.argmax(wp))]
        })

    avg_probs = np.mean(probs, axis=0).tolist()
    final_risk = int(np.argmax(avg_probs))

    return jsonify({
        "status": "success",
        "windows": results,
        "summary": {
            "total_windows": len(results),
            "average_probabilities": avg_probs,
            "final_risk_level": final_risk,
            "final_risk_name": label_names[final_risk]
        }
    })

# =====================================================
# CLEAR DATA
# =====================================================
@app.route("/api/pipeline/clear_data", methods=["POST"])
def clear_data():
    import shutil
    raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
    data_dir = os.path.join(PIPELINE_DIR, "data")
    model_dir = os.path.join(PIPELINE_DIR, "model")
    logs = "=== Clearing Pipeline Data ===\n"

    for csv_f in (os.listdir(raw_dir) if os.path.exists(raw_dir) else []):
        if csv_f.endswith(".csv"):
            try:
                os.remove(os.path.join(raw_dir, csv_f))
            except Exception:
                pass

    for fname in ["processed.json", "features.csv"]:
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            os.remove(p)
            logs += f"Removed {fname}\n"

    for fname in ["model.h5", "scaler_mean.npy", "scaler_scale.npy", "training_report.json", "model.tflite"]:
        p = os.path.join(model_dir, fname)
        if os.path.exists(p):
            os.remove(p)
            logs += f"Removed {fname}\n"

    saved_model = os.path.join(model_dir, "saved_model")
    if os.path.exists(saved_model):
        shutil.rmtree(saved_model)
        logs += "Removed saved_model/\n"

    logs += "=== Cleanup Complete ===\n"
    with lock:
        pipeline_state["status"] = "idle"
        pipeline_state["logs"] = logs
        pipeline_state["active_step_index"] = -1

    return jsonify({"status": "cleared", "logs": logs})

# =====================================================
# SERVE DATA & MODEL FILES
# =====================================================
@app.route("/data/<path:filename>")
def serve_data(filename):
    data_dir = os.path.join(PIPELINE_DIR, "data")
    return send_from_directory(data_dir, filename, as_attachment=True)

@app.route("/model/<path:filename>")
def serve_model(filename):
    model_dir = os.path.join(PIPELINE_DIR, "model")
    return send_from_directory(model_dir, filename, as_attachment=True)

# =====================================================
# CORS HEADERS
# =====================================================
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print(f"[*] PVD PPG Dashboard starting on port {port}")
    print(f"[*] Dashboard: http://localhost:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
