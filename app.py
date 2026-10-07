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
from datetime import datetime
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
# ESP32 ACQUISITION CONTROL (/api/recording/start, stop, status)
# =====================================================
recording_state = {
    "recording": False,
    "subject_id": "PVD_PATCH_001",
    "label": "normal"
}

@app.route("/api/recording/status", methods=["GET"])
def api_recording_status():
    return jsonify(recording_state), 200

@app.route("/api/recording/start", methods=["POST", "OPTIONS"])
def api_recording_start():
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp
    data = request.get_json(silent=True) or {}
    recording_state["recording"] = True
    recording_state["subject_id"] = data.get("subject_id", "PVD_PATCH_001")
    recording_state["label"] = data.get("label", "normal")
    print(f"[RECORDING] Started: {recording_state['subject_id']} ({recording_state['label']})")
    return jsonify({"status": "recording_started", **recording_state}), 200

@app.route("/api/recording/stop", methods=["POST", "OPTIONS"])
def api_recording_stop():
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp
    recording_state["recording"] = False
    print("[RECORDING] Stopped")
    return jsonify({"status": "recording_stopped"}), 200

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
        recording_state["recording"] = False
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
# RAW DATASETS MANAGEMENT (/api/raw/list, label, delete)
# =====================================================
@app.route("/api/raw/list", methods=["GET"])
def raw_list():
    raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    files_list = []
    for f in os.listdir(raw_dir):
        if f.endswith('.csv'):
            filepath = os.path.join(raw_dir, f)
            size = os.path.getsize(filepath)
            mtime = os.path.getmtime(filepath)
            from datetime import timezone
            dt_utc = datetime.fromtimestamp(mtime, tz=timezone.utc)
            captured_at = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
            iso_timestamp = dt_utc.isoformat()
            
            base, _ = os.path.splitext(f)
            parts = base.split("_")
            if len(parts) > 1:
                label_str = parts[-1].lower()
                is_known = False
                for l in ('normal', 'moderate', 'high'):
                    if label_str.startswith(l):
                        label_str = l
                        is_known = True
                        break
                if not is_known:
                    label_str = "unlabeled"
                subject_id = "_".join(parts[:-1])
            else:
                label_str = "unlabeled"
                subject_id = base
                
            files_list.append({
                "filename": f,
                "size_bytes": size,
                "subject_id": subject_id,
                "label": label_str,
                "captured_at": captured_at,
                "iso_timestamp": iso_timestamp,
                "timestamp": int(mtime)
            })
    return jsonify(files_list)

@app.route("/api/raw/label", methods=["POST"])
def raw_label():
    params = request.get_json(silent=True) or {}
    filename = os.path.basename(params.get("filename", ""))
    new_label = params.get("new_label", "").lower()
    if not filename or new_label not in ('normal', 'moderate', 'high'):
        return jsonify({"error": "Invalid request"}), 400
    raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
    src_path = os.path.join(raw_dir, filename)
    if not os.path.exists(src_path):
        return jsonify({"error": "File not found"}), 404

    base, ext = os.path.splitext(filename)
    parts = base.split("_")
    if len(parts) > 1:
        last_part = parts[-1].lower()
        is_known = any(last_part.startswith(l) for l in ('normal', 'moderate', 'high'))
        if is_known:
            parts[-1] = new_label
            new_filename = "_".join(parts) + ext
        else:
            new_filename = f"{base}_{new_label}{ext}"
    else:
        new_filename = f"{base}_{new_label}{ext}"

    dest_path = os.path.join(raw_dir, new_filename)
    if os.path.exists(dest_path) and dest_path != src_path:
        counter = 1
        while True:
            temp_base = os.path.splitext(new_filename)[0]
            temp_filename = f"{temp_base}_{counter}{ext}"
            temp_path = os.path.join(raw_dir, temp_filename)
            if not os.path.exists(temp_path):
                new_filename = temp_filename
                dest_path = temp_path
                break
            counter += 1

    try:
        os.rename(src_path, dest_path)
        return jsonify({"status": "success", "new_filename": new_filename})
    except Exception as e:
        return jsonify({"error": f"Rename failed: {e}"}), 500

@app.route("/api/raw/delete", methods=["POST"])
def raw_delete():
    params = request.get_json(silent=True) or {}
    filename = os.path.basename(params.get("filename", ""))
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400
    raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
    filepath = os.path.join(raw_dir, filename)
    if not os.path.exists(filepath):
        return jsonify({"status": "success", "message": "File already removed"}), 200
    try:
        os.remove(filepath)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": f"Delete failed: {e}"}), 500

@app.route("/api/raw/delete_batch", methods=["POST"])
def raw_delete_batch():
    params = request.get_json(silent=True) or {}
    filenames = params.get("filenames", [])
    if not filenames:
        return jsonify({"error": "No filenames provided"}), 400
    raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
    deleted = []
    errors = []
    for fn in filenames:
        safe_fn = os.path.basename(fn)
        if safe_fn.endswith(".csv"):
            fp = os.path.join(raw_dir, safe_fn)
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                    deleted.append(safe_fn)
                except Exception as e:
                    errors.append(f"{safe_fn}: {str(e)}")
    return jsonify({"status": "success", "deleted_count": len(deleted), "errors": errors})

@app.route("/api/raw/delete_all", methods=["POST"])
def raw_delete_all():
    raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
    deleted = 0
    if os.path.exists(raw_dir):
        for f in os.listdir(raw_dir):
            if f.endswith(".csv"):
                try:
                    os.remove(os.path.join(raw_dir, f))
                    deleted += 1
                except Exception:
                    pass
    return jsonify({"status": "success", "deleted_count": deleted})

@app.route("/api/raw/download/<path:filename>", methods=["GET"])
def download_single_raw(filename):
    safe_fn = os.path.basename(filename)
    raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
    fp = os.path.join(raw_dir, safe_fn)
    if os.path.exists(fp) and safe_fn.endswith(".csv"):
        return send_file(fp, mimetype="text/csv", as_attachment=True, download_name=safe_fn)
    return jsonify({"error": "File not found"}), 404

@app.route("/api/download/raw", methods=["GET"])
def download_raw():
    import zipfile
    raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        return jsonify({"error": "No raw data found"}), 404
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(raw_dir):
            for file in files:
                if file.endswith('.csv'):
                    zf.write(os.path.join(root, file), file)
    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name="synthetic_raw_data.zip")

@app.route("/api/config", methods=["GET"])
def get_config():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "data_preprocessing",
            os.path.join(PIPELINE_DIR, "data_preprocessing.py")
        )
        dp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dp)
        return jsonify({
            "sample_rate_hz":   int(dp.FS),
            "bandpass_low_hz":  float(dp.BANDPASS[0]),
            "bandpass_high_hz": float(dp.BANDPASS[1]),
            "window_seconds":   int(dp.WINDOW_SECONDS),
            "window_samples":   int(dp.WINDOW_LEN),
            "step_seconds":     int(dp.STEP_SECONDS),
            "overlap_pct":      int((1 - dp.STEP_SECONDS / dp.WINDOW_SECONDS) * 100),
            "target_arch":      "ESP32-S3 (TensorFlow Lite Micro)"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/scaler", methods=["GET"])
def get_scaler():
    mean_p = os.path.join(PIPELINE_DIR, "model", "scaler_mean.npy")
    scale_p = os.path.join(PIPELINE_DIR, "model", "scaler_scale.npy")
    if os.path.exists(mean_p) and os.path.exists(scale_p):
        try:
            mean = np.load(mean_p)
            scale = np.load(scale_p)
            feats = [
                "PI_finger", "PI_toe", "PI_ratio", "PTT_ft",
                "AI_finger", "AI_toe", "HRV_rmssd", "dicrotic_ratio",
                "temp_finger", "temp_toe", "temp_diff",
                "accel_std", "gyro_std",
            ]
            resp = {}
            for i, f in enumerate(feats):
                if i < len(mean) and i < len(scale):
                    resp[f] = {"mean": float(mean[i]), "scale": float(scale[i])}
            return jsonify(resp)
        except Exception as e:
            return jsonify({"error": str(e)})
    return jsonify({"error": "Scaler files not found. Run train_model.py first."})

@app.route("/api/model/report", methods=["GET"])
def get_model_report():
    report_path = os.path.join(PIPELINE_DIR, "model", "training_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)})
    return jsonify({"error": "Report not found"})

@app.route("/api/code", methods=["GET"])
def get_code():
    filename = request.args.get("file", "")
    allowed = ["generate_synthetic_data.py", "data_preprocessing.py", "feature_extraction.py", "train_model.py", "convert_to_tflite.py"]
    if filename not in allowed:
        return jsonify({"error": "Invalid or unauthorized file"}), 400
    fp = os.path.join(PIPELINE_DIR, filename)
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/plain; charset=utf-8"}
    return jsonify({"error": "File not found"}), 404

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
_model = None
_scaler_mean = None
_scaler_scale = None
_model_lock = threading.Lock()

try:
    import model_assets
    _m_dir = os.path.join(PIPELINE_DIR, "model")
    _model, _scaler_mean, _scaler_scale = model_assets.load_model_and_scalers(_m_dir)
    print("[*] Preloaded ML model and normalization scalers successfully.")
except Exception as _init_err:
    print(f"[!] Model preload warning: {_init_err}")

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

    try:
        sos = dp.design_bandpass()
        windows = dp.segment_windows(df, sos, label=0, subject_id=subject_id)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {e}"}), 500

    if not windows:
        return jsonify({"error": "No windows extracted"}), 400

    feature_names = [
        "PI_finger", "PI_toe", "PI_ratio", "PTT_ft",
        "AI_finger", "AI_toe", "HRV_rmssd", "dicrotic_ratio",
        "temp_finger", "temp_toe", "temp_diff",
        "accel_std", "gyro_std"
    ]

    features_list = []
    window_start_times = []
    window_features_dict = []
    for win in windows:
        try:
            feats = fe.extract_features_row(win)
            features_list.append(feats)
            window_start_times.append(win["window_start_ms"])
            f_dict = {k: float(v) for k, v in zip(feature_names, feats)}
            window_features_dict.append(f_dict)
        except Exception as e:
            return jsonify({"error": f"Feature extraction failed: {e}"}), 500

    with _model_lock:
        if _model is None or _scaler_mean is None or _scaler_scale is None:
            try:
                import model_assets
                model_dir = os.path.join(PIPELINE_DIR, "model")
                _model, _scaler_mean, _scaler_scale = model_assets.load_model_and_scalers(model_dir)
            except Exception as e:
                return jsonify({"error": f"Model initialization failed: {e}"}), 500

    X = np.array(features_list, dtype=np.float32)
    X_scaled = (X - _scaler_mean) / _scaler_scale
    probs = _model.predict(X_scaled)

    label_names = ["Normal", "Moderate", "High"]
    results = []
    for i, wp in enumerate(probs):
        results.append({
            "window_index": i,
            "window_start_ms": float(window_start_times[i]),
            "features": window_features_dict[i],
            "probabilities": [float(p) for p in wp],
            "risk_level": int(np.argmax(wp)),
            "risk_name": label_names[int(np.argmax(wp))]
        })

    avg_probs = np.mean(probs, axis=0).tolist()
    final_risk = int(np.argmax(avg_probs))
    predicted_label = label_names[final_risk].lower()

    saved_filename = None
    if save_dataset:
        raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in subject_id)
        # If clinician explicitly selected normal, moderate, or high, use that;
        # otherwise (auto, random, empty), auto-label with the model's predicted risk!
        if clinician_label and clinician_label.lower() in ("normal", "moderate", "high"):
            lbl = clinician_label.lower()
        else:
            lbl = predicted_label
        saved_filename = f"{safe}_{lbl}.csv"
        df.to_csv(os.path.join(raw_dir, saved_filename), index=False)

    expected_label = (clinician_label if clinician_label and clinician_label.lower() in ("normal", "moderate", "high") else label_names[final_risk]).capitalize()

    return jsonify({
        "status": "success",
        "windows": results,
        "summary": {
            "total_windows": len(results),
            "average_probabilities": avg_probs,
            "final_risk_level": final_risk,
            "final_risk_name": label_names[final_risk],
            "expected_label": expected_label,
            "saved": bool(save_dataset and saved_filename),
            "saved_filename": saved_filename
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

    for fname in ["model.tflite"]:
        p = os.path.join(model_dir, fname)
        if os.path.exists(p):
            os.remove(p)
            logs += f"Removed {fname}\n"

    # Always ensure production model and scalers remain ready for live prediction
    try:
        import model_assets
        model_assets.ensure_model_files(model_dir)
    except Exception:
        pass

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
