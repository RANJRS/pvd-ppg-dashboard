import http.server
import socketserver
import os
import json
import webbrowser
import numpy as np
import subprocess
import threading
import base64

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = SCRIPT_DIR
PORT = int(os.environ.get("PORT", 8000))

# Global thread-safe state for executing pipeline scripts
pipeline_state = {
    "status": "idle",  # "idle", "running", "success", "failed"
    "current_step": "",
    "logs": "",
    "active_step_index": -1
}
lock = threading.Lock()
running_process = None
stopped_by_user = False

def run_script_thread(script_name, step_key, index):
    global pipeline_state, running_process, stopped_by_user
    
    script_path = os.path.join(PIPELINE_DIR, script_name)
    python_exe = os.path.abspath(os.path.join(PIPELINE_DIR, "..", "..", "venv", "Scripts", "python.exe"))
    if not os.path.exists(python_exe):
        python_exe = "python"
        
    with lock:
        pipeline_state["status"] = "running"
        pipeline_state["current_step"] = step_key
        pipeline_state["active_step_index"] = index
        pipeline_state["logs"] = f"=== Starting Step: {step_key} ===\n"
        stopped_by_user = False

    try:
        process = subprocess.Popen(
            [python_exe, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8"
        )
        
        with lock:
            running_process = process
        
        while True:
            line = process.stdout.readline()
            if not line:
                break
            with lock:
                pipeline_state["logs"] += line
                
        process.wait()
        
        with lock:
            running_process = None
            if stopped_by_user:
                pipeline_state["status"] = "failed"
                pipeline_state["logs"] += f"\n=== Stopped by User: {step_key} ===\n"
            elif process.returncode == 0:
                pipeline_state["status"] = "success"
                pipeline_state["logs"] += f"\n=== Success: {step_key} ===\n"
            else:
                pipeline_state["status"] = "failed"
                pipeline_state["logs"] += f"\n=== Failed with exit code {process.returncode} ===\n"
                
    except Exception as e:
        with lock:
            running_process = None
            pipeline_state["status"] = "failed"
            pipeline_state["logs"] += f"\nInternal Error: {str(e)}\n"

def run_full_pipeline_thread():
    steps = [
        ("generate_synthetic_data.py", "Generate Synthetic Data", 0),
        ("data_preprocessing.py", "Preprocess Raw Data", 1),
        ("feature_extraction.py", "Extract Features", 2),
        ("train_model.py", "Train Risk Classifier", 3),
        ("convert_to_tflite.py", "Quantize & Export TFLite Model", 4)
    ]
    
    global pipeline_state, running_process, stopped_by_user
    python_exe = os.path.abspath(os.path.join(PIPELINE_DIR, "..", "..", "venv", "Scripts", "python.exe"))
    if not os.path.exists(python_exe):
        python_exe = "python"
        
    with lock:
        pipeline_state["status"] = "running"
        pipeline_state["current_step"] = "Full Pipeline"
        pipeline_state["active_step_index"] = 99
        pipeline_state["logs"] = "=== Starting Full ML Pipeline Run ===\n"
        stopped_by_user = False
        
    for script_name, step_key, idx in steps:
        with lock:
            if stopped_by_user:
                break
            pipeline_state["active_step_index"] = idx
            pipeline_state["logs"] += f"\n>>> Launching Step {idx + 1}: {step_key}...\n"
            
        script_path = os.path.join(PIPELINE_DIR, script_name)
        try:
            process = subprocess.Popen(
                [python_exe, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8"
            )
            
            with lock:
                running_process = process
            
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                with lock:
                    pipeline_state["logs"] += line
                    
            process.wait()
            
            with lock:
                running_process = None
                
            if stopped_by_user:
                with lock:
                    pipeline_state["status"] = "failed"
                    pipeline_state["logs"] += f"\n=== Pipeline Stopped by User at Step: {step_key} ===\n"
                return
                
            if process.returncode != 0:
                with lock:
                    pipeline_state["status"] = "failed"
                    pipeline_state["logs"] += f"\n=== Pipeline Failed at Step: {step_key} ===\n"
                return
        except Exception as e:
            with lock:
                running_process = None
                pipeline_state["status"] = "failed"
                pipeline_state["logs"] += f"\nInternal Error during {step_key}: {str(e)}\n"
            return
            
    with lock:
        if not stopped_by_user:
            pipeline_state["status"] = "success"
            pipeline_state["active_step_index"] = -1
            pipeline_state["logs"] += "\n=== Full ML Pipeline Completed Successfully! ===\n"

def run_train_pipeline_thread():
    steps = [
        ("data_preprocessing.py", "Preprocess Raw Data", 1),
        ("feature_extraction.py", "Extract Features", 2),
        ("train_model.py", "Train Risk Classifier", 3),
        ("convert_to_tflite.py", "Quantize & Export TFLite Model", 4)
    ]
    
    global pipeline_state, running_process, stopped_by_user
    python_exe = os.path.abspath(os.path.join(PIPELINE_DIR, "..", "..", "venv", "Scripts", "python.exe"))
    if not os.path.exists(python_exe):
        python_exe = "python"
        
    with lock:
        pipeline_state["status"] = "running"
        pipeline_state["current_step"] = "Feature Extraction & Model Training"
        pipeline_state["active_step_index"] = 98
        pipeline_state["logs"] = "=== Starting Preprocessing, Feature Extraction & Training ===\n"
        stopped_by_user = False
        
    for script_name, step_key, idx in steps:
        with lock:
            if stopped_by_user:
                break
            pipeline_state["active_step_index"] = idx
            pipeline_state["logs"] += f"\n>>> Launching Step {idx + 1}: {step_key}...\n"
            
        script_path = os.path.join(PIPELINE_DIR, script_name)
        try:
            process = subprocess.Popen(
                [python_exe, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8"
            )
            
            with lock:
                running_process = process
            
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                with lock:
                    pipeline_state["logs"] += line
                    
            process.wait()
            
            with lock:
                running_process = None
                
            if stopped_by_user:
                with lock:
                    pipeline_state["status"] = "failed"
                    pipeline_state["logs"] += f"\n=== Pipeline Stopped by User at Step: {step_key} ===\n"
                return
                
            if process.returncode != 0:
                with lock:
                    pipeline_state["status"] = "failed"
                    pipeline_state["logs"] += f"\n=== Pipeline Failed at Step: {step_key} ===\n"
                return
        except Exception as e:
            with lock:
                running_process = None
                pipeline_state["status"] = "failed"
                pipeline_state["logs"] += f"\nInternal Error during {step_key}: {str(e)}\n"
            return
            
    with lock:
        if not stopped_by_user:
            pipeline_state["status"] = "success"
            pipeline_state["active_step_index"] = -1
            pipeline_state["logs"] += "\n=== Features Extracted & Model Trained Successfully! ===\n"


import io
import pandas as pd

# Global variables for model and scaler caching
_model = None
_scaler_mean = None
_scaler_scale = None
_model_lock = threading.Lock()

def predict_on_raw_dataset(csv_string, subject_id="esp32_subject", clinician_label="normal", save_dataset=False):
    global _model, _scaler_mean, _scaler_scale
    
    # 1. Parse CSV
    try:
        df = pd.read_csv(io.StringIO(csv_string))
    except Exception as e:
        return {"error": f"Failed to parse CSV data: {str(e)}"}
        
    required_cols = {"timestamp_ms", "ir_finger", "ir_toe"}
    if not required_cols.issubset(df.columns):
        return {"error": f"Missing required columns in CSV. Expected at minimum: {list(required_cols)}"}

    # Detect optional Temperature and IMU columns
    temp_cols = {"temp_finger", "temp_toe"}
    imu_cols  = {"ax", "ay", "az", "gx", "gy", "gz"}
    has_temp = temp_cols.issubset(df.columns)
    has_imu  = imu_cols.issubset(df.columns)
    if has_temp:
        print(f"[Predict] Temperature columns detected (temp_finger, temp_toe)")
    if has_imu:
        print(f"[Predict] IMU columns detected (ax, ay, az, gx, gy, gz)")
        
    if len(df) < 1600:
        return {"error": "Recording too short. Please record at least 8 seconds of data (1600 samples at 200 Hz)."}
        
    # 2. Check and save if requested
    saved_filename = ""
    saved_on_disk = False
    if save_dataset and subject_id and clinician_label:
        raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        # format: <subject_id>_<label>.csv
        safe_subject_id = "".join([c if c.isalnum() or c in ('-', '_') else '_' for c in subject_id])
        safe_label = clinician_label.lower()
        if safe_label not in ('normal', 'moderate', 'high'):
            safe_label = 'normal'
        filename = f"{safe_subject_id}_{safe_label}.csv"
        dest_path = os.path.join(raw_dir, filename)
        try:
            df.to_csv(dest_path, index=False)
            saved_filename = filename
            saved_on_disk = True
        except Exception as e:
            print(f"[Warning] Failed to save dataset: {str(e)}")
            
    # 3. Preprocess and Segment using imported helper or direct logic
    try:
        import data_preprocessing as dp
        import feature_extraction as fe
    except Exception as e:
        return {"error": f"Failed to load pipeline processing modules: {str(e)}"}
        
    try:
        sos = dp.design_bandpass()
        windows = dp.segment_windows(df, sos, label=0, subject_id=subject_id or "esp32")
    except Exception as e:
        return {"error": f"Failed during signal filtering/segmentation: {str(e)}"}
        
    if not windows:
        return {"error": "No segment windows could be extracted. Please record longer data."}
        
    # 4. Extract Features
    features_list = []
    window_start_times = []
    for win in windows:
        try:
            feats = fe.extract_features_row(win)
            features_list.append(feats)
            window_start_times.append(win["window_start_ms"])
        except Exception as e:
            return {"error": f"Failed to extract features from signal segments: {str(e)}"}
            
    # 5. Load Model and Scaler
    with _model_lock:
        if _model is None:
            try:
                import tensorflow as tf
                model_path = os.path.join(PIPELINE_DIR, "model", "model.h5")
                if not os.path.exists(model_path):
                    return {"error": "Trained model (model.h5) not found. Run model training first."}
                _model = tf.keras.models.load_model(model_path)
            except Exception as e:
                return {"error": f"Failed to load TensorFlow model: {str(e)}"}
                
        if _scaler_mean is None or _scaler_scale is None:
            scaler_mean_path = os.path.join(PIPELINE_DIR, "model", "scaler_mean.npy")
            scaler_scale_path = os.path.join(PIPELINE_DIR, "model", "scaler_scale.npy")
            if not os.path.exists(scaler_mean_path) or not os.path.exists(scaler_scale_path):
                return {"error": "Scaler normalization files not found. Run model training first."}
            try:
                _scaler_mean = np.load(scaler_mean_path)
                _scaler_scale = np.load(scaler_scale_path)
            except Exception as e:
                return {"error": f"Failed to load scaler variables: {str(e)}"}
                
    # 6. Normalize features and run inference
    try:
        X = np.array(features_list, dtype=np.float32)
        X_scaled = (X - _scaler_mean) / _scaler_scale
        probs = _model.predict(X_scaled)
    except Exception as e:
        return {"error": f"Failed during model prediction: {str(e)}"}
        
    # 7. Package results
    ALL_FEATURE_NAMES = [
        "PI_finger", "PI_toe", "PI_ratio", "PTT_ft",
        "AI_finger", "AI_toe", "HRV_rmssd", "dicrotic_ratio",
        "temp_finger", "temp_toe", "temp_diff",
        "accel_std", "gyro_std",
    ]
    results_windows = []
    label_names = ["Normal", "Moderate", "High"]
    for i, win_prob in enumerate(probs):
        win_prob_list = [float(p) for p in win_prob]
        risk_level = int(np.argmax(win_prob))
        feats_dict = {
            name: float(features_list[i][j])
            for j, name in enumerate(ALL_FEATURE_NAMES)
            if j < len(features_list[i])
        }
        results_windows.append({
            "window_index": i,
            "window_start_ms": float(window_start_times[i]),
            "features": feats_dict,
            "probabilities": win_prob_list,
            "risk_level": risk_level,
            "risk_name": label_names[risk_level]
        })
        
    avg_probs = np.mean(probs, axis=0).tolist()
    final_risk_level = int(np.argmax(avg_probs))
    final_risk_name = label_names[final_risk_level]
    
    # Calculate mock risk score for Twilio message context
    risk_score = (avg_probs[1] * 0.5 + avg_probs[2] * 1.0) * 100
    return {
        "status": "success",
        "windows": results_windows,
        "summary": {
            "total_windows": len(results_windows),
            "average_probabilities": avg_probs,
            "final_risk_level": final_risk_level,
            "final_risk_name": final_risk_name,
            "saved": saved_on_disk,
            "saved_filename": saved_filename
        }
    }


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Allow CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        # Disable browser caching
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_OPTIONS(self):
        # Handle Preflight OPTIONS
        self.send_response(200)
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_POST(self):
        global pipeline_state
        from urllib.parse import urlparse
        parsed_path = urlparse(self.path).path

        if parsed_path in ('/api/ppg', '/api/esp32/dataset'):
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            from urllib.parse import parse_qs
            query = parse_qs(urlparse(self.path).query)
            
            device_header = self.headers.get("X-Device-ID")
            subject_id = device_header or query.get("subject_id", [None])[0]
            label = query.get("label", [None])[0]
            csv_text = ""
            
            # Try parsing as JSON first
            try:
                data_json = json.loads(post_data.decode('utf-8'))
                subject_id = data_json.get("subject_id", subject_id)
                label = data_json.get("label", label)
                csv_text = data_json.get("csv_data", data_json.get("csv", ""))
            except Exception:
                # If not JSON, treat raw payload as CSV string
                csv_text = post_data.decode('utf-8', errors='ignore')
                
            if not subject_id:
                subject_id = "PVD_PATCH_001"
            if not label or label.lower() not in ('normal', 'moderate', 'high'):
                label = "normal"
                
            safe_subject_id = "".join([c if c.isalnum() or c in ('-', '_') else '_' for c in subject_id])
            safe_label = label.lower()
            filename = f"{safe_subject_id}_{safe_label}.csv"
            
            raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
            os.makedirs(raw_dir, exist_ok=True)
            dest_path = os.path.join(raw_dir, filename)
            
            try:
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(csv_text)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "filename": filename,
                    "message": f"Successfully received dataset from ESP32: {filename}"
                }).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Failed to save ESP32 dataset: {str(e)}"}).encode('utf-8'))
            return

        if parsed_path == '/api/dataset/upload':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            clear_existing = params.get("clear_existing", False)
            files = params.get("files", [])
            
            raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
            if clear_existing:
                if os.path.exists(raw_dir):
                    for f in os.listdir(raw_dir):
                        if f.endswith('.csv'):
                            try:
                                os.remove(os.path.join(raw_dir, f))
                            except Exception:
                                pass
                                
            os.makedirs(raw_dir, exist_ok=True)
            
            saved_files = []
            errors = []
            
            for file_info in files:
                filename = file_info.get("filename")
                content_b64 = file_info.get("content")
                
                if not filename or not content_b64:
                    continue
                
                ext = os.path.splitext(filename)[1].lower()
                try:
                    file_data = base64.b64decode(content_b64)
                    
                    if ext == ".zip":
                        import zipfile
                        import io
                        with zipfile.ZipFile(io.BytesIO(file_data)) as z:
                            for zip_info in z.infolist():
                                extracted_name = os.path.basename(zip_info.filename)
                                if not extracted_name or extracted_name.startswith('._'):
                                    continue
                                if zip_info.filename.endswith('.csv') and not zip_info.is_dir():
                                    
                                    csv_content = z.read(zip_info.filename)
                                    dest_path = os.path.join(raw_dir, extracted_name)
                                    with open(dest_path, "wb") as f:
                                        f.write(csv_content)
                                    saved_files.append(extracted_name)
                    elif ext == ".csv":
                        dest_path = os.path.join(raw_dir, filename)
                        with open(dest_path, "wb") as f:
                            f.write(file_data)
                        saved_files.append(filename)
                    else:
                        errors.append(f"Unsupported file type: {filename}")
                except Exception as e:
                    errors.append(f"Error saving {filename}: {str(e)}")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "success" if not errors and saved_files else "partial_error",
                "uploaded": saved_files,
                "errors": errors
            }).encode('utf-8'))
            return

        if parsed_path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            csv_data = params.get("csv_data", "")
            subject_id = params.get("subject_id", "esp32_subject")
            clinician_label = params.get("clinician_label", "normal")
            save_dataset = params.get("save_dataset", False)
            # phone_number parameter removed
            
            if not csv_data:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No CSV data provided"}).encode('utf-8'))
                return
                
            res = predict_on_raw_dataset(
                csv_string=csv_data,
                subject_id=subject_id,
                clinician_label=clinician_label,
                save_dataset=save_dataset
            )
            
            if "error" in res:
                self.send_response(400)
            else:
                self.send_response(200)
                
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))
            return

        if parsed_path == '/api/pipeline/stop':
            global stopped_by_user, running_process
            with lock:
                if pipeline_state["status"] != "running" or not running_process:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "No pipeline task is currently running."}).encode('utf-8'))
                    return
                
                stopped_by_user = True
                try:
                    running_process.terminate()
                except Exception as e:
                    pass
                
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "stopping"}).encode('utf-8'))
            return

        if parsed_path == '/api/pipeline/run':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            step = params.get("step")
            
            with lock:
                if pipeline_state["status"] == "running":
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "A pipeline task is already running."}).encode('utf-8'))
                    return
                
            steps_map = {
                "generate_synthetic": ("generate_synthetic_data.py", "Generate Synthetic Data", 0),
                "preprocess": ("data_preprocessing.py", "Preprocess Raw Data", 1),
                "extract_features": ("feature_extraction.py", "Extract Features", 2),
                "train_model": ("train_model.py", "Train Risk Classifier", 3),
                "convert_tflite": ("convert_to_tflite.py", "Quantize & Export TFLite Model", 4)
            }
            
            if step == "all":
                t = threading.Thread(target=run_full_pipeline_thread)
                t.start()
                response = {"status": "started", "step": "all"}
            elif step == "process_and_train":
                t = threading.Thread(target=run_train_pipeline_thread)
                t.start()
                response = {"status": "started", "step": "process_and_train"}
            elif step in steps_map:
                script, name, idx = steps_map[step]
                t = threading.Thread(target=run_script_thread, args=(script, name, idx))
                t.start()
                response = {"status": "started", "step": step}
            else:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid pipeline step specifier"}).encode('utf-8'))
                return
                
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return

        if parsed_path == '/api/pipeline/clear_data':
            
            with lock:
                if pipeline_state["status"] == "running":
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Cannot clear data while a pipeline task is running."}).encode('utf-8'))
                    return
            
            import shutil
            raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
            data_dir = os.path.join(PIPELINE_DIR, "data")
            model_dir = os.path.join(PIPELINE_DIR, "model")
            
            logs = "=== Clearing Pipeline Data & Outputs ===\n"
            
            # 1. Clear raw/ files
            deleted_raw_count = 0
            if os.path.exists(raw_dir):
                for f in os.listdir(raw_dir):
                    if f.endswith('.csv'):
                        try:
                            os.remove(os.path.join(raw_dir, f))
                            deleted_raw_count += 1
                        except Exception:
                            pass
                logs += f"Deleted {deleted_raw_count} synthetic subject CSV files under data/raw/.\n"
                
            # 2. Clear processed.json & features.csv
            processed_file = os.path.join(data_dir, "processed.json")
            if os.path.exists(processed_file):
                try:
                    os.remove(processed_file)
                    logs += "Removed data/processed.json.\n"
                except Exception:
                    pass
                
            features_file = os.path.join(data_dir, "features.csv")
            if os.path.exists(features_file):
                try:
                    os.remove(features_file)
                    logs += "Removed data/features.csv.\n"
                except Exception:
                    pass
                
            # 3. Clear model directory files
            scaler_mean = os.path.join(model_dir, "scaler_mean.npy")
            if os.path.exists(scaler_mean):
                try:
                    os.remove(scaler_mean)
                    logs += "Removed model/scaler_mean.npy.\n"
                except Exception:
                    pass
                
            scaler_scale = os.path.join(model_dir, "scaler_scale.npy")
            if os.path.exists(scaler_scale):
                try:
                    os.remove(scaler_scale)
                    logs += "Removed model/scaler_scale.npy.\n"
                except Exception:
                    pass
                
            tflite_model = os.path.join(model_dir, "model.tflite")
            if os.path.exists(tflite_model):
                try:
                    os.remove(tflite_model)
                    logs += "Removed model/model.tflite.\n"
                except Exception:
                    pass
                
            saved_model_dir = os.path.join(model_dir, "saved_model")
            if os.path.exists(saved_model_dir):
                try:
                    shutil.rmtree(saved_model_dir)
                    logs += "Removed model/saved_model/ directory.\n"
                except Exception:
                    pass

            model_h5 = os.path.join(model_dir, "model.h5")
            if os.path.exists(model_h5):
                try:
                    os.remove(model_h5)
                    logs += "Removed model/model.h5.\n"
                except Exception:
                    pass

            report_file = os.path.join(model_dir, "training_report.json")
            if os.path.exists(report_file):
                try:
                    os.remove(report_file)
                    logs += "Removed model/training_report.json.\n"
                except Exception:
                    pass

            firmware_model = os.path.abspath(os.path.join(PIPELINE_DIR, "..", "firmware", "model", "model_data.h"))
            if os.path.exists(firmware_model):
                try:
                    os.remove(firmware_model)
                    logs += "Removed firmware model_data.h.\n"
                except Exception:
                    pass
                
            logs += "=== Cleanup Complete! ===\n"
            
            with lock:
                pipeline_state["status"] = "idle"
                pipeline_state["current_step"] = "Cleanup"
                pipeline_state["active_step_index"] = -1
                pipeline_state["logs"] = logs
                
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "cleared", "logs": logs}).encode('utf-8'))
            return

        if parsed_path == '/api/raw/label':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            filename = params.get("filename", "")
            new_label = params.get("new_label", "")
            
            filename = os.path.basename(filename)
            new_label = new_label.lower()
            
            if not filename or new_label not in ('normal', 'moderate', 'high'):
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid filename or label"}).encode('utf-8'))
                return
                
            raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
            src_path = os.path.join(raw_dir, filename)
            
            if not os.path.exists(src_path):
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "File not found"}).encode('utf-8'))
                return
                
            # Compute new filename
            base, ext = os.path.splitext(filename)
            parts = base.split("_")
            if len(parts) > 1:
                last_part = parts[-1].lower()
                is_known = False
                for l in ('normal', 'moderate', 'high'):
                    if last_part.startswith(l):
                        is_known = True
                        break
                if is_known:
                    parts[-1] = new_label
                    new_filename = "_".join(parts) + ext
                else:
                    new_filename = f"{base}_{new_label}{ext}"
            else:
                new_filename = f"{base}_{new_label}{ext}"
                
            dest_path = os.path.join(raw_dir, new_filename)
            
            try:
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
                
                os.rename(src_path, dest_path)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "new_filename": new_filename}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Rename failed: {str(e)}"}).encode('utf-8'))
            return

        if parsed_path == '/api/raw/delete':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            filename = params.get("filename", "")
            filename = os.path.basename(filename)
            
            if not filename:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid filename"}).encode('utf-8'))
                return
                
            raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
            filepath = os.path.join(raw_dir, filename)
            
            if not os.path.exists(filepath):
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "File not found"}).encode('utf-8'))
                return
                
            try:
                os.remove(filepath)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Delete failed: {str(e)}"}).encode('utf-8'))
            return

        self.send_error(404, f"API endpoint not found: {self.path}")

    def do_GET(self):
        from urllib.parse import urlparse
        parsed_path = urlparse(self.path).path

        # Redirect / to serve dashboard.html
        if parsed_path == '/' or parsed_path == '':
            self.path = '/dashboard.html'
            return super().do_GET()

        # Handle API endpoint for pipeline status
        if parsed_path == '/api/pipeline/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            with lock:
                response = {
                    "status": pipeline_state["status"],
                    "current_step": pipeline_state["current_step"],
                    "active_step_index": pipeline_state["active_step_index"],
                    "logs": pipeline_state["logs"]
                }
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return

        # Handle API endpoint for scaler statistics
        if parsed_path == '/api/scaler':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            scaler_mean_path = os.path.join(PIPELINE_DIR, "model", "scaler_mean.npy")
            scaler_scale_path = os.path.join(PIPELINE_DIR, "model", "scaler_scale.npy")
            
            response = {}
            if os.path.exists(scaler_mean_path) and os.path.exists(scaler_scale_path):
                try:
                    mean = np.load(scaler_mean_path)
                    scale = np.load(scaler_scale_path)
                    features = [
                        "PI_finger", "PI_toe", "PI_ratio", "PTT_ft",
                        "AI_finger", "AI_toe", "HRV_rmssd", "dicrotic_ratio",
                        "temp_finger", "temp_toe", "temp_diff",
                        "accel_std", "gyro_std",
                    ]
                    for i, feat in enumerate(features):
                        if i < len(mean) and i < len(scale):
                            response[feat] = {
                                "mean": float(mean[i]),
                                "scale": float(scale[i])
                            }
                except Exception as e:
                    response = {"error": f"Failed to load numpy files: {str(e)}"}
            else:
                response = {"error": "Scaler files not found. Run train_model.py first."}
                
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return

        # Handle API endpoint for pipeline config (reads constants from Python source)
        if parsed_path == '/api/config':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "data_preprocessing",
                    os.path.join(PIPELINE_DIR, "data_preprocessing.py")
                )
                dp = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(dp)
                fs          = int(dp.FS)
                bp_low      = float(dp.BANDPASS[0])
                bp_high     = float(dp.BANDPASS[1])
                win_sec     = int(dp.WINDOW_SECONDS)
                win_samples = int(dp.WINDOW_LEN)
                step_sec    = int(dp.STEP_SECONDS)
                overlap_pct = int((1 - step_sec / win_sec) * 100)
                config = {
                    "sample_rate_hz":   fs,
                    "bandpass_low_hz":  bp_low,
                    "bandpass_high_hz": bp_high,
                    "window_seconds":   win_sec,
                    "window_samples":   win_samples,
                    "step_seconds":     step_sec,
                    "overlap_pct":      overlap_pct,
                    "target_arch":      "ESP32-S3 (TensorFlow Lite Micro)"
                }
            except Exception as e:
                config = {"error": str(e)}
            self.wfile.write(json.dumps(config).encode('utf-8'))
            return

        # Handle API endpoint for raw dataset zip download
        if parsed_path == '/api/download/raw':
            import zipfile
            import io
            
            raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
            if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No synthetic raw data found. Generate synthetic data first."}).encode('utf-8'))
                return
                
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(raw_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            zipf.write(os.path.join(root, file), file)
                            
            memory_file.seek(0)
            self.send_response(200)
            self.send_header('Content-type', 'application/zip')
            self.send_header('Content-Disposition', 'attachment; filename="synthetic_raw_data.zip"')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(memory_file.read())
            return

        # Handle API endpoint to list raw data files
        if parsed_path == '/api/raw/list':
            raw_dir = os.path.join(PIPELINE_DIR, "data", "raw")
            os.makedirs(raw_dir, exist_ok=True)
            files_list = []
            for f in os.listdir(raw_dir):
                if f.endswith('.csv'):
                    filepath = os.path.join(raw_dir, f)
                    size = os.path.getsize(filepath)
                    mtime = os.path.getmtime(filepath)
                    from datetime import datetime
                    captured_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    
                    base, ext = os.path.splitext(f)
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
                        "timestamp": int(mtime)
                    })
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(files_list).encode('utf-8'))
            return

        # Handle API endpoint to fetch source code of pipeline scripts
        if parsed_path == '/api/code':
            from urllib.parse import parse_qs
            query = parse_qs(urlparse(self.path).query)
            filename = query.get("file", [None])[0]
            
            allowed_files = [
                "generate_synthetic_data.py",
                "data_preprocessing.py",
                "feature_extraction.py",
                "train_model.py",
                "convert_to_tflite.py"
            ]
            
            if not filename or filename not in allowed_files:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid or unauthorized file request"}).encode('utf-8'))
                return
                
            file_path = os.path.join(PIPELINE_DIR, filename)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.send_response(200)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
                return
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "File not found"}).encode('utf-8'))
                return

        # Handle API endpoint for model training report
        if parsed_path == '/api/model/report':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            import pandas as pd
            import tensorflow as tf
            
            report_path = os.path.join(PIPELINE_DIR, "model", "training_report.json")
            features_path = os.path.join(PIPELINE_DIR, "data", "features.csv")
            model_path = os.path.join(PIPELINE_DIR, "model", "model.h5")
            scaler_mean_path = os.path.join(PIPELINE_DIR, "model", "scaler_mean.npy")
            scaler_scale_path = os.path.join(PIPELINE_DIR, "model", "scaler_scale.npy")
            
            # Default fallback if files don't exist
            base_report = {}
            if os.path.exists(report_path):
                try:
                    with open(report_path, 'r') as f:
                        base_report = json.load(f)
                except Exception:
                    pass
            
            # Check if we have features and model to compute custom metrics
            if os.path.exists(features_path) and os.path.exists(model_path) and os.path.exists(scaler_mean_path) and os.path.exists(scaler_scale_path):
                try:
                    df = pd.read_csv(features_path)
                    # Include all subjects (do not filter out 'subj')
                    df_filtered = df
                    
                    if len(df_filtered) > 0:
                        # Load model and scaler parameters
                        model = tf.keras.models.load_model(model_path)
                        mean = np.load(scaler_mean_path)
                        scale = np.load(scaler_scale_path)
                        
                        # Use feature_cols saved in report, or fall back to full 13-feature list
                        feature_cols = base_report.get("feature_cols", [
                            "PI_finger", "PI_toe", "PI_ratio", "PTT_ft",
                            "AI_finger", "AI_toe", "HRV_rmssd", "dicrotic_ratio",
                            "temp_finger", "temp_toe", "temp_diff",
                            "accel_std", "gyro_std",
                        ])
                        # Only keep cols that are actually present in CSV
                        feature_cols = [c for c in feature_cols if c in df_filtered.columns]
                        X = df_filtered[feature_cols].values.astype(np.float32)
                        y = df_filtered["label"].values.astype(np.int32)
                        
                        X_scaled = (X - mean) / scale
                        probs = model.predict(X_scaled)
                        y_pred = np.argmax(probs, axis=1)
                        
                        accuracy = float(np.mean(y == y_pred))
                        
                        # Confusion Matrix
                        cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                        for true_lbl, pred_lbl in zip(y, y_pred):
                            if 0 <= true_lbl < 3 and 0 <= pred_lbl < 3:
                                cm[int(true_lbl)][int(pred_lbl)] += 1
                                
                        # Per Class Metrics
                        per_class = {}
                        class_names = ["Normal", "Moderate", "High"]
                        for i, name in enumerate(class_names):
                            tp = int(np.sum((y == i) & (y_pred == i)))
                            fp = int(np.sum((y != i) & (y_pred == i)))
                            fn = int(np.sum((y == i) & (y_pred != i)))
                            support = int(np.sum(y == i))
                            
                            precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
                            recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
                            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                            
                            per_class[name] = {
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "support": support
                            }
                            
                        # Build custom report
                        custom_report = {
                            "accuracy": accuracy,
                            "confusion_matrix": cm,
                            "per_class": per_class,
                            "train_samples": len(df_filtered),
                            "test_samples": 0,
                            "total_samples": len(df_filtered),
                            "total_subjects": len(df_filtered['subject_id'].unique()),
                            "epochs_run": base_report.get("epochs_run", 200)
                        }
                        self.wfile.write(json.dumps(custom_report).encode('utf-8'))
                        return
                except Exception as e:
                    # In case of any error running prediction, fallback to base_report or zero_report
                    print("[Server Error] Failed to compute custom report:", e)
                    pass
            
            # Fallback to saved base_report if available
            if base_report and "accuracy" in base_report:
                self.wfile.write(json.dumps(base_report).encode('utf-8'))
                return
            
            # If no custom data or failed, return a zero metrics report for clean display
            zero_report = {
                "accuracy": 0.0,
                "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                "per_class": {
                    "Normal": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
                    "Moderate": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
                    "High": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                },
                "train_samples": 0,
                "test_samples": 0,
                "total_samples": 0,
                "total_subjects": 0,
                "epochs_run": base_report.get("epochs_run", 0)
            }
            self.wfile.write(json.dumps(zero_report).encode('utf-8'))
            return

        # Handle serving files from pipeline data or model folders
        if parsed_path.startswith('/data/') or parsed_path.startswith('/model/'):
            # Strip leading slash and route to pipeline directory
            local_path = os.path.join(PIPELINE_DIR, parsed_path.lstrip('/'))
            if os.path.exists(local_path) and os.path.isfile(local_path):
                self.send_response(200)
                # Set mime types and attachment disposition
                filename = os.path.basename(local_path)
                if local_path.endswith('.csv'):
                    self.send_header('Content-type', 'text/csv')
                    self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
                elif local_path.endswith('.json'):
                    self.send_header('Content-type', 'application/json')
                else:
                    self.send_header('Content-type', 'application/octet-stream')
                    self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                with open(local_path, 'rb') as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, f"File not found: {self.path}")
                return

        # Serve static assets from the dashboard folder
        local_path = os.path.join(SCRIPT_DIR, parsed_path.lstrip('/'))
        if os.path.exists(local_path) and os.path.isfile(local_path):
            self.path = parsed_path
            return super().do_GET()
        
        self.send_error(404, f"File not found: {self.path}")

def run_server():
    # Change working directory to ensure SimpleHTTPRequestHandler operates inside the dashboard folder
    os.chdir(SCRIPT_DIR)
    
    # Allow port reuse to avoid 'address already in use' errors
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), DashboardRequestHandler) as httpd:
        print(f"==================================================")
        print(f"[*] PVD PPG Pipeline Dashboard starting at:")
        print(f"-> http://localhost:{PORT}/")
        print(f"==================================================")
        print("Press Ctrl+C to stop the server.")
        
        # Auto-open browser
        webbrowser.open(f"http://localhost:{PORT}/")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

if __name__ == "__main__":
    run_server()
