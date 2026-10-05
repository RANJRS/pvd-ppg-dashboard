import http.server
import socketserver
import os
import json
import webbrowser
import numpy as np
import subprocess
import threading
import base64

PORT = 8005
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

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
                        "AI_finger", "AI_toe", "HRV_rmssd", "dicrotic_ratio"
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
            self.send_header('Content-Disposition', 'attachment; filename=synthetic_raw_data.zip')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(memory_file.read())
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
            report_path = os.path.join(PIPELINE_DIR, "model", "training_report.json")
            if os.path.exists(report_path):
                try:
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                except Exception as e:
                    report = {"error": str(e)}
            else:
                report = {"error": "training_report.json not found. Run train_model.py first."}
            self.wfile.write(json.dumps(report).encode('utf-8'))
            return

        # Handle serving files from pipeline data or model folders
        if parsed_path.startswith('/data/') or parsed_path.startswith('/model/'):
            # Strip leading slash and route to pipeline directory
            local_path = os.path.join(PIPELINE_DIR, parsed_path.lstrip('/'))
            if os.path.exists(local_path) and os.path.isfile(local_path):
                self.send_response(200)
                # Set mime types
                if local_path.endswith('.csv'):
                    self.send_header('Content-type', 'text/csv')
                elif local_path.endswith('.json'):
                    self.send_header('Content-type', 'application/json')
                else:
                    self.send_header('Content-type', 'application/octet-stream')
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
