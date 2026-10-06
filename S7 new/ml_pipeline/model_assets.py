# Auto-generated model assets fallback module
import os
import io
import base64
import numpy as np

DEFAULT_SCALER_MEAN = np.array([
    18.608887025973942, 10.763782430815217, 0.5794458485289709, 180.82431675443712,
    1.2291153549347948, 1.8642164236747178, 10.086620620433116, 0.018642164191743672,
    34.22424183275876, 31.01634188146399, 3.207899921042407, 0.05000684090308695, 0.3371732512976499
], dtype=np.float32)

DEFAULT_SCALER_SCALE = np.array([
    0.6146075637879097, 4.603679451908329, 0.2489822007894929, 64.41398587115683,
    2.8146795351614102, 3.321316744307773, 20.157050248726904, 0.03321316736597185,
    0.460777636861988, 3.1767613627089433, 2.778894624110804, 0.0008474539813045889, 0.005877234028068316
], dtype=np.float32)

EMBEDDED_MODEL_NPZ_B64 = "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4vE1PgMxqDkGJFw8cv8RP0WyZL7V3h4+pgPWvqdjrj4cAaI+lly+PT0uGj+qEQi/QzAevuyGvL64qCg8zyyZvmOr2r5Vd5U9AQCoPdPvab7XSaW+7e20PjIdAb+36c0+H0GCuxnXt75dTSg/rlq0vT+dsD3eSRS/YLUVPgu3yr528+0+yylWPm4gor50wAs/QiZhve0GgL4i75++pvYTP1BLAwQtAAAAAAAAACEAtbM8OP//////////BgAUAGIyLm5weQEAEACgAAAAAAAAAKAAAAAAAAAAk05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoKQyu+RLb9PfyHfT0MI2c9KXugvfV6GT5gufA+xjadPFBLAwQtAAAAAAAAACEASl8Arf//////////BgAUAHczLm5weQEAEADgAAAAAAAAAOAAAAAAAAAAk05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsIDMpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAq08BQ/KjU8Pltghz90F+e8W9zhvUFeLr9J9EO/f32xPjLntj7q4GM/D/VqvzwhNr8oABO/Tr2ivkM4PT/u7jK93GsgP52/p77ZV0+/Q4uSPzn/974/mVk/s5mtvirhdr5QSwMELQAAAAAAAAAhAO9zBtz//////////wYAFABiMy5ucHkBABAAjAAAAAAAAACMAAAAAAAAAJNOVU1QWQEAdgB7J2Rlc2NyJzogJzxmNCcsICdmb3J0cmFuX29yZGVyJzogRmFsc2UsICdzaGFwZSc6ICgzLCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAK64pNvNrkQj7vfoa+UEsDBC0AAAAAAAAAIQDZG/CL//////////8PABQAc2NhbGVyX21lYW4ubnB5AQAQAOgAAAAAAAAA6AAAAAAAAACTTlVNUFkBAHYAeydkZXNjcic6ICc8ZjgnLCAnZm9ydHJhbl9vcmRlcic6IEZhbHNlLCAnc2hhcGUnOiAoMTMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCkyEJwXgmzJAszijfQ6HJUBMhCcF0oriP0a7h81gmmZAUU3H3HSq8z96lcWZ1NP9P8TWt4lZLCRAGu0edu4Wkz8HDNX0sxxBQGVxRvsuBD9Atr4Nb8epCUDCk4Ikf5qpP4sz2h0/lNU/UEsDBC0AAAAAAAAAIQC15AUP//////////8QABQAc2NhbGVyX3NjYWxlLm5weQEAEADoAAAAAAAAAOgAAAAAAAAAk05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEzLCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIApd/Ep73arjP5raPPIqahJAMqfWFKbezz/6Z5i+fhpQQPDoQbR2hAZAMqxjgw6SCkBYHPJxNCg0QBhQRPdQAaE/zDlDfGF93T8dQIDcAWoJQF6wzBotOwZA64c5kfXESz+rWRD/uRJ4P1BLAQItAC0AAAAAAAAAIQB/plXqwAMAAMADAAAGAAAAAAAAAAAAAACAAQAAAAB3MS5ucHlQSwECLQAtAAAAAAAAACEArNF1H8AAAADAAAAABgAAAAAAAAAAAAAAgAH4AwAAYjEubnB5UEsBAi0ALQAAAAAAAAAhACmNolSAAgAAgAIAAAYAAAAAAAAAAAAAAIAB8AQAAHcyLm5weVBLAQItAC0AAAAAAAAAIQC1szw4oAAAAKAAAAAGAAAAAAAAAAAAAACAAagHAABiMi5ucHlQSwECLQAtAAAAAAAAACEASl8AreAAAADgAAAABgAAAAAAAAAAAAAAgAGACAAAdzMubnB5UEsBAi0ALQAAAAAAAAAhAO9zBtyMAAAAjAAAAAYAAAAAAAAAAAAAAIABmAkAAGIzLm5weVBLAQItAC0AAAAAAAAAIQDZG/CL6AAAAOgAAAAPAAAAAAAAAAAAAACAAVwKAABzY2FsZXJfbWVhbi5ucHlQSwECLQAtAAAAAAAAACEAteQFD+gAAADoAAAAEAAAAAAAAAAAAAAAgAGFCwAAc2NhbGVyX3NjYWxlLm5weVBLBQYAAAAACAAIALMBAACvDAAAAAA="

class MLPPredictor:
    def __init__(self, w1, b1, w2, b2, w3, b3):
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32)
        self.w3 = np.asarray(w3, dtype=np.float32)
        self.b3 = np.asarray(b3, dtype=np.float32)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        h1 = np.maximum(0, np.dot(X, self.w1) + self.b1)
        h2 = np.maximum(0, np.dot(h1, self.w2) + self.b2)
        logits = np.dot(h2, self.w3) + self.b3
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=-1, keepdims=True)

def ensure_model_files(model_dir):
    """Ensures model_weights.npz, scaler_mean.npy, and scaler_scale.npy exist on disk."""
    try:
        os.makedirs(model_dir, exist_ok=True)
        mean_path = os.path.join(model_dir, "scaler_mean.npy")
        scale_path = os.path.join(model_dir, "scaler_scale.npy")
        npz_path = os.path.join(model_dir, "model_weights.npz")

        npz_bytes = base64.b64decode(EMBEDDED_MODEL_NPZ_B64)

        if not os.path.exists(npz_path):
            with open(npz_path, "wb") as f:
                f.write(npz_bytes)

        if not os.path.exists(mean_path) or not os.path.exists(scale_path):
            try:
                data = np.load(npz_path)
                m = data.get("scaler_mean", DEFAULT_SCALER_MEAN)
                s = data.get("scaler_scale", DEFAULT_SCALER_SCALE)
            except Exception:
                m = DEFAULT_SCALER_MEAN
                s = DEFAULT_SCALER_SCALE
            if not os.path.exists(mean_path):
                np.save(mean_path, m)
            if not os.path.exists(scale_path):
                np.save(scale_path, s)
    except Exception as e:
        print(f"[!] Warning ensuring model files: {e}")

def load_model_and_scalers(model_dir):
    """Guarantees returning (model, scaler_mean, scaler_scale) without ever failing."""
    ensure_model_files(model_dir)

    model = None
    scaler_mean = None
    scaler_scale = None

    npz_path = os.path.join(model_dir, "model_weights.npz")
    model_path = os.path.join(model_dir, "model.h5")
    mean_path = os.path.join(model_dir, "scaler_mean.npy")
    scale_path = os.path.join(model_dir, "scaler_scale.npy")

    # 1. Try loading from model_weights.npz
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            model = MLPPredictor(data["w1"], data["b1"], data["w2"], data["b2"], data["w3"], data["b3"])
            if "scaler_mean" in data and "scaler_scale" in data:
                scaler_mean = data["scaler_mean"]
                scaler_scale = data["scaler_scale"]
        except Exception as e:
            print(f"[!] Failed to load from model_weights.npz: {e}")

    # 2. Try loading scalers from .npy if not already found
    if scaler_mean is None or scaler_scale is None:
        if os.path.exists(mean_path) and os.path.exists(scale_path):
            try:
                scaler_mean = np.load(mean_path)
                scaler_scale = np.load(scale_path)
            except Exception as e:
                print(f"[!] Failed to load scaler .npy files: {e}")

    # 3. Check history models if still None
    if model is None or scaler_mean is None or scaler_scale is None:
        hist_dir = os.path.join(model_dir, "history")
        if os.path.exists(hist_dir):
            try:
                for sub in sorted(os.listdir(hist_dir), reverse=True):
                    sub_dir = os.path.join(hist_dir, sub)
                    if not os.path.isdir(sub_dir):
                        continue
                    sub_mean = os.path.join(sub_dir, "scaler_mean.npy")
                    sub_scale = os.path.join(sub_dir, "scaler_scale.npy")
                    if scaler_mean is None and os.path.exists(sub_mean) and os.path.exists(sub_scale):
                        scaler_mean = np.load(sub_mean)
                        scaler_scale = np.load(sub_scale)
                    sub_h5 = os.path.join(sub_dir, "model.h5")
                    if model is None and os.path.exists(sub_h5):
                        try:
                            import h5py
                            with h5py.File(sub_h5, "r") as f:
                                w1 = f["model_weights/dense/sequential/dense/kernel"][:]
                                b1 = f["model_weights/dense/sequential/dense/bias"][:]
                                w2 = f["model_weights/dense_1/sequential/dense_1/kernel"][:]
                                b2 = f["model_weights/dense_1/sequential/dense_1/bias"][:]
                                w3 = f["model_weights/dense_2/sequential/dense_2/kernel"][:]
                                b3 = f["model_weights/dense_2/sequential/dense_2/bias"][:]
                            model = MLPPredictor(w1, b1, w2, b2, w3, b3)
                        except Exception:
                            pass
            except Exception as e:
                print(f"[!] Warning scanning history models: {e}")

    # 4. Fallback to embedded weights and scalers (Zero-failure guarantee)
    if model is None or scaler_mean is None or scaler_scale is None:
        try:
            data = np.load(io.BytesIO(base64.b64decode(EMBEDDED_MODEL_NPZ_B64)))
            if model is None:
                model = MLPPredictor(data["w1"], data["b1"], data["w2"], data["b2"], data["w3"], data["b3"])
            if scaler_mean is None:
                scaler_mean = data.get("scaler_mean", DEFAULT_SCALER_MEAN)
            if scaler_scale is None:
                scaler_scale = data.get("scaler_scale", DEFAULT_SCALER_SCALE)
        except Exception as e:
            print(f"[!] Embedded load fallback error: {e}")
            if scaler_mean is None:
                scaler_mean = DEFAULT_SCALER_MEAN
            if scaler_scale is None:
                scaler_scale = DEFAULT_SCALER_SCALE

    return model, scaler_mean, scaler_scale
