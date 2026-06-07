import numpy as np
import time
import pandas as pd
import joblib

model = joblib.load("model_stress.pkl")
scaler = joblib.load("scaler.pkl")

def measure_p90_custom(scaler, model, df, n_runs=100):
    """Mengukur latensi p50, p90, p99, mean, dan max untuk proses prediksi."""
    
    def full_pipeline_predict(data):
        scaled = scaler.transform(data)
        return model.predict(scaled)
    full_pipeline_predict(df)

    durations_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        full_pipeline_predict(df)
        durations_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "p50_ms": float(np.percentile(durations_ms, 50)),
        "p90_ms": float(np.percentile(durations_ms, 90)),
        "p99_ms": float(np.percentile(durations_ms, 99)),
        "mean_ms": float(np.mean(durations_ms)),
        "max_ms": float(np.max(durations_ms)),
    }