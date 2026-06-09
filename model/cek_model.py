import joblib
import numpy as np
import os

MODEL_PATH = "model_stress.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: File '{MODEL_PATH}' tidak ditemukan di folder ini")
    exit()

try:
    model = joblib.load(MODEL_PATH)
    print(" Model Berhasil Dimuat ke Memori.")
except Exception as e:
    print(f"ERROR saat memuat model: {e}")
    exit()


print("\n--- Atribut Model ---")
if hasattr(model, "classes_"):
    print(f"Target Classes (Kelas Target): {model.classes_}")
else:
    print("Model tidak memiliki atribut classes_")

if hasattr(model, "feature_importances_"):
    print("Feature Importances terdeteksi.")
    print(f"Jumlah Fitur yang Diharapkan Model: {len(model.feature_importances_)} kolom")
else:
    print("WARNING: Model tidak memiliki feature_importances_. Apakah ini benar XGBoost?")


print("\n--- Test Case 1: Simulasi Input Stres Ekstrem Tinggi ---")
input_stres_tinggi = np.array([[
    21,    
    0,     
    3,    
    12.0,  
    10,   
    2.1,   
    10,    
    10,   
    3,   
    0,    
    1,  
    14.0, 
    12.0, 
    10, 
    10,
    10,
    10  
]], dtype=np.float32)

pred_1 = model.predict(input_stres_tinggi)[0]
print(f"Hasil Prediksi Kasus Stres Tinggi: {pred_1}")

if hasattr(model, "predict_proba"):
    proba_1 = model.predict_proba(input_stres_tinggi)[0]
    print(f"Probabilitas Tiap Kelas [Low, Moderate, High]: {proba_1}")


print("\n--- Test Case 2: Simulasi Input Stres Rendah ---")
input_stres_rendah = np.array([[
    20,   
    1,     
    1,    
    4.0,  
    1,   
    3.9, 
    1, 
    1,    
    8,   
    5,    
    10,
    2.0,  
    2.0, 
    1, 
    1,  
    1,    
    1   
]], dtype=np.float32)

pred_2 = model.predict(input_stres_rendah)[0]
print(f"Hasil Prediksi Kasus Stres Rendah: {pred_2}")

if hasattr(model, "predict_proba"):
    proba_2 = model.predict_proba(input_stres_rendah)[0]
    print(f"Probabilitas Tiap Kelas [Low, Moderate, High]: {proba_2}")

print("==================================================")