import joblib
import numpy as np
import os

<<<<<<< HEAD
MODEL_PATH = "model_stress.pkl"
=======
MODEL_PATH = "model_stress2.pkl"
SCALER_PATH = "scaler2.pkl"
>>>>>>> fresh-main

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: File '{MODEL_PATH}' tidak ditemukan di folder ini")
    exit()

try:
    model = joblib.load(MODEL_PATH)
<<<<<<< HEAD
=======
    scaler = joblib.load(SCALER_PATH)
>>>>>>> fresh-main
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
<<<<<<< HEAD
    print("WARNING: Model tidak memiliki feature_importances_. Apakah ini benar XGBoost?")
=======
    print("WARNING: Model tidak memiliki feature_importances_.")
>>>>>>> fresh-main


print("\n--- Test Case 1: Simulasi Input Stres Ekstrem Tinggi ---")
input_stres_tinggi = np.array([[
<<<<<<< HEAD
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
=======
    21, 0, 3, 12.0, 10, 2.1, 10, 10, 3, 0, 1, 14.0, 12.0, 10, 10, 10 
]], dtype=np.float32)

input_stres_tinggi_scaled = scaler.transform(input_stres_tinggi)

pred_1 = model.predict(input_stres_tinggi_scaled)[0]
print(f"Hasil Prediksi Kasus Stres Tinggi: {pred_1}")

if hasattr(model, "predict_proba"):
    proba_1 = model.predict_proba(input_stres_tinggi_scaled)[0]
>>>>>>> fresh-main
    print(f"Probabilitas Tiap Kelas [Low, Moderate, High]: {proba_1}")


print("\n--- Test Case 2: Simulasi Input Stres Rendah ---")
input_stres_rendah = np.array([[
<<<<<<< HEAD
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
=======
    20, 1, 1, 4.0, 1, 3.9, 1, 1, 8, 5, 10, 2.0, 2.0, 1, 1, 1  
]], dtype=np.float32)

input_stres_rendah_scaled = scaler.transform(input_stres_rendah)

pred_2 = model.predict(input_stres_rendah_scaled)[0]
print(f"Hasil Prediksi Kasus Stres Rendah: {pred_2}")

if hasattr(model, "predict_proba"):
    proba_2 = model.predict_proba(input_stres_rendah_scaled)[0]
    print(f"Probabilitas Tiap Kelas [Low, Moderate, High]: {proba_2}")

print("\n--- Test Case 3: Simulasi Input Stres Tingkat Sedang (Moderate) ---")

input_stres_moderate = np.array([[ 20, 0, 2, 6, 6, 3.20, 5, 4, 6, 5, 5, 7, 3, 3, 3, 3]], dtype=np.float32)

input_stres_moderate_scaled = scaler.transform(input_stres_moderate)


pred_3 = model.predict(input_stres_moderate_scaled)[0]
print(f"Hasil Prediksi Kasus Stres Moderate: {pred_3}")
if hasattr(model, "predict_proba"):
    proba_3 = model.predict_proba(input_stres_moderate_scaled)[0]
    print(f"Probabilitas Tiap Kelas [Low, Moderate, High]: {proba_3}")


print("==================================================")
>>>>>>> fresh-main
