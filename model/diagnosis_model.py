import numpy as np
import joblib
import os

MODEL_PATH = "model_stress.pkl"

print("==================================================")
print("🔍 DIAGNOSTIC (.NPY): MEMBONGKAR MISTERI PREDIKSI 0")
print("==================================================")

# 1. Load Model XGBoost
if not os.path.exists(MODEL_PATH):
    print(f"❌ File '{MODEL_PATH}' tidak ditemukan di folder ini!")
    exit()

model = joblib.load(MODEL_PATH)
print("✅ Model Berhasil Dimuat.")

# 2. Cari File .npy di Folder
files_in_dir = os.listdir('.')
npy_files = [f for f in files_in_dir if f.endswith('.npy')]
print(f"📁 File .npy yang terdeteksi di folder: {npy_files}")

# Deteksi file test otomatis
x_test_file = "x_test.npy" if "x_test.npy" in npy_files else (npy_files[0] if len(npy_files) > 0 else None)
y_test_file = "y_test.npy" if "y_test.npy" in npy_files else (npy_files[1] if len(npy_files) > 1 else None)

if not x_test_file:
    print("❌ Tidak ada file .npy yang bisa diperiksa.")
    exit()

# 3. Load Data Test .npy Asli
try:
    print(f"\n⏳ Mencoba memuat {x_test_file}...")
    X_test = np.load(x_test_file)
    print(f"✅ Berhasil memuat X_test dengan bentuk matriks: {X_test.shape}")
    
    # Jalankan prediksi massal pada seluruh data test asli
    preds = model.predict(X_test)
    
    # Hitung sebaran hasil prediksi model pada data asli
    unique, counts = np.unique(preds, return_counts=True)
    sebaran_prediksi = dict(zip(unique, counts))
    print(f"📊 Sebaran Hasil Prediksi Model pada Seluruh Data Test Asli: {sebaran_prediksi}")

    # 4. Ambil Contoh Baris Data Asli dari Dataset
    print("\n🔬 MEMBEDAH CONTOH DATA BARIS PERTAMA DARI JALUR DATASET:")
    row_0 = X_test[0:1] # Ambil baris ke-0
    pred_0 = preds[0]
    proba_0 = model.predict_proba(row_0)[0] if hasattr(model, "predict_proba") else "N/A"
    
    print(f"👉 Model Memprediksi Baris Ini Sebagai Kelas: {pred_0}")
    print(f"   Probabilitas [Low, Moderate, High]: {proba_0}")
    print(f"   Isi Array Mentah Baris Ini (Total {X_test.shape[1]} kolom):")
    print(row_0[0])
    
    # Cek apakah datanya sudah di-scale atau belum
    if np.any(row_0 < 0) or np.any((row_0 > 0) & (row_0 < 1) & (row_0 != 0)):
        print("\n🚨 KESIMPULAN AWAL: Angka di atas berupa desimal/minus. Data kamu SUDAH DI-SCALE saat training!")
    else:
        print("\n🚨 KESIMPULAN AWAL: Angka di atas berupa bilangan bulat utuh. Data kamu MASIH MENTAH saat training!")

except Exception as e:
    print(f"❌ Gagal membaca atau memproses file .npy: {e}")

print("==================================================")