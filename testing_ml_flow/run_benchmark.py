import pandas as pd
from benchmark import measure_p90_custom, scaler, model

NAMA_FITUR = [
    'Faktor Umur', 'Faktor Gender', 'Tahun Angkatan Kuliah', 'Durasi Waktu Belajar Harian', 
    'Beban Tekanan Ujian', 'Performa Akademik / IPK', 'Tingkat Kecemasan (Anxiety)', 
    'Kondisi Suasana Hati (Mood/Depresi)', 'Kualitas & Durasi Istirahat/Tidur', 
    'Rendahnya Aktivitas Fisik/Olahraga', 'Keterbatasan Dukungan Sosial', 
    'Durasi Paparan Layar Gadget (Screen Time)', 'Pola Konsumsi Internet Harian', 
    'Tekanan Finansial/Keuangan Mahasiswa', 'Tuntutan/Ekspektasi dari Keluarga', 
    'Tingkat Kejenuhan Akademik (Burnout)'  
]

dummy_data = pd.DataFrame([[
    20, 1, 3, 5, 8, 3.5, 7, 6, 6, 2, 5, 8, 9, 4, 7, 6
]], columns=NAMA_FITUR)

stats = measure_p90_custom(scaler, model, dummy_data)

print("\n--- Hasil Benchmark Latensi (ms) ---")
print(f"P50 (Median) : {stats['p50_ms']:.4f} ms")
print(f"P90 (Target) : {stats['p90_ms']:.4f} ms")
print(f"P99          : {stats['p99_ms']:.4f} ms")
print(f"Rata-rata    : {stats['mean_ms']:.4f} ms")
print(f"Maksimum     : {stats['max_ms']:.4f} ms")