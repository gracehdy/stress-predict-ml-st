import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="StressPredict API")

model = joblib.load("model_stress3.pkl")
scaler = joblib.load("scaler2.pkl")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

NAMA_FITUR = [
    'Faktor Umur', 'Faktor Gender', 'Tahun Angkatan Kuliah', 'Durasi Waktu Belajar Harian', 
    'Beban Tekanan Ujian', 'Performa Akademik / IPK', 'Tingkat Kecemasan (Anxiety)', 
    'Kondisi Suasana Hati (Mood/Depresi)', 'Kualitas & Durasi Istirahat/Tidur', 
    'Rendahnya Aktivitas Fisik/Olahraga', 'Keterbatasan Dukungan Sosial', 
    'Durasi Paparan Layar Gadget (Screen Time)', 'Pola Konsumsi Internet Harian', 
    'Tekanan Finansial/Keuangan Mahasiswa', 'Tuntutan/Ekspektasi dari Keluarga', 
    'Tingkat Kejenuhan Akademik (Burnout)'
]

class KuesionerInput(BaseModel):
    umur: int
    gender: str
    tahun_akademik: str
    jam_belajar: int
    ipk: float
    tekanan_ujian: int
    ekspektasi_keluarga: int
    anxiety_score: int
    depression_score: int
    jam_tidur: int
    aktivitas_fisik: int
    screen_time: int
    internet_usage: int
    social_support: int
    financial_stress: int
    burnout_score: int
    
class RecommendInput(BaseModel):
    status_stres: str
    faktor_dominan: list[str]

@app.post("/predict")
def predict_stress(data: KuesionerInput):
    gender_encoded = 1 if data.gender == "Laki-laki" else 0
    
    academic_year_encoded = 1
    if "2" in data.tahun_akademik: academic_year_encoded = 2
    elif "3" in data.tahun_akademik: academic_year_encoded = 3
    elif "4" in data.tahun_akademik: academic_year_encoded = 4

    academic_performance_scaled = min(100.0, max(1.0, data.ipk * 25.0))
    raw_features = np.array([[
        data.umur, gender_encoded, academic_year_encoded, data.jam_belajar, 
        data.tekanan_ujian, academic_performance_scaled, data.anxiety_score, 
        data.depression_score, data.jam_tidur, data.aktivitas_fisik, 
        data.social_support, data.screen_time, data.internet_usage,
        data.financial_stress, data.ekspektasi_keluarga, data.burnout_score
    ]], dtype=np.float32)

    scaled_features = scaler.transform(raw_features)
    prediction = int(model.predict(scaled_features)[0])
    categories = {0: "Stres Rendah (Low)", 1: "Stres Sedang (Moderate)", 2: "Stres Tinggi (High)"}
    status_terprediksi = categories.get(prediction, "Stres Sedang (Moderate)")
    score_display = 7.0
    if hasattr(model, "predict_proba"):
        score_display = float(round(model.predict_proba(scaled_features)[0][prediction] * 10, 1))

    try:
        importances = model.feature_importances_
        z_scores = scaled_features[0].copy()
        for idx in [8, 10, 16, 5]: 
            z_scores[idx] = -z_scores[idx]
            
        kontribusi_fitur = np.maximum(0, z_scores) * importances
        indeks_teratas = np.argsort(kontribusi_fitur)[::-1][:3]
        
        faktor_dominan = [str(NAMA_FITUR[idx]) for idx in indeks_teratas if kontribusi_fitur[idx] > 0]
        if not faktor_dominan:
            faktor_dominan = ["Beban Tekanan Ujian"]
    except AttributeError:
        faktor_dominan = ["Beban Tekanan Ujian", "Tingkat Kecemasan (Anxiety)"]
    return {
        "status": status_terprediksi,
        "confidence_score": score_display,
        "faktor_dominan": faktor_dominan
    }
    
@app.post("/recommend")
def get_recommendation(data: RecommendInput):
    faktor_teks = ", ".join(data.faktor_dominan)
    prompt = f"Saya seorang mahasiswa dengan tingkat stres '{data.status_stres}', dan faktor pemicu utamanya adalah: {faktor_teks}. Berikan 3 saran singkat, praktis, dan suportif dalam bentuk poin-poin (bullet points) untuk mengatasi hal ini."
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=250
        )
        return {"rekomendasi_ai": chat_completion.choices[0].message.content}
    except Exception as e:
        return {"rekomendasi_ai": """Sistem AI sedang sibuk. Atur jadwal tidur malam minimal 7 jam. 
Gunakan teknik Pomodoro. 
Diskusikan beban akademik dengan konselor kampus."""}