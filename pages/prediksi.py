import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="StressPredict - Form Kuesioner", layout="centered")

st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap">
    
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapseButton"], .st-emotion-cache-1wbqy5l {
            display: none !important;
            width: 0px !important;
    }
    .block-container {
        padding-top: 0rem !important;
        max-width: 800px !important;
    }
    
   * {font-family: 'Poppins', sans-serif !important;}
    [class*="st-emotion-cache"] i, 
        [data-testid="stSidebarCollapseButton"] i,
        .material-icons,
        [id*="Icon"] {
            font-family: inherit !important;
    }
    
    .stApp {
        background-color: #E2F0ED !important; 
    }
    
    [data-testid="stHeader"], [data-testid="stSidebar"] {
        background-color: transparent !important;
    }

    div[data-testid="stForm"] { 
        border-radius: 24px; 
        border: none !important; 
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.04); 
        padding: 50px; 
        background-color: #FFFFFF !important; 
        margin-top: 20px;
        margin-bottom: 40px;
    }
    

    .main-title { font-size: 2.4rem; font-weight: 800; color: #1F2937; margin-bottom: 8px; }
    .subtitle { font-size: 0.95rem; color: #6B7280; margin-bottom: 30px; font-weight: 400; line-height: 1.5; }
    

    div[data-testid="stForm"] label p {
        color: #4B5563 !important; 
        font-weight: 600 !important; 
        font-size: 0.75rem !important;
    }

    .section-title { font-size: 1.15rem; font-weight: 700; margin-top: 25px; margin-bottom: 15px; border-bottom: 2px solid #F3F4F6; padding-bottom: 5px; }
    .text-blue { color: #80A1BA; } 
    .text-mint { color: #91C4C3; } 
    .text-green { color: #87b390; } 
    .text-purple { color: #8B8EAD; }
    

    div[data-testid="stForm"] input, div[data-testid="stForm"] select {
        border: 1px solid #E5E7EB !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    
    div[data-testid="stForm"] button[type="submit"] {
        background-color: #91C4C3 !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(145, 196, 195, 0.3);
        transition: all 0.2s ease;
    }
    div[data-testid="stForm"] button[type="submit"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(145, 196, 195, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

def render_navbar():
    st.markdown("""
        <style>
        [data-testid="stSidebar"], [data-testid="stSidebarCollapseButton"], .st-emotion-cache-1wbqy5l {
            display: none !important;
            width: 0px !important;
        }
        
        .block-container {
            padding-top: 0rem !important;
            max-width: 800px !important; /* Memastikan lebar card di tengah proporsional */
        }
        
        .vue-navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #FFFFFF;
            padding: 18px 40px;
            border-radius: 0px 0px 20px 20px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04);
            margin-bottom: 35px;
            width: 100%;
            box-sizing: border-box;
        }
        

        .nav-brand {
            font-size: 1.35rem;
            font-weight: 800;
            color: #91C4C3;
            letter-spacing: -0.5px;
        }
        

        .nav-links {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        

        .nav-item-link {
            color: #4B5563 !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            padding: 8px 16px !important;
            text-decoration: none !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
            display: inline-block !important;
        }
       
        .nav-item-link:hover {
            color: #91C4C3 !important;
            background-color: #F3F4F6 !important;
        }
        

        .nav-item-disabled {
            color: #D1D5DB !important;
            cursor: not-allowed !important;
            pointer-events: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    is_disabled_result = "input_data" not in st.session_state
    
    disabled_class = "nav-item-disabled" if is_disabled_result else ""

    st.markdown(f"""
        <div class="vue-navbar">
            <div class="nav-brand">StressPredict</div>
            <div class="nav-links">
                <a href="?page=home" target="_self" class="nav-item-link">Beranda</a>
                <a href="?page=prediksi" target="_self" class="nav-item-link">Prediksi</a>
                <a href="?page=hasil" target="_self" class="nav-item-link {disabled_class}">Hasil</a>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    query_params = st.query_params
    if "page" in query_params:
        target = query_params["page"]
        st.query_params.clear() 
        
        if target == "home":
            st.switch_page("app.py")
        elif target == "prediksi":
            st.switch_page("pages/prediksi.py")
        elif target == "hasil" and not is_disabled_result:
            st.switch_page("pages/hasilprediksi.py")



<<<<<<< HEAD
MODEL_PATH = "model_stress.pkl"
SCALER_PATH = "scaler.pkl"  
=======
MODEL_PATH = "model_stress2.pkl"
SCALER_PATH = "scaler2.pkl"  
>>>>>>> fresh-main

@st.cache_resource
def load_ml_components():
    m = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    s = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return m, s

model, scaler = load_ml_components()

if model is None or scaler is None:
    st.error("Komponen Model (.pkl) gagal dimuat.")
    st.stop()
    
render_navbar()

with st.form("kuesioner_form"):

    st.markdown("<div class='main-title'>Cari Prediksi Tingkat Stress</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Masukkan kondisi akademik dan gaya hidup Anda untuk mengetahui estimasi tingkat stress!</div>", unsafe_allow_html=True)
    
    
    st.markdown("<div class='section-title text-blue'>1. Informasi Dasar</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2) 
    with col1: 
        umur = st.number_input("Umur (15-50 tahun)", min_value=15, max_value=50, value=None, placeholder=None)
        tahun_akademik = st.selectbox("Tahun Akademik", ["Tahun 1", "Tahun 2", "Tahun 3", "Tahun 4"], index=None, placeholder=None)
    with col2: 
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], index=None, placeholder=None)

    st.markdown("<div class='section-title text-mint'>2. Aktivitas Akademik</div>", unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        jam_belajar = st.number_input("Jam Belajar per Hari (0-24 jam)", min_value=0, max_value=24, value=None, placeholder=None)
        ipk = st.number_input("Performa Akademik (IPK 1.00-4.00)", min_value=1.00, max_value=4.00, value=None, placeholder=None)
    with col5:
        tekanan_ujian = st.number_input("Tekanan Ujian (1: Tidak Tertekan, 10: Sangat Tertekan)", min_value=1, max_value=10, value=None, placeholder=None)
        ekspektasi_keluarga = st.number_input("Ekspektasi Keluarga (1: Tidak Tertekan, 10: Sangat Tertekan)", min_value=1, max_value=10, value=None, placeholder=None)


    st.markdown("<div class='section-title text-green'>3. Mental & Gaya Hidup</div>", unsafe_allow_html=True)
    col_baris1_kiri, col_baris1_kanan = st.columns(2)
    with col_baris1_kiri:
        anxiety_score = st.number_input("Skor Kecemasan (1: Tidak Cemas, 10: Sangat Cemas)", min_value=1, max_value=10, value=None, placeholder=None)
    with col_baris1_kanan:
        aktivitas_fisik = st.number_input("Aktivitas Fisik per Minggu (0-7 hari aktif)", min_value=0, max_value=7, value=None, placeholder=None)


    col_baris2_kiri, col_baris2_kanan = st.columns(2)
    with col_baris2_kiri:
        depression_score = st.number_input("Skor Depresi (1: Tidak Depresi, 10: Sangat Depresi)", min_value=1, max_value=10, value=None, placeholder=None,)
    with col_baris2_kanan:
        screen_time = st.number_input("Waktu Layar (Screen Time) per Hari (0-24 jam)", min_value=0, max_value=24, value=None, placeholder=None)

    col_baris3_kiri, col_baris3_kanan = st.columns(2)
    with col_baris3_kiri:
        jam_tidur = st.number_input("Jam Tidur per Malam (0-24 jam)", min_value=0, max_value=24, value=None, placeholder=None)
    with col_baris3_kanan:
        internet_usage = st.number_input("Penggunaan Internet per Hari (0-24 jam)", min_value=0, max_value=24, value=None, placeholder=None)

    st.markdown("<div class='section-title text-purple'>4. Sosial & Finansial + Indeks Khusus</div>", unsafe_allow_html=True)
    col8, col9 = st.columns(2)
    with col8:
        social_support = st.number_input("Dukungan Sosial (1: Tidak Ada, 10: Sangat Ada)", min_value=1, max_value=10, value=None, placeholder=None)
        financial_stress = st.number_input("Stres Keuangan (1: Tidak Tertekan, 10: Sangat Tertekan)", min_value=1, max_value=10, value=None, placeholder=None)
    with col9:
        burnout_score = st.number_input("Skor Kejenuhan / Burnout (1: Tidak Kejenuhan, 10: Sangat Kejenuhan)", min_value=1, max_value=10, value=None, placeholder=None)
        mental_health_index = st.number_input("Indeks Kesehatan Mental (1: Sangat Buruk, 10: Sangat Baik)", min_value=1, max_value=10, value=None, placeholder=None)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("Analyze My Stress Level")
    
if submit_btn:
    if (umur is None or gender is None or tahun_akademik is None or 
        jam_belajar is None or ipk is None or tekanan_ujian is None or 
        ekspektasi_keluarga is None or anxiety_score is None or 
        depression_score is None or jam_tidur is None or 
        aktivitas_fisik is None or screen_time is None or 
        internet_usage is None or social_support is None or 
        financial_stress is None or burnout_score is None or 
        mental_health_index is None):
        
        st.error("Mohon lengkapi seluruh data kuesioner Anda sebelum melakukan analisis!")
        st.stop()
        
    st.session_state.input_data = {
        "umur": umur, "gender": gender, "tahun_akademik": tahun_akademik,
        "jam_belajar": jam_belajar, "ipk": ipk, "tekanan_ujian": tekanan_ujian,
        "ekspektasi_keluarga": ekspektasi_keluarga, "anxiety_score": anxiety_score,
        "depression_score": depression_score, "jam_tidur": jam_tidur,
        "aktivitas_fisik": aktivitas_fisik, "screen_time": screen_time,
        "internet_usage": internet_usage, "social_support": social_support,
        "financial_stress": financial_stress, "burnout_score": burnout_score,
        "mental_health_index": mental_health_index
    }
    st.switch_page("pages/hasilprediksi.py")