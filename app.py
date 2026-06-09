import streamlit as st
from groq import Groq
import numpy as np
import joblib
import json
import time


st.set_page_config(
    page_title="StressPredict - Beranda", 
    layout="centered"
)

st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap">
    
    <style>
    header[data-testid="stHeader"] {
            display: none !important;
            height: 0px !important;
        }
        
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
    
        
   .main-card {
        border-radius: 24px !important; 
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.04) !important; 
        padding: 50px !important; 
        background-color: #FFFFFF !important; 
        margin-top: 20px !important;
        margin-bottom: 40px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    
    div[data-testid="stVerticalBlock"] {
        gap: 0rem !important;
    
 
    .block-container {
        padding-top: 0rem !important;
        max-width: 800px !important;
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
    .nav-item button {
        background: none !important;
        border: none !important;
        color: #4B5563 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        cursor: pointer;
        transition: all 0.2s ease !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    .nav-item button:hover {
        color: #91C4C3 !important;
        background-color: #F3F4F6 !important;
    }
    .nav-item button:disabled {
        color: #D1D5DB !important;
        background: none !important;
        cursor: not-allowed;
    }

    .main-card {
        border-radius: 24px; 
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.04); 
        padding: 50px; 
        background-color: #FFFFFF !important; 
        margin-bottom: 40px;
    }
    .main-title { font-size: 2.4rem; font-weight: 800; color: #1F2937; text-align: center; margin-bottom: 12px; }
    .subtitle { font-size: 1rem; color: #6B7280; text-align: center; margin-bottom: 40px; font-weight: 400; line-height: 1.6; }

    .feature-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #374151;
        margin-bottom: 20px;
        border-bottom: 2px solid #F3F4F6;
        padding-bottom: 8px;
    }
    
    
    .grid-2-kolom {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 20px;
    }
    
    @media (max-width: 600px) {
        .grid-2-kolom {
            grid-template-columns: 1fr;
        }
    }
    
    .box-info-custom { 
        background-color: #FFF7DD !important; 
        padding: 20px !important; 
        border-radius: 16px !important; 
        border: 1px solid #FCD34D !important;
        box-sizing: border-box;
    }
    
    .box-info-custom h4 { 
        margin: 0 0 6px 0 !important; 
        color: #92400E !important; 
        font-size: 1rem;
        font-weight: 600;
    }
    
    .box-info-custom p {
        color: #6B7280 !important;
        font-size: 0.85rem !important;
        line-height: 1.5 !important;
        margin: 0 !important;
    }
    
    .list-item {
        display: flex; 
        align-items: flex-start; 
        gap: 10px; 
        margin-bottom: 14px;
    }
    
    .list-icon {
        color: #91C4C3; 
        font-weight: bold; 
        font-size: 1.1rem; 
        line-height: 1;
    }
    
    .list-text {
        color: #4B5563; 
        font-size: 0.9rem; 
        line-height: 1.5;
    }
    
    div.stButton > button.cta-btn {
        background-color: #91C4C3 !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 14px 0px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(145, 196, 195, 0.3);
        transition: all 0.2s ease;
    }
    div.stButton > button.cta-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(145, 196, 195, 0.5);
        background-color: #7DB3B2 !important;
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
            max-width: 800px !important;
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

render_navbar()

st.markdown("""
<div class="main-card">
    <div class="main-title">Student Stress Prediction System</div>
    <div class="subtitle">Sistem analisis berbasis Machine Learning yang dirancang khusus untuk mendeteksi tingkat stres mahasiswa berdasarkan performa akademik, beban ujian, serta pola gaya hidup harian Anda.</div>
</div>
<br>
""", unsafe_allow_html=True)

_, col_center, _ = st.columns([0.5, 2, 0.5])
with col_center:
        if st.button("Mulai Prediksi Tingkat Stress Anda!", use_container_width=True, key="cta_main_btn"):
            st.switch_page("pages/prediksi.py")

st.markdown("""
<div class="main-card">
    <div class="feature-header">Fitur Unggulan Sistem</div>
    <div class="grid-2-kolom">
        <div class="box-info-custom">
            <h4>1. Prediksi Tingkat Stres</h4>
            <p>Analisis tingkat kerentanan stres berdasarkan metrik akademik dan aktivitas harian menggunakan algoritma ML yang akurat.</p>
        </div>
        <div class="box-info-custom">
            <h4>2. Analisis Faktor Penyebab</h4>
            <p>Identifikasi langsung indikator atau faktor dominan utama yang paling memicu kejenuhan (burnout) diri Anda.</p>
        </div>
        <div class="box-info-custom">
            <h4>3. Rekomendasi Konselor AI</h4>
            <p>Dapatkan saran preventif medis maupun psikologis personal secara instan yang diproses cerdas via Groq AI Infrastructure.</p>
        </div>
        <div class="box-info-custom">
            <h4>4. Dashboard Indeks Mental</h4>
            <p>Pemetaan visual data parameter kesehatan mental mahasiswa yang dikemas ringkas, infografis, dan mudah dipahami.</p>
        </div>
    </div>
    <div class="grid-2-kolom" style="margin-top: 40px;">
        <div>
            <div class="feature-header">Tujuan Sistem</div>
            <div class="list-item"><span class="list-icon">✓</span><span class="list-text">Membantu mahasiswa mengenali tingkat stres sedini mungkin</span></div>
            <div class="list-item"><span class="list-icon">✓</span><span class="list-text">Mendorong gaya hidup sehat dan keseimbangan akademik</span></div>
            <div class="list-item"><span class="list-icon">✓</span><span class="list-text">Menyediakan data berbasis bukti untuk institusi pendidikan</span></div>
            <div class="list-item"><span class="list-icon">✓</span><span class="list-text">Mengurangi stigma terkait kesehatan mental di lingkungan kampus</span></div>
        </div>
        <div>
            <div class="feature-header">Dampak & Manfaat</div>
            <div class="list-item"><span class="list-icon">✓</span><span class="list-text">Mahasiswa lebih sadar akan kesehatan mentalnya</span></div>
            <div class="list-item"><span class="list-icon">✓</span><span class="list-text">Peningkatan performa akademik setelah intervensi dini</span></div>
            <div class="list-item"><span class="list-icon">✓</span><span class="list-text">Efektivitas rekomendasi berbasis data vs umum</span></div>
        </div>
    </div>
    <br>
</div>
""", unsafe_allow_html=True)



st.markdown("""
    <script>
    var buttons = window.parent.document.querySelectorAll('button');
    buttons.forEach(function(button) {
        if (button.textContent.includes("Mulai Prediksi Tingkat Stress")) {
            button.classList.add('cta-btn');
        }
    });
    </script>
""", unsafe_allow_html=True)

