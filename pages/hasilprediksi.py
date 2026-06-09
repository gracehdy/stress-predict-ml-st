import streamlit as st
import requests
import time


st.set_page_config(
    page_title="StressPredict - Hasil Analisis", 
    layout="centered"
)


st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap">
    
    <style>
    * {font-family: 'Poppins', sans-serif !important;}
    [class*="st-emotion-cache"] i, 
        [data-testid="stSidebarCollapseButton"] i,
        .material-icons,
        [id*="Icon"] {
            font-family: inherit !important;
    }
    
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0px !important;
    }
    
    [data-testid="stSidebar"], [data-testid="stSidebarCollapseButton"], .st-emotion-cache-1wbqy5l {
        display: none !important;
        width: 0px !important;
    }
    
    .stApp {
        background-color: #E2F0ED !important;
    }
    
   .hasil-container-custom {
        border-radius: 24px !important; 
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.04) !important; 
        padding: 45px !important; 
        background-color: #FFFFFF !important;
        margin-top: 20px !important;
        margin-bottom: 25px !important;
        width: 100% !important;
        box-sizing: border-box;
    }

    .block-container {
        padding-top: 0rem !important;
        max-width: 850px !important;
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

    

    .main-title { font-size: 2.3rem; font-weight: 800; color: #1F2937; text-align: center; margin-bottom: 15px; line-height: 1.2; }
    .subtitle { font-size: 0.95rem; color: #6B7280; text-align: center; margin-bottom: 35px; font-weight: 400; line-height: 1.6; }
    
    .vue-inner-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 10px;
    }

    .result-section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #374151;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 2px solid #F3F4F6;
        padding-bottom: 8px;
    }

    .status-banner {
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 20px;
    }
    .status-high { background-color: #FEE2E2; color: #991B1B; }
    .status-mid { background-color: #FEF3C7; color: #92400E; }
    .status-low { background-color: #D1FAE5; color: #065F46; }

    .factor-box {
        background-color: #FFF7DD;
        border: none !important;
        padding: 16px 24px;
        color: #92400E;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-radius: 12px;
    }

    .recommendation-item {
        padding: 8px 0px;
        color: #4B5563;
        font-size: 0.95rem;
        line-height: 1.6;
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }

    div.stButton > button.reload-btn {
        background-color: #91C4C3 !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 14px 35px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 12px rgba(145, 196, 195, 0.3);
        transition: all 0.2s ease;
        margin-top: 15px;
    }
    div.stButton > button.reload-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(145, 196, 195, 0.5);
        background-color: #7DB3B2 !important;
    }
    </style>
""", unsafe_allow_html=True)

def render_navbar():
    is_disabled_result = "input_data" not in st.session_state
    disabled_class = "nav-item-disabled" if is_disabled_result else ""
    
    st.markdown(f"""
        <div class="vue-navbar">
            <div class="nav-brand">StressPredict</div>
            <div class="nav-links">
                <a href="?page=home" target="_self" class="nav-item-link">Beranda</a>
                <a href="?page=prediksi" target="_self" class="nav-item-link">Prediksi</a>
                <a href="?page=hasil" target="_self" class="nav-item-link {disabled_class}" style="color: #91C4C3 !important; background-color: #F3F4F6;">Hasil</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

    query_params = st.query_params
    if "page" in query_params:
        target = query_params["page"]
        st.query_params.clear() 
        if target == "home": st.switch_page("app.py")
        elif target == "prediksi": st.switch_page("pages/prediksi.py")
        elif target == "hasil" and not is_disabled_result: st.switch_page("pages/hasilprediksi.py")

render_navbar()

if "input_data" not in st.session_state:
    st.warning(" Silakan isi kuesioner terlebih dahulu di halaman Prediksi.")
    st.stop()

d = st.session_state.input_data
URL_PREDICT = "https://gracehdyc-stress-predict-api-2.hf.space/api/predict"
URL_RECOMMEND = "https://gracehdyc-stress-predict-api-2.hf.space/recommend"
start_time = time.time()

try:
    response = requests.post(URL_PREDICT, json=d, timeout=10)
    latency_ms = (time.time() - start_time) * 1000
    if response.status_code == 200:
        hasil_api = response.json()
        status_terprediksi = hasil_api.get("status", "Stres Sedang (Moderate)")
        score_display = hasil_api.get("confidence_score", 7.0)
        faktor_dominan = hasil_api.get("faktor_dominan", ["Beban Tekanan Ujian"])
        rekomendasi_ai = hasil_api.get("rekomendasi_ai", [])

        if "Tinggi" in status_terprediksi:
            status_class = "status-high"
        elif "Sedang" in status_terprediksi:
            status_class = "status-mid"
        else:
            status_class = "status-low"
       
        with st.spinner("Meminta saran konselor AI..."):
            data_ai = {
                "status_stres": status_terprediksi,
                "faktor_dominan": faktor_dominan
            }
            try:
                response_ai = requests.post(URL_RECOMMEND, json=data_ai, timeout=15)
                if response_ai.status_code == 200:
                    rekomendasi_ai = response_ai.json().get("rekomendasi_ai", "Tidak ada rekomendasi.")
                else:
                    rekomendasi_ai = "Atur jadwal tidur malam minimal 7 jam untuk pemulihan energi otak. Gunakan teknik Pomodoro (50 menit belajar, 10 menit istirahat). Diskusikan beban akademik dengan dosen pembimbing atau konselor kampus."
            except requests.exceptions.RequestException:
                rekomendasi_ai = "Koneksi ke server AI terputus."
                
                
        rekomendasi_list = response_ai.json().get("rekomendasi_ai", [])
        html_rekomendasi = ""
        for item in rekomendasi_list:
            html_rekomendasi += f'<div class="recommendation-item">• {item}</div>'

        st.markdown(f"""
<div class="hasil-container-custom">
<div class="main-title">Hasil Analisis Tingkat Stres</div>
<div class="subtitle">Berikut adalah kalkulasi prediksi tingkat stres Anda beserta rekomendasi solusinya.</div>
<div class="vue-inner-title">Status Deteksi Sistem:</div>
<div class="status-banner {status_class}">
{status_terprediksi} &nbsp;|&nbsp; Skor Keyakinan Model: {score_display}/10
</div>
<div class="result-section-title">Faktor Pemicu Terbesar</div>
<div class="factor-box">{', '.join(faktor_dominan)}</div>
<div class="result-section-title">Rekomendasi Tindakan Konselor AI</div>
{html_rekomendasi}
<br><hr style='border-color: #F3F4F6;'><br>
</div>
""", unsafe_allow_html=True)

        st.caption(f"Latensi: {latency_ms:.2f} ms")

    else:
        st.error(f"Gagal mengambil data dari server API. Kode Error: {response.status_code}")
        st.write("Respons Server:", response.text)

except requests.exceptions.ConnectionError:
    st.error("Gagal terhubung ke API Server. Pastikan Backend (FastAPI) sudah berjalan di terminal terpisah menggunakan perintah 'uvicorn api:app --reload'")


if st.button("Isi Ulang Kuesioner", key="btn_reload_form", use_container_width=True):
        del st.session_state.input_data
        st.switch_page("pages/prediksi.py")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
    <script>
    var buttons = window.parent.document.querySelectorAll('button');
    buttons.forEach(function(button) {
        if (button.textContent.includes("Isi Ulang Kuesioner")) {
            button.classList.add('reload-btn');
        }
    });
    </script>
""", unsafe_allow_html=True)
