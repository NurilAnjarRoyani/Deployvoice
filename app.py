import streamlit as st
import os
from utils.preprocess import extract_features
from utils.predict import predict_voice

st.set_page_config(page_title="Voice Identification App", page_icon="🎤")

# === CSS CUSTOM ===
st.markdown("""
    <style>
        /* Background utama dan warna teks */
        .stApp {
            background-color: #000000;
            color: white;
        }

        /* Hilangkan background abu-abu di uploader */
        [data-testid="stFileUploaderDropzone"] {
            background-color: #000000 !important;
            border: 2px dashed #444 !important;
        }

        /* Warna teks dan ikon di uploader */
        [data-testid="stFileUploaderDropzone"] * {
            color: white !important;
        }

        /* Warna heading */
        h1, h2, h3, h4, h5 {
            color: #00ffcc !important;
        }

        /* Tombol custom */
        .stButton>button {
            background-color: #00ffcc;
            color: #000;
            border-radius: 8px;
            font-weight: bold;
            transition: 0.3s;
            border: none;
        }

        .stButton>button:hover {
            background-color: #00ffaa;
            color: black;
            transform: scale(1.05);
        }

        /* Pesan info, success, error agar cocok dengan tema hitam */
        .stAlert {
            background-color: #111 !important;
            color: white !important;
            border-left: 5px solid #00ffcc !important;
        }
    </style>
""", unsafe_allow_html=True)

# === UI Aplikasi ===
st.title("🎤 Voice Identification App")
st.write("Unggah file audio (format `.wav`) untuk dikenali suaranya.")

uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("🔍 Identifikasi Suara"):
        with st.spinner("Sedang memproses..."):
            features = extract_features("temp.wav")
            if features is not None:
                pred_class, confidence = predict_voice(features)
                if pred_class:
                    st.success(f"**Hasil Prediksi:** {pred_class}")
                    st.info(f"Kepercayaan: {confidence:.2f}%")
                else:
                    st.error("Gagal melakukan prediksi.")
            else:
                st.error("Gagal mengekstraksi fitur dari audio.")
else:
    st.info("Silakan unggah file audio terlebih dahulu.")
