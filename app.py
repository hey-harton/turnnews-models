import os
import logging
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

# 1. Konfigurasi Logging (Standard Industri)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 2. Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
CORS(app)

# Inisialisasi Client Groq
client = Groq(api_key=GROQ_API_KEY)

# 3. Load Model & Vectorizer dengan Path Aman (Linux/Windows Compatible)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'truenews_model_xgb_v1.pkl')
    VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer_v1.pkl')
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("Sistem Backbone dan Analyst berhasil dimuat")
except Exception as e:
    logger.error(f"Gagal memuat model atau vectorizer: {str(e)}")

def get_groq_interpretation(text, result, confidence):
    """
    Menghasilkan analisis cerdas dengan pembersihan karakter otomatis.
    """
    try:
        prompt = f"""
        Tugas: Analisis Verifikasi Berita TrueNews
        Teks: "{text}"
        Klasifikasi Awal: {result}
        Keyakinan Model: {confidence}

        Instruksi:
        1. Jelaskan fakta/hoax berita ini berdasarkan basis data pengetahuan global Anda.
        2. Gunakan kalimat: "Model TrueNews memiliki keyakinan sebesar {confidence}".
        3. Jelaskan alasan teknis (bahasa/sumber) dan berikan kesimpulan akhir yang tegas.

        Aturan: Bahasa Indonesia, profesional, maksimal 4 kalimat, tanpa karakter spesial berlebih.
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Anda adalah Senior Auditor TrueNews yang bertugas memberikan edukasi literasi berita secara objektif."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        # Membersihkan hasil interpretasi dari tanda kutip ganda yang tidak perlu
        raw_content = completion.choices[0].message.content
        clean_content = raw_content.replace('\\"', '"').replace('\"', '"').strip()
        return clean_content

    except Exception as e:
        logger.error(f"Kesalahan pada Groq Analyst: {str(e)}")
        return "Interpretasi saat ini tidak tersedia, silakan verifikasi secara manual."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Payload tidak valid atau teks kosong"}), 400
        
        raw_text = data['text']

        # 4. Transformasi dan Prediksi
        vector = vectorizer.transform([raw_text])
        prediction = int(model.predict(vector)[0])
        probability = model.predict_proba(vector)[0]

        raw_label = "HOAX" if prediction == 1 else "FAKTA"
        confidence_val = float(max(probability)) * 100
        confidence_str = f"{confidence_val:.2f}%"
        
        # 5. Logic Gate: Thresholding (Standard Keamanan Data)
        final_label = raw_label
        if confidence_val < 80.0:
            final_label = "PERLU VERIFIKASI"

        # 6. Interpretasi Analyst
        interpretation = get_groq_interpretation(raw_text, final_label, confidence_str)

        logger.info(f"Prediksi berhasil: {final_label} ({confidence_str})")

        return jsonify({
            "status": "success",
            "result": final_label,
            "confidence": confidence_str,
            "class_id": prediction,
            "interpretation": interpretation
        })

    except Exception as e:
        logger.error(f"Terjadi kesalahan sistem: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

# Entry point untuk Gunicorn (Azure Standard)
if __name__ == '__main__':
    # Gunakan port dari environment variable jika ada (Azure default)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)