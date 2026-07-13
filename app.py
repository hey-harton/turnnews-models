import logging
import os

import aws_lambda_wsgi
import joblib
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from groq import Groq


# Logging configuration.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Load environment variables.
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


app = Flask(__name__)


# CORS configuration for allowed frontend origins.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://turnnews-site.vercel.app",
]

CORS(
    app,
    resources={
        r"/*": {
            "origins": ALLOWED_ORIGINS,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "Accept"],
            "supports_credentials": True,
        }
    },
)


# Groq client initialization.
client = Groq(api_key=GROQ_API_KEY)


# Load model and vectorizer from local paths compatible with AWS Lambda.
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "truenews_model_xgb_v1.pkl")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer_v1.pkl")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("Sistem Backbone dan Analyst berhasil dimuat di AWS Lambda")
except Exception as exc:
    logger.error(f"Gagal memuat model atau vectorizer: {str(exc)}")


def get_groq_interpretation(text, result, confidence):
    """Menghasilkan interpretasi analitis dari hasil klasifikasi."""

    try:
        prompt = f"""
        Tugas: Analisis Verifikasi Berita TrunNews_
        Teks: "{text}"
        Klasifikasi Awal: {result}
        Keyakinan Model: {confidence}

        Instruksi:
        1. Jelaskan fakta/hoax berita ini berdasarkan basis data pengetahuan global Anda.
        2. Gunakan kalimat: "Model TrunNews_ memiliki keyakinan sebesar {confidence}".
        3. Jelaskan alasan teknis (bahasa/sumber) dan berikan kesimpulan akhir yang tegas.

        Aturan: Bahasa Indonesia, profesional, maksimal 4 kalimat, tanpa karakter spesial berlebih.
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Anda adalah Senior Auditor TrunNews_ yang bertugas "
                        "memberikan edukasi literasi berita secara objektif."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )

        raw_content = completion.choices[0].message.content
        clean_content = raw_content.replace('\\"', '"').replace('"', '"').strip()
        return clean_content

    except Exception as exc:
        logger.error(f"Kesalahan pada Groq Analyst: {str(exc)}")
        return "Interpretasi saat ini tidak tersedia, silakan verifikasi secara manual."


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Payload tidak valid atau teks kosong"}), 400

        raw_text = data["text"]

        # Transform text into vector representation and predict the class.
        vector = vectorizer.transform([raw_text])
        prediction = int(model.predict(vector)[0])
        probability = model.predict_proba(vector)[0]

        raw_label = "HOAX" if prediction == 1 else "FAKTA"
        confidence_val = float(max(probability)) * 100
        confidence_str = f"{confidence_val:.2f}%"

        # Apply confidence thresholding.
        final_label = raw_label
        if confidence_val < 80.0:
            final_label = "PERLU VERIFIKASI"

        # Build analyst interpretation.
        interpretation = get_groq_interpretation(raw_text, final_label, confidence_str)

        logger.info(f"Prediksi berhasil: {final_label} ({confidence_str})")

        return jsonify(
            {
                "status": "success",
                "result": final_label,
                "confidence": confidence_str,
                "class_id": prediction,
                "interpretation": interpretation,
            }
        )

    except Exception as exc:
        logger.error(f"Terjadi kesalahan sistem: {str(exc)}")
        return jsonify({"error": "Internal Server Error"}), 500


# AWS Lambda handler called by API Gateway or Function URL.
def handler(event, context):
    return aws_lambda_wsgi.response(app, event, context)


# Local entry point.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)