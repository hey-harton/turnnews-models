import logging
import os
import tempfile
import json

import awsgi
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
    "https://turnnews-dev.dataspace.my.id",
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

# =====================================================================
# HYBRID MODEL LOADER (Cloudflare R2 + AWS Lambda / Local Fallback)
# =====================================================================
model = None
vectorizer = None
model_load_error = "Sistem belum mencoba memuat model."

try:
    R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")

    # Jika R2_ENDPOINT_URL ada di Environment Variables (Mode Produksi AWS Lambda)
    if R2_ENDPOINT:
        import boto3
        logger.info("Mode Cloud: Mencoba mengunduh model dari Cloudflare R2...")
        
        s3_client = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            region_name="auto"
        )
        BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "turnnews-bucket")
        
        # tempfile.gettempdir() aman untuk Windows (C:\Temp) dan Linux (/tmp)
        TEMP_DIR = tempfile.gettempdir() 
        MODEL_PATH = os.path.join(TEMP_DIR, "truenews_model_xgb_v1.pkl")
        VECTORIZER_PATH = os.path.join(TEMP_DIR, "tfidf_vectorizer_v1.pkl")

        # Mengunduh hanya jika file belum ada di memory sementara (Warm Start Optimization)
        if not os.path.exists(MODEL_PATH):
            logger.info("Mengunduh truenews_model_xgb_v1.pkl dari R2...")
            s3_client.download_file(BUCKET_NAME, "truenews_model_xgb_v1.pkl", MODEL_PATH)

        if not os.path.exists(VECTORIZER_PATH):
            logger.info("Mengunduh tfidf_vectorizer_v1.pkl dari R2...")
            s3_client.download_file(BUCKET_NAME, "tfidf_vectorizer_v1.pkl", VECTORIZER_PATH)
            
    # Jika R2_ENDPOINT_URL tidak ada (Mode Development Lokal di VS Code)
    else:
        logger.info("Mode Lokal: Menggunakan model dari folder lokal...")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "models", "truenews_model_xgb_v1.pkl")
        VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer_v1.pkl")

    # Load ke memory
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    model_load_error = None  # Kosongkan error jika berhasil
    logger.info("Sistem Backbone dan Analyst berhasil dimuat!")

except Exception as exc:
    model_load_error = str(exc)
    logger.error(f"Gagal memuat model atau vectorizer: {model_load_error}")
# =====================================================================


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

    # 🚀 CEK ERROR INISIALISASI MODEL DARI R2
    global model, vectorizer, model_load_error
    if model is None or vectorizer is None:
        return jsonify({
            "error": "Sistem gagal memuat model ML dari Cloudflare R2.",
            "r2_detail": model_load_error
        }), 500

    try:
        # 🛡️ BLOK PARSING ANTI-BADAI AWS
        data = request.get_json(force=True, silent=True)
        if not data:
            raw_data = request.get_data(as_text=True)
            data = json.loads(raw_data) if raw_data else {}
        
        # Mengambil "content" (Postman/DB) atau "text" (Default)
        raw_text = data.get("content") or data.get("text")
        
        if not raw_text or not isinstance(raw_text, str):
            return jsonify({"error": "Payload tidak valid atau teks kosong"}), 400

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
        return jsonify({"error": f"Internal Server Error: {str(exc)}"}), 500


# AWS Lambda handler called by API Gateway or Function URL.
def handler(event, context):
    if "httpMethod" not in event and "requestContext" in event and "http" in event["requestContext"]:
        event["httpMethod"] = event["requestContext"]["http"]["method"]
        event["path"] = event["requestContext"]["http"]["path"]
        if "queryStringParameters" not in event:
            event["queryStringParameters"] = {}

    return awsgi.response(app, event, context)


# Local entry point.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)