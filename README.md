# 🕵️‍♂️ TrueNews: AI-Powered Fact-Checker

TrueNews adalah platform deteksi berita palsu (Hoax) berbasis AI yang menggabungkan kecepatan Machine Learning Konvensional dengan kecerdasan Large Language Models (LLM). Sistem ini dirancang untuk memberikan verifikasi informasi yang akurat, cepat, dan edukatif.

---

## 🚀 Fitur Utama
1. Hybrid Intelligence: Menggunakan XGBoost sebagai backbone klasifikasi cepat dan Groq (Llama 3.3) sebagai analis interpretasi.
2. Logic Gate Thresholding: Sistem secara otomatis memberikan status "PERLU VERIFIKASI" jika tingkat keyakinan model di bawah 80%.
3. Branded Interpretation: Memberikan penjelasan mendalam dengan identitas Model TrueNews.
4. Industry Standard API: Dibangun dengan Flask dan siap dideploy ke Microsoft Azure App Service.

---

## 🛠️ Tech Stack
1. Backend: Python, Flask
2. Machine Learning: XGBoost, Scikit-Learn (v1.7.2)
3. AI Analyst: Groq Cloud API (Llama-3.3-70b-versatile)
4. Deployment: Azure App Service, GitHub Actions

---

## Struktur Proyek
```plaintext
(root)
├── models/
│   ├── tfidf_vectorizer_v1.pkl      # Vectorizer untuk ekstraksi fitur teks
│   └── truenews_model_xgb_v1.pkl    # Model XGBoost yang sudah dilatih
├── app.py                           # Flask API utama (Standard Industri)
├── requirements.txt                 # Daftar dependensi library
├── .env                             # Environment Variables (Local only)
└── .gitignore                       # Filter untuk file yang tidak di-upload
```

---

## ⚙️ Instalasi Lokal
1. Clone Repository:
```bash
git clone https://github.com/username/truenews-backend.git
cd truenews-backend
```

2. Buat Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instal Dependensi:
```bash
pip install -r requirements.txt
```

4. Konfigurasi Environment:
Buat file `.env` di root folder dan tambahkan API Key kamu:
```plaintext
GROQ_API_KEY=your_api_key_here
```

5. Jalankan Server:
```bash
python app.py
```
