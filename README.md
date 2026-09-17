# 🔍 SmartVision — Object Classifier!

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras%20Model-FF6F00?logo=tensorflow&logoColor=white)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-Browser%20Inference-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Proof-of-concept Computer Vision untuk **SmartVision Sdn. Bhd.** yang mengklasifikasikan
4 jenis objek harian — **Phone, Mouse, Bottle, Watch** — menggunakan model
[Google Teachable Machine](https://teachablemachine.withgoogle.com/), dengan dua mod inferens
yang berasingan.

---

## 📑 Kandungan

- [Ciri-ciri](#-ciri-ciri)
- [Struktur Projek](#-struktur-projek)
- [Tech Stack](#-tech-stack)
- [Setup](#-setup)
- [Cara Guna](#-cara-guna)
- [Endpoint API](#-endpoint-api)
- [Troubleshooting](#-troubleshooting-masalah-yang-pernah-berlaku--sudah-dibaiki)
- [Nota](#-nota)

---

## ✨ Ciri-ciri

- **Mod Webcam** — inferens *real-time* terus dalam browser (TensorFlow.js), guna
  model tempatan dalam `frontend/model/`. Tiada round-trip ke server.
- **Mod Upload Imej** — imej dihantar ke backend **FastAPI + Pydantic**
  (`POST /predict`) yang menjalankan inferens sebenar menggunakan model
  Keras (`keras_model.h5`) hasil eksport Teachable Machine.
- **Threshold confidence** — ramalan dengan keyakinan < 70% dipaparkan sebagai
  `UNKNOWN / UNCERTAIN`, selaras di kedua-dua frontend dan backend.

## 📂 Struktur Projek

```
SmartVision-ObjectClassifier/
├── backend/
│   ├── main.py            # FastAPI app + endpoint /predict
│   ├── requirements.txt
│   └── model/
│       ├── keras_model.h5 # eksport dari Teachable Machine (Tensorflow > Keras)
│       ├── labels.txt
│       └── README.txt
└── frontend/
    ├── index.html
    ├── script.js
    ├── style.css
    └── model/              # eksport dari Teachable Machine (Tensorflow.js > Download)
        ├── model.json
        ├── metadata.json
        └── weights.bin
```

## 🛠 Tech Stack

| Lapisan | Teknologi |
|---|---|
| Frontend | HTML, CSS, JavaScript, TensorFlow.js, Teachable Machine Image Library |
| Backend | Python, FastAPI, Pydantic, Uvicorn |
| Model | Google Teachable Machine (Keras `.h5` + TF.js `model.json`) |
| Image Processing | Pillow, NumPy |

## 🚀 Setup

### 1. Jalankan Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Backend akan jalan di `http://127.0.0.1:8000`.
Dokumentasi Swagger UI (untuk uji endpoint terus): `http://127.0.0.1:8000/docs`

> **Nota versi TensorFlow:** `requirements.txt` termasuk package `tf-keras`.
> Ini **wajib** — TensorFlow terkini (2.16+) guna Keras 3 secara default, yang
> **tidak serasi** dengan format `.h5` lama yang dieksport Teachable Machine
> (akan gagal dengan ralat `Unrecognized keyword arguments: {'groups': 1}`).
> `main.py` sudah set `TF_USE_LEGACY_KERAS=1` untuk elak masalah ini — jangan
> buang baris tersebut.

### 2. Jalankan Frontend

Buka `frontend/index.html` menggunakan **Live Server (VS Code)** — **jangan**
buka fail terus melalui `file://` (double-click), kerana:

- Webcam & fetch API memerlukan konteks HTTP
- Model tempatan (`frontend/model/*.json`) hanya boleh dimuatkan melalui HTTP

## 🖱 Cara Guna

- Klik **START CAMERA** untuk inferens real-time terus dalam browser (guna
  model tempatan `frontend/model/`).
- Klik **UPLOAD IMAGE** untuk hantar imej ke backend FastAPI dan dapatkan
  ramalan daripada server (pastikan backend sudah berjalan di port 8000).
- Confidence < 70% akan dipaparkan sebagai `UNKNOWN / UNCERTAIN`.
- Klik **RESET** untuk mula semula.

## 🔌 Endpoint API

| Method | Path | Input | Output |
|---|---|---|---|
| `GET` | `/` | – | `{ "message": ..., "model_loaded": true/false }` |
| `POST` | `/predict` | `multipart/form-data`, field `file` (imej) | `{ "prediction": "PHONE", "confidence": 0.94, "status": "SUCCESS" }` |

## 🐛 Troubleshooting (masalah yang pernah berlaku & sudah dibaiki)

| Gejala | Punca | Status |
|---|---|---|
| Klik UPLOAD IMAGE — tiada apa berlaku, tiada ralat pun | `script.js` guna nama pembolehubah `URL` yang menimpa Web API global `URL.createObjectURL` | ✅ Dibaiki — ditukar ke `MODEL_URL` |
| Mod webcam predict kelas lain daripada yang dilatih | `MODEL_URL` masih tunjuk ke model contoh Google, bukan model sendiri | ✅ Dibaiki — guna fail tempatan `frontend/model/` |
| Backend bagi ralat `Unrecognized keyword arguments: {'groups': 1}` bila startup | TensorFlow 2.16+ (Keras 3) tak serasi dengan format `.h5` Teachable Machine | ✅ Dibaiki — `TF_USE_LEGACY_KERAS=1` + `tf-keras` dalam `requirements.txt` |
| Windows: `pip install` tensorflow bagi ralat "No such file or directory" / path panjang | Windows Long Path tidak diaktifkan | Aktifkan Long Path support atau abaikan jika install tetap berjaya |

## 📝 Nota

Projek ini dibangunkan untuk tugasan **DKB3263 — AI Image Classification &
Application Deployment**.

---

<p align="center">Dibangunkan untuk SmartVision Sdn. Bhd.</p>
