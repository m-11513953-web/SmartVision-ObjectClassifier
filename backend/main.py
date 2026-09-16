"""
SmartVision Object Classifier API
----------------------------------
Backend FastAPI + Pydantic yang memuatkan model Google Teachable Machine
(format Keras .h5) dan menjalankan inferens SEBENAR ke atas imej yang
dihantar oleh frontend, menggantikan mock prediction sebelum ini.

Cara dapatkan fail model:
1. Buka projek anda di Teachable Machine.
2. Klik "Export Model" -> tab "Tensorflow" -> pilih "Keras".
3. Klik "Download my model" -> anda akan dapat fail zip mengandungi:
      keras_model.h5
      labels.txt
4. Letak KEDUA-DUA fail tersebut dalam folder: backend/model/
"""

import io
import os

# PENTING: TensorFlow versi terkini (2.16+) guna Keras 3 secara default, yang
# TIDAK serasi dengan format fail .h5 lama yang dieksport oleh Google Teachable
# Machine (punca ralat "Unrecognized keyword arguments: {'groups': 1}" /
# "expects 1 named input(s)... but it received 2 input tensors").
# Baris ini paksa TensorFlow guna Keras 2 (legacy) untuk baca fail TM dengan betul.
# Perlukan package tambahan: pip install tf-keras
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
CONFIDENCE_THRESHOLD = 0.70  # selaras dengan threshold di frontend (script.js)
IMAGE_SIZE = (224, 224)      # saiz input piawai Teachable Machine

app = FastAPI(
    title="SmartVision Object Classifier API",
    description="API Inference Endpoint bagi integrasi Backend Python (SmartVision Sdn. Bhd.)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Load model & label sekali sahaja semasa startup (bukan setiap request)
# ----------------------------------------------------------------------
model = None
class_names: list[str] = []
model_load_error: str | None = None


@app.on_event("startup")
def load_model():
    global model, class_names, model_load_error
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            model_load_error = (
                "Fail model tidak dijumpai. Sila eksport model dari Teachable "
                "Machine (Tensorflow -> Keras) dan letakkan keras_model.h5 & "
                "labels.txt dalam folder backend/model/."
            )
            print(f"[AMARAN] {model_load_error}")
            return

        # Import di sini supaya server masih boleh start walaupun tensorflow
        # belum di-install (mesej ralat lebih jelas kepada pelajar).
        from tensorflow.keras.models import load_model as keras_load_model

        model = keras_load_model(MODEL_PATH, compile=False)

        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            # Format labels.txt Teachable Machine: "0 NamaKelas"
            class_names = [line.strip().split(" ", 1)[-1] for line in f if line.strip()]

        print(f"[OK] Model berjaya dimuatkan. Kelas: {class_names}")
    except Exception as exc:  # noqa: BLE001
        model_load_error = f"Gagal memuatkan model: {exc}"
        print(f"[RALAT] {model_load_error}")


# ----------------------------------------------------------------------
# Skema Pydantic (Pydantic V2 compliant)
# ----------------------------------------------------------------------
class PredictionResponse(BaseModel):
    prediction: str = Field(..., json_schema_extra={"example": "Buku"})
    confidence: float = Field(..., json_schema_extra={"example": 0.94})
    status: str = Field(..., json_schema_extra={"example": "SUCCESS"})


class HealthResponse(BaseModel):
    message: str
    model_loaded: bool


# ----------------------------------------------------------------------
# Fungsi pembantu: pra-proses imej ikut cara Teachable Machine
# ----------------------------------------------------------------------
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Fail imej tidak sah: {exc}") from exc

    # Resize + crop tengah supaya sepadan dengan cara Teachable Machine
    # memproses imej (letterbox-free center crop ke 224x224).
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)

    image_array = np.asarray(image, dtype=np.float32)
    # Normalisasi ke julat [-1, 1] — sama seperti tmImage.js
    normalized = (image_array / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/", response_model=HealthResponse)
def read_root():
    return HealthResponse(
        message="SmartVision API is running successfully.",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_object(file: UploadFile = File(...)):
    """
    Endpoint inferens API sebenar mengikut spesifikasi tugasan (seksyen 9.0).
    Terima fail imej (multipart/form-data), pulangkan kelas ramalan + confidence.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Model belum dimuatkan.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sila muat naik fail imej sahaja.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Imej tidak disediakan.")

    input_tensor = preprocess_image(image_bytes)
    predictions = model.predict(input_tensor, verbose=0)[0]

    top_index = int(np.argmax(predictions))
    confidence = float(predictions[top_index])
    label = class_names[top_index] if top_index < len(class_names) else f"Kelas {top_index}"

    # Pengendalian threshold confidence rendah (selaras dengan seksyen 8.0 & 11.0)
    final_label = label if confidence >= CONFIDENCE_THRESHOLD else "UNKNOWN / UNCERTAIN"
    status = "SUCCESS" if confidence >= CONFIDENCE_THRESHOLD else "LOW_CONFIDENCE"

    return PredictionResponse(
        prediction=final_label,
        confidence=round(confidence, 4),
        status=status,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
