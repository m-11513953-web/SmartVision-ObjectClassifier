Letakkan fail eksport Teachable Machine di sini:

  keras_model.h5
  labels.txt

Cara eksport:
1. Buka projek di Teachable Machine (teachablemachine.withgoogle.com)
2. Export Model -> tab "Tensorflow" -> pilih "Keras"
3. Download my model -> extract zip -> salin 2 fail di atas ke folder ini

Selepas itu, jalankan backend:
  cd backend
  pip install -r requirements.txt
  python main.py

Server akan berjalan di http://127.0.0.1:8000
Dokumentasi automatik (Swagger UI): http://127.0.0.1:8000/docs
