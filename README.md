  Como usar

  Fase 1 - Recolectar datos:
  python3 data/collect_data.py
  - Se abre la camara con menu interactivo
  - Presiona 1-9, 0, a-d para seleccionar una sena
  - Countdown de 3 seg, luego graba 15 seg de landmarks
  - Los datos se guardan en data/landmarks.csv
  - Graba cada sena minimo 3-5 veces

  Fase 2 - Entrenar modelo:
  python3 model/train.py
  - Carga el CSV, aplica augmentation (6x), entrena 100 epochs
  - Guarda el mejor modelo en model/sign_model.pth
  - Guarda mapeo de clases en model/label_encoder.json
  - Muestra confusion matrix y accuracy por sena