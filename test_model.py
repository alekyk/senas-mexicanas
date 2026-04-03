"""
Mira&Tek - Script de Prueba en Tiempo Real del Modelo LSM
Technovation Girls 2026

=== QUE HACE ESTE ARCHIVO? ===
Este es el PASO 3 del proyecto. Abre la camara, detecta tus manos,
y usa el modelo entrenado para PREDECIR que sena estas haciendo
en tiempo real, mostrando el nombre de la sena y el porcentaje
de confianza.

Esto sirve para verificar que el modelo se entreno correctamente.

=== COMO SE USA? ===
1. Primero: haber entrenado el modelo con model/train.py
2. Abre la terminal
3. Escribe: python3 test_model.py
4. Se abre la camara
5. Haz una sena frente a la camara
6. Veras en pantalla que sena detecta y con que porcentaje
7. Presiona 'q' para salir
"""

import os
import sys
import json
import argparse

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

# Agregar model/ al path para importar la arquitectura
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'))
from model import SignLanguageMLP

# ============================================================
# RUTAS DE ARCHIVOS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'sign_model.pth')
ENCODER_PATH = os.path.join(BASE_DIR, 'model', 'label_encoder.json')
HAND_MODEL_PATH = os.path.join(BASE_DIR, 'data', 'hand_landmarker.task')

# Umbral minimo de confianza para mostrar una prediccion
# Si el modelo tiene menos de 50% de confianza, muestra "No detectado"
MIN_CONFIDENCE = 0.50


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def get_device():
    """Detecta el mejor hardware disponible."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def extract_landmarks(result):
    """Extrae las 126 coordenadas de las manos del resultado de MediaPipe.

    Misma funcion que en collect_data.py para mantener consistencia.
    """
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    if result.hand_landmarks:
        for hand_landmarks, handedness_list in zip(
            result.hand_landmarks,
            result.handedness
        ):
            label = handedness_list[0].category_name
            coords = []
            for lm in hand_landmarks:
                coords.extend([lm.x, lm.y, lm.z])

            if label == 'Left':
                left_hand = coords
            else:
                right_hand = coords

    return left_hand + right_hand


def draw_hand_landmarks(frame, result):
    """Dibuja los puntos y conexiones de las manos sobre el video."""
    draw_landmarks = mp.tasks.vision.drawing_utils.draw_landmarks
    drawing_styles = mp.tasks.vision.drawing_styles
    hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_landmarks(
                frame,
                hand_landmarks,
                hand_connections,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )


def draw_prediction(frame, sign_name, confidence, color):
    """Dibuja la prediccion del modelo en la parte superior de la pantalla."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Fondo semi-transparente arriba
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Nombre de la sena (grande)
    cv2.putText(frame, sign_name, (15, 40), font, 1.2, color, 3, cv2.LINE_AA)

    # Porcentaje de confianza
    pct_text = f"{confidence:.1f}%"
    cv2.putText(frame, pct_text, (15, 75), font, 0.8, color, 2, cv2.LINE_AA)

    # Barra de confianza visual
    bar_x = 200
    bar_w = w - 220
    bar_y = 58
    bar_h = 20
    # Fondo gris
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)
    # Barra de color segun confianza
    fill_w = int(bar_w * min(confidence / 100.0, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                  color, -1)
    # Borde
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (255, 255, 255), 1)


def draw_status_bar(frame, has_hands, class_names):
    """Dibuja la barra inferior con informacion de estado."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    hands_status = "Manos: SI" if has_hands else "Manos: NO"
    hands_color = (0, 255, 0) if has_hands else (0, 0, 255)
    cv2.putText(frame, hands_status, (10, h - 10), font, 0.5,
                hands_color, 1, cv2.LINE_AA)

    info = f"Senas: {len(class_names)} | 'q' para salir"
    cv2.putText(frame, info, (w - 300, h - 10), font, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)


# ============================================================
# FUNCION PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Prueba en tiempo real del modelo LSM')
    parser.add_argument('--camera', type=int, default=0,
                        help='Indice de la camara (default: 0)')
    args = parser.parse_args()

    # --- Verificar archivos necesarios ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontro el modelo entrenado: {MODEL_PATH}")
        print("Primero entrena el modelo con: python3 model/train.py")
        sys.exit(1)

    if not os.path.exists(ENCODER_PATH):
        print(f"Error: No se encontro el label encoder: {ENCODER_PATH}")
        print("Primero entrena el modelo con: python3 model/train.py")
        sys.exit(1)

    if not os.path.exists(HAND_MODEL_PATH):
        print(f"Error: No se encontro el modelo de MediaPipe: {HAND_MODEL_PATH}")
        sys.exit(1)

    # --- Cargar lista de senas ---
    with open(ENCODER_PATH, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"Senas que conoce el modelo ({num_classes}): {class_names}")

    # --- Cargar modelo entrenado ---
    device = get_device()
    print(f"Dispositivo: {device}")

    model = SignLanguageMLP(input_size=126, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()  # Modo evaluacion (desactiva Dropout)
    print("Modelo cargado correctamente.")

    # --- Inicializar MediaPipe ---
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = HandLandmarker.create_from_options(options)

    # --- Abrir camara ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la camara {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nCamara abierta. Haz una sena frente a la camara.")
    print("Presiona 'q' para salir.\n")

    frame_timestamp = 0

    # ============================================================
    # LOOP PRINCIPAL
    # ============================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer frame de la camara")
            break

        frame = cv2.flip(frame, 1)  # Modo espejo

        # --- Detectar manos ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp += 33
        result = hand_landmarker.detect_for_video(mp_image, frame_timestamp)

        # Dibujar landmarks sobre las manos
        draw_hand_landmarks(frame, result)

        has_hands = len(result.hand_landmarks) > 0

        if has_hands:
            # --- Extraer landmarks y predecir ---
            landmarks = extract_landmarks(result)
            input_tensor = torch.FloatTensor([landmarks]).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                # Softmax convierte los numeros de salida en probabilidades (0-1)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                confidence_pct = confidence.item() * 100
                predicted_class = class_names[predicted_idx.item()]

            # --- Mostrar resultado ---
            if confidence_pct >= MIN_CONFIDENCE * 100:
                # Color segun confianza: verde (alta), amarillo (media), rojo (baja)
                if confidence_pct >= 80:
                    color = (0, 255, 0)      # Verde
                elif confidence_pct >= 60:
                    color = (0, 255, 255)    # Amarillo
                else:
                    color = (0, 165, 255)    # Naranja

                draw_prediction(frame, predicted_class, confidence_pct, color)

                # Imprimir en terminal tambien
                print(f"  {predicted_class}: {confidence_pct:.1f}%", end='\r')
            else:
                draw_prediction(frame, "No seguro", confidence_pct, (100, 100, 100))
        else:
            draw_prediction(frame, "Muestra tus manos", 0, (100, 100, 100))

        # Barra de estado inferior
        draw_status_bar(frame, has_hands, class_names)

        cv2.imshow('Prueba del Modelo LSM - Mira&Tek', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Limpiar ---
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nPrueba finalizada.")


if __name__ == '__main__':
    main()
