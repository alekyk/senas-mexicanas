"""
Mira&Tek - Script de Recoleccion de Datos para LSM
Technovation Girls 2026

=== QUE HACE ESTE ARCHIVO? ===
Este es el PASO 1 del proyecto. Abre la camara de tu computadora,
detecta tus manos usando inteligencia artificial (MediaPipe de Google),
y guarda las posiciones de tus dedos en un archivo CSV.

Piensen en esto como "tomarle fotos" a cada sena, pero en vez de guardar
la imagen completa, solo guardamos los NUMEROS que describen donde estan
los dedos. Esos numeros son los "landmarks" (puntos clave).

=== COMO SE USA? ===
1. Abre la terminal
2. Escribe: python3 data/collect_data.py
3. Se abre la camara
4. Presiona la tecla de la sena que quieres grabar (ej: 1 para Hola)
5. Tienes 3 segundos para prepararte
6. Haz la sena durante 15 segundos frente a la camara
7. Los datos se guardan automaticamente en data/landmarks.csv
8. Presiona 'q' para salir

=== CUANTAS VECES GRABAR? ===
Minimo 3-5 veces por sena, idealmente variando:
  - Angulo (un poco a la izquierda, derecha, de frente)
  - Distancia (cerca, lejos)
  - Iluminacion (cuarto con luz, cuarto oscuro)
  - Que ambas integrantes graben las mismas senas
"""

# ============================================================
# IMPORTS - Librerias que necesitamos
# ============================================================
# Cada "import" trae herramientas que alguien mas ya programo
# para que nosotras no tengamos que hacerlo desde cero.

import os       # Para trabajar con archivos y carpetas
import sys      # Para cerrar el programa si hay un error
import time     # Para medir el tiempo (countdown, duracion de grabacion)
import argparse # Para leer opciones de la terminal (ej: --camera 1)
import math     # Para funciones matematicas (ej: redondear el countdown)

import cv2          # OpenCV: abre la camara y muestra video en pantalla
import mediapipe as mp  # MediaPipe (Google): detecta las manos con IA
import pandas as pd     # Pandas: guarda datos en tablas (CSV)
import numpy as np      # NumPy: trabaja con numeros y arreglos

# ============================================================
# CONFIGURACION DE SENAS
# ============================================================
# Este diccionario conecta cada tecla del teclado con una sena.
# ord('1') convierte el caracter '1' a su codigo numerico (49).
# Cuando presionas '1' en el teclado, la compu sabe que quieres
# grabar "Hola".

SIGNS = {
    ord('1'): 'Hola',
    ord('2'): 'Gracias',
    ord('3'): 'Por favor',
    ord('4'): 'Si',
    ord('5'): 'No',
    ord('6'): 'Ayuda',
    ord('7'): 'Agua',
    ord('8'): 'Comida',
    ord('9'): 'Bano',
    ord('0'): 'Doctor',
    ord('a'): 'Familia',
    ord('b'): 'Amigo',
    ord('c'): 'Disculpa',
    ord('d'): 'Te quiero',
}

RECORD_DURATION = 15  # Cuantos segundos dura cada grabacion
COUNTDOWN_DURATION = 3  # Segundos de cuenta regresiva antes de grabar

# ============================================================
# NOMBRES DE COLUMNAS DEL CSV
# ============================================================
# Cada mano tiene 21 puntos (landmarks), y cada punto tiene 3
# coordenadas: x (horizontal), y (vertical), z (profundidad).
#
# L = Left (mano izquierda), R = Right (mano derecha)
# L0_x = coordenada X del punto 0 de la mano izquierda
# R20_z = coordenada Z del punto 20 de la mano derecha
#
# Total: 21 puntos x 3 coordenadas x 2 manos = 126 numeros por frame
# + 1 columna "label" (nombre de la sena) = 127 columnas

COLUMNS = []
for hand_prefix in ['L', 'R']:       # L = izquierda, R = derecha
    for i in range(21):               # 21 puntos por mano (0 al 20)
        for coord in ['x', 'y', 'z']: # 3 coordenadas por punto
            COLUMNS.append(f'{hand_prefix}{i}_{coord}')
COLUMNS.append('label')  # Ultima columna: nombre de la sena

# Rutas de archivos (se calculan automaticamente segun donde este este script)
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'landmarks.csv')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def get_sample_counts():
    """Cuenta cuantas muestras hay guardadas de cada sena en el CSV.

    Esto nos ayuda a saber cuantos datos tenemos y cuales senas
    necesitan mas grabaciones.
    """
    if not os.path.exists(CSV_PATH):
        return {}  # Si no existe el CSV, no hay datos aun
    try:
        df = pd.read_csv(CSV_PATH)
        # value_counts() cuenta cuantas veces aparece cada sena
        return df['label'].value_counts().to_dict()
    except Exception:
        return {}


def extract_landmarks(result):
    """Extrae las coordenadas de los dedos del resultado de MediaPipe.

    MediaPipe detecta 21 puntos en cada mano:
      - Punto 0: muneca
      - Puntos 1-4: dedo pulgar
      - Puntos 5-8: dedo indice
      - Puntos 9-12: dedo medio
      - Puntos 13-16: dedo anular
      - Puntos 17-20: dedo menique

    Cada punto tiene 3 coordenadas (x, y, z), donde:
      - x: posicion horizontal (0.0 = izquierda, 1.0 = derecha)
      - y: posicion vertical (0.0 = arriba, 1.0 = abajo)
      - z: profundidad (que tan cerca o lejos esta del lente)

    Si solo se detecta UNA mano, la otra se llena con ceros.
    Esto se llama "zero-padding" y es importante para que todos
    los datos tengan el mismo tamano (126 numeros siempre).

    Retorna: lista de 126 numeros [L0_x, L0_y, L0_z, ..., R20_z]
    """
    # Empezamos con todo en ceros (por si no se detecta alguna mano)
    left_hand = [0.0] * 63   # 21 puntos x 3 coordenadas = 63
    right_hand = [0.0] * 63

    if result.hand_landmarks:
        # Recorremos cada mano detectada
        for hand_landmarks, handedness_list in zip(
            result.hand_landmarks,
            result.handedness
        ):
            # MediaPipe nos dice si la mano es "Left" o "Right"
            label = handedness_list[0].category_name

            # Extraemos las 63 coordenadas de esta mano
            coords = []
            for lm in hand_landmarks:
                coords.extend([lm.x, lm.y, lm.z])

            # Guardamos en el lugar correcto
            if label == 'Left':
                left_hand = coords
            else:
                right_hand = coords

    # Juntamos izquierda + derecha en una sola lista de 126 numeros
    return left_hand + right_hand


# ============================================================
# FUNCIONES DE DIBUJO EN PANTALLA
# ============================================================
# Estas funciones dibujan cosas sobre el video de la camara
# para que sea mas facil usar el programa.

def draw_overlay(frame, text_lines, position='top', color=(255, 255, 255),
                 bg_alpha=0.6, font_scale=0.5):
    """Dibuja texto con fondo semi-transparente sobre el video.

    Ejemplo: el menu de senas que aparece arriba de la pantalla.
    """
    h, w = frame.shape[:2]  # Altura y ancho del video
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_height = int(25 * font_scale / 0.5)
    padding = 8

    # Calcular tamano del fondo
    total_height = len(text_lines) * line_height + padding * 2
    overlay = frame.copy()

    # Posicion: arriba o abajo de la pantalla
    if position == 'top':
        y_start = 0
    elif position == 'bottom':
        y_start = h - total_height
    else:
        y_start = 0

    # Dibujar rectangulo negro semi-transparente como fondo
    cv2.rectangle(overlay, (0, y_start), (w, y_start + total_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

    # Escribir cada linea de texto
    for i, line in enumerate(text_lines):
        y = y_start + padding + (i + 1) * line_height - 5
        cv2.putText(frame, line, (10, y), font, font_scale, color,
                    thickness, cv2.LINE_AA)


def draw_progress_bar(frame, progress, y_pos=None):
    """Dibuja una barra de progreso verde en la parte inferior.

    'progress' es un numero entre 0.0 (vacia) y 1.0 (llena).
    """
    h, w = frame.shape[:2]
    if y_pos is None:
        y_pos = h - 30
    bar_width = w - 40
    bar_height = 20

    # Fondo gris de la barra
    cv2.rectangle(frame, (20, y_pos), (20 + bar_width, y_pos + bar_height),
                  (50, 50, 50), -1)
    # Parte verde (progreso)
    fill_width = int(bar_width * min(progress, 1.0))
    cv2.rectangle(frame, (20, y_pos), (20 + fill_width, y_pos + bar_height),
                  (0, 200, 0), -1)
    # Borde blanco
    cv2.rectangle(frame, (20, y_pos), (20 + bar_width, y_pos + bar_height),
                  (255, 255, 255), 1)


def draw_recording_border(frame):
    """Dibuja un borde ROJO alrededor del video para indicar que esta grabando."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)


def draw_center_text(frame, text, color=(0, 255, 255), scale=2.0):
    """Dibuja texto grande y centrado en la pantalla.

    Se usa para el countdown ("Hola en 3...") y mensajes importantes.
    El texto tiene una sombra negra detras para que se lea mejor.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 3
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - text_w) // 2  # Centrar horizontalmente
    y = (h + text_h) // 2  # Centrar verticalmente

    # Sombra negra (un poco desplazada)
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    # Texto principal con el color elegido
    cv2.putText(frame, text, (x, y), font, scale, color,
                thickness, cv2.LINE_AA)


def draw_hand_landmarks(frame, result):
    """Dibuja los puntos y conexiones de las manos sobre el video.

    Esto es lo que hace que vean los puntitos de colores sobre sus
    manos cuando estan frente a la camara. Usa funciones de MediaPipe
    para dibujarlo automaticamente.
    """
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


# ============================================================
# FUNCION PRINCIPAL - Aqui es donde todo sucede
# ============================================================

def main():
    # --- Leer opciones de la terminal ---
    # Si tienen mas de una camara, pueden escribir: python collect_data.py --camera 1
    parser = argparse.ArgumentParser(description='Recoleccion de datos LSM')
    parser.add_argument('--camera', type=int, default=0,
                        help='Indice de la camara (default: 0)')
    args = parser.parse_args()

    # --- Verificar que existe el modelo de MediaPipe ---
    # Este archivo (.task) es el "cerebro" que detecta las manos.
    # Se descarga una sola vez y pesa ~7.5 MB.
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontro el modelo de MediaPipe: {MODEL_PATH}")
        print("Descargalo con:")
        print(f'  curl -sL -o "{MODEL_PATH}" '
              '"https://storage.googleapis.com/mediapipe-models/'
              'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"')
        sys.exit(1)

    # --- Inicializar MediaPipe HandLandmarker ---
    # Configuramos el detector de manos con estos parametros:
    #   - num_hands=2: detectar hasta 2 manos al mismo tiempo
    #   - min_hand_detection_confidence=0.7: solo detectar manos si esta
    #     70% seguro (si ponen 0.5 detecta mas pero con mas errores)
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,  # Modo video (no imagen individual)
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

    # Configurar resolucion del video (640x480 es estandar)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camara abierta. Ventana de recoleccion de datos LSM lista.")
    print("Presiona el numero/letra de una sena para grabar, 'q' para salir.")

    # --- Variables de estado ---
    # El programa funciona como una "maquina de estados":
    #   menu     -> Esperando que presionen una tecla
    #   countdown -> Cuenta regresiva de 3 segundos
    #   recording -> Grabando landmarks durante 15 segundos
    #   saving   -> Mostrando mensaje de "Guardado!" por 2 segundos
    state = 'menu'
    current_sign = None      # Que sena estamos grabando
    countdown_start = 0      # Cuando empezo la cuenta regresiva
    record_start = 0         # Cuando empezo la grabacion
    frames_data = []         # Lista donde se acumulan los landmarks
    save_start = 0           # Cuando termino de guardar
    saved_count = 0          # Cuantos frames se guardaron
    frame_timestamp = 0      # Timestamp para MediaPipe (va incrementando)

    # Cargar conteos de grabaciones previas
    sample_counts = get_sample_counts()

    # ============================================================
    # LOOP PRINCIPAL - Se repite ~30 veces por segundo
    # ============================================================
    while True:
        # Leer un frame (imagen) de la camara
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer frame de la camara")
            break

        # Voltear la imagen como espejo (selfie mode)
        # Sin esto, si mueves la mano a la derecha, en la pantalla
        # se veria a la izquierda, lo cual es confuso.
        frame = cv2.flip(frame, 1)

        # --- Detectar manos con MediaPipe ---
        # OpenCV usa colores BGR (azul-verde-rojo) pero MediaPipe
        # necesita RGB (rojo-verde-azul), asi que convertimos.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp += 33  # Simulamos ~30fps (1000ms / 30 = 33ms)
        result = hand_landmarker.detect_for_video(mp_image, frame_timestamp)

        # Dibujar los puntitos de colores sobre las manos
        draw_hand_landmarks(frame, result)

        # Verificar si se detectaron manos en este frame
        has_hands = len(result.hand_landmarks) > 0

        # ==========================================================
        # ESTADO: MENU - Esperando que la usuaria presione una tecla
        # ==========================================================
        if state == 'menu':
            # Mostrar el menu de senas disponibles
            menu_lines = [
                "=== Recoleccion de Datos LSM - Mira&Tek ===",
                "[1] Hola   [2] Gracias   [3] Por favor   [4] Si   [5] No",
                "[6] Ayuda  [7] Agua  [8] Comida  [9] Bano  [0] Doctor",
                "[a] Familia  [b] Amigo  [c] Disculpa  [d] Te quiero",
            ]
            draw_overlay(frame, menu_lines, position='top', color=(0, 255, 0))

            # Mostrar conteo de muestras en la parte inferior
            total = sum(sample_counts.values()) if sample_counts else 0
            count_lines = [f"Total muestras: {total} | Presiona tecla para grabar | 'q' para salir"]
            if sample_counts:
                counts_str = "  ".join(
                    f"{sign}: {sample_counts.get(sign, 0)}"
                    for sign in sorted(set(SIGNS.values()))
                )
                count_lines.append(counts_str)
            draw_overlay(frame, count_lines, position='bottom',
                         color=(200, 200, 200), font_scale=0.4)

            # Esperar tecla
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break  # Salir del programa
            elif key in SIGNS:
                # La usuaria presiono una tecla valida -> iniciar countdown
                current_sign = SIGNS[key]
                countdown_start = time.time()
                state = 'countdown'

        # ==========================================================
        # ESTADO: COUNTDOWN - Cuenta regresiva de 3 segundos
        # ==========================================================
        elif state == 'countdown':
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_DURATION - elapsed

            if remaining > 0:
                # Mostrar "Hola en 3...", "Hola en 2...", "Hola en 1..."
                draw_center_text(frame,
                                 f"{current_sign} en {math.ceil(remaining)}...",
                                 color=(0, 255, 255), scale=1.5)
                draw_overlay(frame,
                             ["Prepara la sena frente a la camara..."],
                             position='top', color=(0, 255, 255))
            else:
                # Se acabaron los 3 segundos -> empezar a grabar
                record_start = time.time()
                frames_data = []  # Lista vacia para los nuevos datos
                state = 'recording'

            cv2.waitKey(1)

        # ==========================================================
        # ESTADO: RECORDING - Grabando landmarks durante 15 segundos
        # ==========================================================
        elif state == 'recording':
            elapsed = time.time() - record_start
            remaining = RECORD_DURATION - elapsed

            if remaining > 0:
                # Si se detectaron manos, guardar los landmarks de este frame
                if has_hands:
                    landmarks = extract_landmarks(result)
                    # Agregamos los 126 numeros + el nombre de la sena
                    frames_data.append(landmarks + [current_sign])

                # Mostrar indicadores visuales
                draw_recording_border(frame)  # Borde rojo = grabando
                status_color = (0, 0, 255) if has_hands else (0, 165, 255)
                draw_overlay(
                    frame,
                    [f"GRABANDO: {current_sign} | {remaining:.1f}s | "
                     f"Frames: {len(frames_data)}"],
                    position='top', color=status_color,
                )
                # Advertencia si no se detectan manos
                if not has_hands:
                    draw_center_text(frame, "Sin manos detectadas",
                                     color=(0, 0, 255), scale=0.8)

                # Barra de progreso verde
                draw_progress_bar(frame, elapsed / RECORD_DURATION)
            else:
                # Se acabaron los 15 segundos -> guardar datos al CSV
                state = 'saving'
                save_start = time.time()
                saved_count = len(frames_data)

                if frames_data:
                    # Crear tabla con los datos y guardar al CSV
                    df = pd.DataFrame(frames_data, columns=COLUMNS)
                    file_exists = os.path.exists(CSV_PATH)
                    # mode='a' significa "append" (agregar al final, no sobreescribir)
                    # header=not file_exists: solo poner encabezados si el archivo es nuevo
                    df.to_csv(CSV_PATH, mode='a', header=not file_exists,
                              index=False)
                    # Actualizar conteo
                    sample_counts[current_sign] = (
                        sample_counts.get(current_sign, 0) + saved_count
                    )
                    print(f"Guardado: {saved_count} frames de '{current_sign}' "
                          f"-> {CSV_PATH}")
                else:
                    print(f"Advertencia: 0 frames capturados para '{current_sign}'")

            cv2.waitKey(1)

        # ==========================================================
        # ESTADO: SAVING - Mostrando confirmacion por 2 segundos
        # ==========================================================
        elif state == 'saving':
            elapsed = time.time() - save_start
            if elapsed < 2.0:
                if saved_count > 0:
                    draw_center_text(
                        frame,
                        f"Guardado: {saved_count} frames!",
                        color=(0, 255, 0), scale=0.8,
                    )
                else:
                    draw_center_text(
                        frame,
                        "0 frames - intenta de nuevo",
                        color=(0, 0, 255), scale=0.8,
                    )
                cv2.waitKey(1)
            else:
                # Despues de 2 segundos, volver al menu
                state = 'menu'

        # Mostrar el frame en la ventana
        cv2.imshow('LSM Data Collection - Mira&Tek', frame)

    # --- Limpiar al salir ---
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    # --- Mostrar resumen de todo lo que se grabo ---
    print("\n=== Resumen de datos recolectados ===")
    counts = get_sample_counts()
    if counts:
        for sign, count in sorted(counts.items()):
            print(f"  {sign}: {count} muestras")
        print(f"  TOTAL: {sum(counts.values())} muestras")
    else:
        print("  No se recolectaron datos.")


if __name__ == '__main__':
    main()
