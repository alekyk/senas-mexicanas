"""
Mira&Tek - Arquitectura del Modelo MLP para clasificacion de LSM
Technovation Girls 2026

=== QUE HACE ESTE ARCHIVO? ===
Define la "estructura" de nuestra red neuronal (el cerebro de la IA).

Imaginen la red neuronal como un embudo inteligente:
  - Entran 126 numeros (las posiciones de los dedos)
  - Pasan por capas de "neuronas" que aprenden patrones
  - Sale 1 respuesta: cual sena es (Hola, Gracias, etc.)

=== QUE ES UN MLP? ===
MLP = Multi-Layer Perceptron (Perceptron Multicapa)
Es el tipo mas basico de red neuronal. Funciona asi:

  Capa 1: Recibe los 126 numeros y los transforma en 128 numeros
           (encuentra patrones simples como "los dedos estan cerrados")

  Capa 2: Toma esos 128 numeros y los reduce a 64
           (combina patrones: "dedos cerrados + mano arriba = Hola")

  Capa 3: Toma los 64 numeros y produce 14 (una por cada sena)
           (el numero mas alto indica la sena detectada)
"""

import torch
import torch.nn as nn


class SignLanguageMLP(nn.Module):
    """Red neuronal para clasificar senas de Lengua de Senas Mexicana.

    Arquitectura (como esta construida):

        Entrada: 126 numeros (posiciones de los dedos)
            |
            v
        Capa 1: 126 -> 128 neuronas
          + BatchNorm (estabiliza el aprendizaje)
          + ReLU (funcion de activacion: solo deja pasar numeros positivos)
          + Dropout 30% (apaga neuronas al azar para evitar memorizacion)
            |
            v
        Capa 2: 128 -> 64 neuronas
          + BatchNorm
          + ReLU
          + Dropout 30%
            |
            v
        Salida: 64 -> 14 numeros (uno por cada sena)

    Que significan las partes:
      - Linear: conexion entre capas (multiplica numeros por "pesos")
      - BatchNorm: normaliza los datos para que el modelo aprenda mas rapido
      - ReLU: si el numero es negativo lo convierte a 0, si es positivo lo deja
      - Dropout: apaga el 30% de las neuronas al azar durante el entrenamiento.
                 Esto EVITA que el modelo "memorice" los datos en vez de aprender
                 patrones generales. Es como estudiar con diferentes resumenes
                 en vez de memorizar uno solo.
    """

    def __init__(self, input_size=126, hidden1=128, hidden2=64, num_classes=14):
        """Construye las capas de la red neuronal.

        Args:
            input_size: cuantos numeros entran (126 = landmarks de las manos)
            hidden1: neuronas en la primera capa oculta (128)
            hidden2: neuronas en la segunda capa oculta (64)
            num_classes: cuantas senas diferentes queremos reconocer (14)
        """
        super().__init__()

        # nn.Sequential ejecuta estas capas una tras otra, en orden
        self.network = nn.Sequential(
            # --- Capa 1: de 126 a 128 neuronas ---
            nn.Linear(input_size, hidden1),   # Conexion con pesos
            nn.BatchNorm1d(hidden1),          # Estabiliza el aprendizaje
            nn.ReLU(),                         # Activacion (quita negativos)
            nn.Dropout(0.3),                   # Apaga 30% de neuronas al azar

            # --- Capa 2: de 128 a 64 neuronas ---
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),

            # --- Capa de salida: de 64 a N senas ---
            nn.Linear(hidden2, num_classes),
            # No ponemos ReLU aqui porque la salida puede ser cualquier numero.
            # El numero mas alto indica la sena que el modelo cree que es.
        )

    def forward(self, x):
        """Procesa los datos a traves de todas las capas.

        Esta funcion se llama automaticamente cuando hacemos model(datos).
        'x' son los 126 numeros de landmarks que entran al modelo.
        """
        return self.network(x)
