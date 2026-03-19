"""
Mira&Tek - Data Augmentation para dataset de LSM
Technovation Girls 2026

=== QUE HACE ESTE ARCHIVO? ===
Multiplica los datos de entrenamiento para que el modelo aprenda mejor.

Imaginen que solo tienen 100 fotos de la sena "Hola". Con data augmentation,
podemos crear 600 versiones diferentes a partir de esas 100 originales.
Mas datos = el modelo aprende mejor y se equivoca menos.

=== QUE TECNICAS USAMOS? ===

1. ESPEJEO (Mirror):
   Si grabaste "Hola" con la mano derecha, creamos una copia
   como si lo hubieras hecho con la izquierda (como verte en un espejo).
   Esto DUPLICA los datos (100 -> 200).

2. JITTER (Ruido):
   Agregamos variaciones muy pequenitas a los numeros, como si la mano
   temblara un poquito. Esto hace que el modelo sea mas tolerante a
   imprecisiones. Creamos 2 copias con jitter del conjunto espejado.

   Resultado final: 100 originales -> 600 muestras (6 veces mas!)
"""

import pandas as pd
import numpy as np


# ============================================================
# NOMBRES DE COLUMNAS
# ============================================================
# Estas listas definen que columnas del CSV pertenecen a cada mano.
# Las usamos para saber que datos intercambiar durante el espejeo.

# Columnas de la mano izquierda: L0_x, L0_y, L0_z, L1_x, ..., L20_z
LEFT_COLS = [f'L{i}_{c}' for i in range(21) for c in ['x', 'y', 'z']]

# Columnas de la mano derecha: R0_x, R0_y, R0_z, R1_x, ..., R20_z
RIGHT_COLS = [f'R{i}_{c}' for i in range(21) for c in ['x', 'y', 'z']]

# Solo las columnas X (horizontal) - necesarias para invertir en el espejeo
X_COLS = [c for c in LEFT_COLS + RIGHT_COLS if c.endswith('_x')]

# Todas las columnas de datos (126 en total, sin la columna "label")
FEATURE_COLS = LEFT_COLS + RIGHT_COLS


# ============================================================
# TECNICA 1: ESPEJEO (MIRROR)
# ============================================================

def mirror_augment(df):
    """Crea versiones espejadas de cada muestra.

    Que hace paso a paso:
      1. Intercambia los datos de mano izquierda <-> derecha
         (lo que estaba en columnas L pasa a R, y viceversa)
      2. Invierte las coordenadas X (horizontal)
         x_nueva = 1.0 - x_original
         (si la mano estaba a la derecha, ahora aparece a la izquierda)

    Ejemplo visual:
      Original:  mano derecha haciendo "Hola" a la derecha de la pantalla
      Espejado:  mano izquierda haciendo "Hola" a la izquierda de la pantalla

    Nota importante: si una mano no fue detectada (tiene valores 0.0),
    NO le invertimos la X, porque 1.0 - 0.0 = 1.0 y eso seria incorrecto.
    Los ceros deben quedarse como ceros.
    """
    mirrored = df.copy()

    # Paso 1: Intercambiar columnas izquierda <-> derecha
    mirrored[LEFT_COLS] = df[RIGHT_COLS].values
    mirrored[RIGHT_COLS] = df[LEFT_COLS].values

    # Paso 2: Encontrar cuales manos tienen datos reales (no son todo ceros)
    left_has_data = (mirrored[LEFT_COLS] != 0.0).any(axis=1)
    right_has_data = (mirrored[RIGHT_COLS] != 0.0).any(axis=1)

    # Paso 3: Invertir las coordenadas X solo donde hay datos
    left_x_cols = [c for c in LEFT_COLS if c.endswith('_x')]
    right_x_cols = [c for c in RIGHT_COLS if c.endswith('_x')]

    for col in left_x_cols:
        mirrored.loc[left_has_data, col] = 1.0 - mirrored.loc[left_has_data, col]
    for col in right_x_cols:
        mirrored.loc[right_has_data, col] = 1.0 - mirrored.loc[right_has_data, col]

    return mirrored


# ============================================================
# TECNICA 2: JITTER (RUIDO GAUSSIANO)
# ============================================================

def jitter_augment(df, noise_std=0.02, n_copies=2):
    """Crea copias con pequenas variaciones aleatorias.

    Imaginense que mueven la mano un milimetro a la izquierda, o un milimetro
    arriba. La sena sigue siendo la misma, pero los numeros cambian un poquito.
    Eso es lo que simulamos aqui: variaciones naturales del movimiento humano.

    Args:
        df: tabla con los datos de landmarks
        noise_std: que tan grande es la variacion (0.02 = 2% de cambio)
                   Valores mas grandes = variaciones mas grandes
        n_copies: cuantas copias con ruido crear (default: 2)

    Nota: solo agrega ruido a valores que NO son cero.
    Los ceros significan "mano no detectada" y deben quedarse en cero.
    """
    augmented_list = []

    for _ in range(n_copies):
        noisy = df.copy()

        # Generar numeros aleatorios pequenos (distribucion gaussiana/normal)
        # La mayoria seran cercanos a 0, pocos seran mas grandes
        noise = np.random.normal(0, noise_std, size=(len(noisy), len(FEATURE_COLS)))

        # Crear mascara: True donde hay datos, False donde hay ceros
        feature_values = noisy[FEATURE_COLS].values
        mask = feature_values != 0.0

        # Sumar el ruido SOLO donde hay datos reales
        # mask hace que noise * False = 0 (no cambia los ceros)
        feature_values = feature_values + noise * mask
        noisy[FEATURE_COLS] = feature_values

        augmented_list.append(noisy)

    # Juntar todas las copias en una sola tabla
    return pd.concat(augmented_list, ignore_index=True)


# ============================================================
# FUNCION PRINCIPAL: COMBINAR AMBAS TECNICAS
# ============================================================

def augment_dataframe(df):
    """Aplica espejeo + jitter para multiplicar los datos ~6 veces.

    Proceso:
      1. Datos originales: 100 muestras
      2. + Espejeo: 100 originales + 100 espejadas = 200
      3. + Jitter x2 copias del conjunto de 200 = 400 adicionales
      4. Total: 200 + 400 = 600 muestras (6x)

    Esto es CRUCIAL para que el modelo funcione bien con pocos datos.
    Sin augmentation, 100 muestras no serian suficientes para entrenar.
    Con augmentation, esas 100 se convierten en 600 y el modelo aprende
    patrones mas generales.
    """
    original_size = len(df)

    # Paso 1: Crear versiones espejadas
    mirrored = mirror_augment(df)
    combined = pd.concat([df, mirrored], ignore_index=True)
    # Ahora tenemos 2x datos

    # Paso 2: Crear copias con ruido del conjunto combinado
    jittered = jitter_augment(combined, noise_std=0.02, n_copies=2)
    final = pd.concat([combined, jittered], ignore_index=True)
    # Ahora tenemos 6x datos

    print(f"Data augmentation: {original_size} -> {len(final)} muestras "
          f"({len(final)/original_size:.1f}x)")

    return final
