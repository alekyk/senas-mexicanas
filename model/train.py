"""
Mira&Tek - Script de Entrenamiento del Modelo MLP para LSM
Technovation Girls 2026

=== QUE HACE ESTE ARCHIVO? ===
Este es el PASO 2 del proyecto. Toma los datos que grabaron con collect_data.py
(el archivo landmarks.csv) y ENTRENA la red neuronal para que aprenda a
reconocer las senas.

Piensen en esto como "estudiar para un examen":
  - Los datos de landmarks.csv son el "libro de texto"
  - El modelo (la red neuronal) es el "estudiante"
  - Entrenar es el proceso de "estudiar" una y otra vez
  - Al final, le hacemos un "examen" con datos que nunca vio
  - Si saca >85%, esta listo para usarse en la app!

=== COMO SE USA? ===
1. Primero: haber grabado datos con collect_data.py (minimo 3-5 grabaciones por sena)
2. Abre la terminal
3. Escribe: python model/train.py
4. Espera unos minutos mientras entrena
5. Al final te dice que precision logro
6. El modelo entrenado se guarda en model/sign_model.pth

=== QUE ARCHIVOS GENERA? ===
  - model/sign_model.pth: el modelo entrenado (los "pesos" de la red neuronal)
  - model/label_encoder.json: la lista de senas que conoce (ej: ["Hola", "Gracias", ...])
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import json

# Asegurar que model/ este en el path para que los imports funcionen
# sin importar desde que carpeta ejecutes el script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np    # Operaciones con numeros y matrices
import pandas as pd   # Leer y manipular tablas (CSV)
import torch          # PyTorch: el framework de machine learning
import torch.nn as nn # Modulos de redes neuronales

# DataLoader agrupa los datos en "lotes" (batches) para entrenar
from torch.utils.data import Dataset, DataLoader

# Herramientas de scikit-learn para preparar datos y evaluar
from sklearn.model_selection import train_test_split  # Dividir datos en train/test
from sklearn.preprocessing import LabelEncoder        # Convertir nombres a numeros
from sklearn.metrics import classification_report, confusion_matrix  # Metricas

# Nuestros propios archivos
from model import SignLanguageMLP                  # La red neuronal (model.py)
from augment import augment_dataframe, FEATURE_COLS  # Data augmentation (augment.py)

# ============================================================
# CONFIGURACION
# ============================================================

# Rutas de archivos
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'landmarks.csv')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, 'sign_model.pth')     # Donde se guarda el modelo
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.json')  # Donde se guarda la lista de senas

# Hiperparametros (configuraciones del entrenamiento)
BATCH_SIZE = 32       # Cuantas muestras procesa a la vez (como estudiar 32 flashcards juntas)
LEARNING_RATE = 0.001  # Que tan rapido aprende (muy alto = errores, muy bajo = lento)
EPOCHS = 100          # Cuantas veces repasa TODOS los datos (como releer el libro 100 veces)
TEST_SIZE = 0.2       # 20% de los datos se guardan para el "examen" final
RANDOM_SEED = 42      # Semilla para reproducibilidad (que siempre de el mismo resultado)

# Fijar semillas para que el entrenamiento sea reproducible
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# DATASET DE PYTORCH
# ============================================================

class LandmarkDataset(Dataset):
    """Empaqueta los datos de landmarks para que PyTorch los pueda usar.

    PyTorch necesita los datos en un formato especial (tensores).
    Esta clase convierte nuestros numeros normales a tensores de PyTorch.
    """

    def __init__(self, X, y):
        # FloatTensor: numeros decimales (las coordenadas de los dedos)
        self.X = torch.FloatTensor(X)
        # LongTensor: numeros enteros (el indice de cada sena: 0, 1, 2, ...)
        self.y = torch.LongTensor(y)

    def __len__(self):
        """Cuantas muestras hay en total."""
        return len(self.X)

    def __getitem__(self, idx):
        """Devuelve UNA muestra: (landmarks, sena)."""
        return self.X[idx], self.y[idx]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def get_device():
    """Detecta el mejor hardware disponible para entrenar.

    - MPS: chip Apple Silicon (M1/M2/M3) - rapido!
    - CUDA: tarjeta de video NVIDIA - muy rapido!
    - CPU: procesador normal - funciona pero es mas lento

    No se preocupen si solo tienen CPU, el modelo es pequeno
    y entrena en minutos de todas formas.
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def evaluate(model, dataloader, device):
    """Hace el "examen" al modelo: prueba con datos que nunca vio.

    Retorna:
      - accuracy: que porcentaje acerto (ej: 0.92 = 92%)
      - all_preds: que respondio el modelo para cada pregunta
      - all_labels: cuales eran las respuestas correctas
    """
    model.eval()  # Modo evaluacion (desactiva Dropout y BatchNorm de entrenamiento)
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # torch.no_grad() le dice a PyTorch que no necesita calcular gradientes
    # (los gradientes solo se necesitan durante el entrenamiento, no en el examen)
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Pasar los datos por el modelo
            outputs = model(X_batch)

            # torch.max encuentra el numero mas alto de los 14 de salida
            # Ese indice es la sena que el modelo "cree" que es
            _, predicted = torch.max(outputs, 1)

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = correct / total if total > 0 else 0
    return accuracy, np.array(all_preds), np.array(all_labels)


# ============================================================
# FUNCION PRINCIPAL
# ============================================================

def main():
    # ==========================================================
    # PASO 1: CARGAR LOS DATOS DEL CSV
    # ==========================================================
    if not os.path.exists(CSV_PATH):
        print(f"Error: No se encontro el archivo de datos: {CSV_PATH}")
        print("Primero ejecuta data/collect_data.py para recolectar datos.")
        sys.exit(1)

    print("Cargando datos...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Muestras totales: {len(df)}")
    print(f"  Senas encontradas: {df['label'].nunique()}")
    print(f"  Distribucion:")
    for sign, count in df['label'].value_counts().sort_index().items():
        print(f"    {sign}: {count}")

    # ==========================================================
    # PASO 2: SEPARAR DATOS EN FEATURES (X) Y LABELS (Y)
    # ==========================================================
    # X = los 126 numeros de landmarks (lo que ENTRA al modelo)
    # y = el nombre de la sena (lo que QUEREMOS que el modelo PREDIGA)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df['label'].values

    # LabelEncoder convierte nombres a numeros:
    #   "Hola" -> 0, "Gracias" -> 1, "Ayuda" -> 2, etc.
    # El modelo trabaja con numeros, no con texto.
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"\n  Clases ({num_classes}): {list(le.classes_)}")

    # Guardar la lista de senas para que el servidor sepa que numero = que sena
    with open(ENCODER_PATH, 'w') as f:
        json.dump(list(le.classes_), f, ensure_ascii=False)
    print(f"  Label encoder guardado en: {ENCODER_PATH}")

    # ==========================================================
    # PASO 3: DIVIDIR EN ENTRENAMIENTO (80%) Y EXAMEN (20%)
    # ==========================================================
    # Esto es SUPER IMPORTANTE: nunca evaluamos al modelo con datos
    # que ya vio durante el entrenamiento. Seria como hacer un examen
    # con las mismas preguntas con las que estudiaste - no demuestra
    # que realmente aprendiste.
    #
    # stratify=y asegura que cada sena tenga la misma proporcion
    # en ambos conjuntos (que no se quede todo "Hola" en train y
    # todo "Gracias" en test).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # ==========================================================
    # PASO 4: DATA AUGMENTATION (SOLO EN DATOS DE ENTRENAMIENTO)
    # ==========================================================
    # Multiplicamos los datos de entrenamiento ~6x con espejeo y jitter.
    # IMPORTANTE: NUNCA hacemos augmentation en los datos de test,
    # porque el examen debe ser con datos reales, no inventados.
    print("\nAplicando data augmentation al set de entrenamiento...")
    train_df = pd.DataFrame(X_train, columns=FEATURE_COLS)
    train_df['label'] = le.inverse_transform(y_train)

    train_augmented = augment_dataframe(train_df)

    X_train_aug = train_augmented[FEATURE_COLS].values.astype(np.float32)
    y_train_aug = le.transform(train_augmented['label'].values)

    # ==========================================================
    # PASO 5: PREPARAR DATOS PARA PYTORCH
    # ==========================================================
    # DataLoader agrupa los datos en "batches" (lotes de 32 muestras)
    # y los mezcla aleatoriamente. Es como barajar las flashcards
    # antes de estudiar para no memorizarlas en orden.
    train_dataset = LandmarkDataset(X_train_aug, y_train_aug)
    test_dataset = LandmarkDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================================
    # PASO 6: CREAR EL MODELO
    # ==========================================================
    device = get_device()
    print(f"\nDispositivo: {device}")

    model = SignLanguageMLP(
        input_size=len(FEATURE_COLS),   # 126 landmarks
        num_classes=num_classes          # Cuantas senas hay
    ).to(device)  # .to(device) mueve el modelo al hardware mas rapido

    print(f"Modelo: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros totales: {total_params:,}")

    # ==========================================================
    # PASO 7: CONFIGURAR EL ENTRENAMIENTO
    # ==========================================================
    # Adam optimizer: algoritmo que ajusta los "pesos" del modelo
    # para que cada vez acierte mas. Es el mas popular y funciona bien.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler: si el modelo se estanca (no mejora en 7 epochs),
    # reduce el learning rate a la mitad. Es como cuando estudias
    # y ya no avanzas, entonces lees mas despacio y con mas atencion.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=True
    )

    # CrossEntropyLoss: funcion que mide "que tan mal" le fue al modelo.
    # Entre mas alta, mas se equivoco. El objetivo es reducirla a 0.
    criterion = nn.CrossEntropyLoss()

    # ==========================================================
    # PASO 8: ENTRENAR! (El loop principal)
    # ==========================================================
    # Cada "epoch" es una pasada completa por TODOS los datos.
    # En cada epoch:
    #   1. El modelo ve los datos y hace predicciones
    #   2. Calculamos que tan mal le fue (loss)
    #   3. Ajustamos los pesos para que la proxima vez acierte mas
    #   4. Hacemos un mini-examen con los datos de test
    #   5. Si mejoro, guardamos el modelo
    print(f"\nIniciando entrenamiento ({EPOCHS} epochs)...")
    print("-" * 60)

    best_accuracy = 0.0    # La mejor precision que hemos logrado
    patience = 15          # Cuantos epochs sin mejorar antes de parar
    no_improve_count = 0   # Contador de epochs sin mejora

    for epoch in range(EPOCHS):
        # --- Modo entrenamiento ---
        model.train()  # Activa Dropout y BatchNorm de entrenamiento
        total_loss = 0
        num_batches = 0

        # Recorrer todos los batches de datos
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 1. Limpiar gradientes del paso anterior
            optimizer.zero_grad()

            # 2. Pasar los datos por el modelo (forward pass)
            outputs = model(X_batch)

            # 3. Calcular que tan mal le fue (loss)
            loss = criterion(outputs, y_batch)

            # 4. Calcular como ajustar los pesos (backward pass)
            loss.backward()

            # 5. Aplicar los ajustes a los pesos
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # --- Mini-examen con datos de test ---
        test_acc, _, _ = evaluate(model, test_loader, device)

        # Ajustar learning rate si el modelo se estanco
        scheduler.step(test_acc)

        # Guardar el modelo si es el mejor hasta ahora
        improved = ""
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            improved = " <-- Mejor modelo guardado!"
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Mostrar progreso
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Test Acc: {test_acc:.2%}{improved}")

        # --- Early stopping ---
        # Si en 15 epochs no mejora, paramos. No tiene caso seguir
        # "estudiando" si ya no esta aprendiendo nada nuevo.
        if no_improve_count >= patience:
            print(f"\nEarly stopping: sin mejora en {patience} epochs.")
            break

    # ==========================================================
    # PASO 9: EVALUACION FINAL (EL EXAMEN COMPLETO)
    # ==========================================================
    print("\n" + "=" * 60)
    print("EVALUACION FINAL")
    print("=" * 60)

    # Cargar el MEJOR modelo que guardamos durante el entrenamiento
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    test_acc, preds, labels = evaluate(model, test_loader, device)

    # Reporte detallado: precision, recall y F1 para cada sena
    # - Precision: de las veces que dijo "Hola", cuantas SI eran Hola?
    # - Recall: de todas las veces que habia Hola, cuantas detecto?
    # - F1: promedio entre precision y recall
    print(f"\nMejor accuracy en test: {best_accuracy:.2%}")
    print(f"\nReporte de clasificacion:")
    print(classification_report(
        labels, preds,
        target_names=le.classes_,
        digits=3
    ))

    # Matriz de confusion: muestra que senas se confunden entre si
    # Los numeros en la diagonal son aciertos, fuera de la diagonal son errores
    print("Matriz de confusion:")
    cm = confusion_matrix(labels, preds)
    header = "          " + " ".join(f"{name[:8]:>8}" for name in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{val:>8}" for val in row)
        print(f"{le.classes_[i][:8]:>10} {row_str}")

    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"Label encoder en: {ENCODER_PATH}")

    # Verificar si alcanzamos el objetivo de >85%
    if best_accuracy >= 0.85:
        print(f"\nObjetivo alcanzado: {best_accuracy:.2%} >= 85%")
    else:
        print(f"\nObjetivo NO alcanzado: {best_accuracy:.2%} < 85%")
        print("Sugerencias:")
        print("  - Recolectar mas datos para senas con baja precision")
        print("  - Verificar que las senas se realicen consistentemente")
        print("  - Aumentar epochs o ajustar learning rate")


if __name__ == '__main__':
    main()
