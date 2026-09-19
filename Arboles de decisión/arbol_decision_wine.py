"""Entrenamiento de un árbol de decisión para clasificar vinos.

Este script:
- carga el dataset Wine de scikit-learn;
- divide los datos en entrenamiento y prueba;
- evalúa distintos valores de max_depth;
- muestra la precisión del modelo y las reglas generadas por el árbol.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text

# Cargar datos
wine = load_wine()
X, y = wine.data, wine.target

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Muestras totales: {len(X)}")
print(f"Muestras de entrenamiento: {len(X_train)}")
print(f"Muestras de prueba: {len(X_test)}\n")


def entrenar_y_evaluar(max_depth):
    """Entrena un árbol con la profundidad indicada y devuelve su rendimiento."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    reglas = export_text(model, feature_names=list(wine.feature_names))

    return model, acc, reglas


# Modelo base: max_depth = 2
print("=" * 60)
print("MODELO CON max_depth = 2")
print("=" * 60)
model_2, acc_2, reglas_2 = entrenar_y_evaluar(max_depth=2)
print(f"Precisión (accuracy): {acc_2:.4f} ({acc_2 * 100:.2f}%)\n")
print("Reglas del árbol:")
print(reglas_2)

# Comparación por profundidad
print("=" * 60)
print("COMPARACIÓN DE DISTINTOS max_depth")
print("=" * 60)

profundidades = [1, 2, 3, 4, 5, 6, None]

for prof in profundidades:
    modelo, acc, _ = entrenar_y_evaluar(max_depth=prof)
    profundidad_real = modelo.get_depth()
    n_hojas = modelo.get_n_leaves()
    etiqueta = "None (sin límite)" if prof is None else prof

    print(
        f"max_depth={etiqueta:<18} -> profundidad real: {profundidad_real:<3} "
        f"hojas: {n_hojas:<3} accuracy: {acc:.4f}"
    )

# Modelo sin límite de profundidad
print("\n" + "=" * 60)
print("MODELO SIN LÍMITE DE PROFUNDIDAD (max_depth=None)")
print("=" * 60)
model_full, acc_full, reglas_full = entrenar_y_evaluar(max_depth=None)
print(f"Profundidad alcanzada: {model_full.get_depth()}")
print(f"Número de hojas: {model_full.get_n_leaves()}")
print(f"Precisión (accuracy): {acc_full:.4f} ({acc_full * 100:.2f}%)\n")
print("Reglas del árbol:")
print(reglas_full)
