# Clasificación con árbol de decisión (Wine dataset)

Práctica de Inteligencia Artificial (SCC-1012) — TECNM Ensenada

## Descripción

Este proyecto usa `DecisionTreeClassifier` de scikit-learn para clasificar los vinos del dataset `wine` según sus características químicas. El objetivo era ver si un árbol de decisión podía distinguir bien entre las 3 clases sin complicarse demasiado.

## Cómo ejecutarlo

```bash
pip install scikit-learn
python arbol_decision_wine.py
```

## Resultados

Probé varios valores de `max_depth` para ver cómo cambiaba la precisión del modelo.

- División de datos: 80% entrenamiento / 20% prueba.
- Con `max_depth=2`: **86.11%**
- Con `max_depth=3` y superiores: **94.44%**
- La profundidad que el árbol usó de forma natural fue **4**.

### Qué pasó al cambiar `max_depth`

Lo más claro fue el cambio al pasar de `max_depth=2` a `max_depth=3`. Ahí la precisión saltó de **86.11%** a **94.44%**. Eso tiene sentido porque el árbol pudo usar reglas más específicas para separar mejor las tres clases.

A partir de `max_depth=3`, no mejoró más aunque aumentara la profundidad: con 4, 5 o 6, y también sin límite, el árbol se quedó igual. En la práctica, la profundidad útil para este dataset fue 4. Más allá de eso, ya no encontraba divisiones que aportaran algo real.

### Árbol sin límite de profundidad (`max_depth=None`)

Cuando dejé el árbol sin límite, terminó con la misma estructura que en `max_depth=4`: profundidad real de 4 y 7 hojas, con **94.44%** de precisión. La diferencia es que las reglas quedaron más detalladas y usó más variables, como `color_intensity`, `proline`, `flavanoides`, `ash` y `alcohol`.

En este caso no hubo overfitting notable porque el dataset es pequeño y bastante limpio. En otros datos más grandes o más ruidosos, dejar el árbol sin límite podría memorizar el entrenamiento y perder capacidad de generalización.

## Qué me llamó la atención

Lo más interesante fue que el árbol no usó muchas variables para llegar a un buen resultado. Con solo unas cuantas reglas, logró clasificar bien más del 94% de los ejemplos de prueba.

Además, las variables que más aparecieron fueron `color_intensity`, `proline` y `flavanoides`. Eso parece bastante lógico: estas mediciones varían bastante entre los tipos de vino y ayudan a separar bien las clases.

## ¿Este dataset sí sirve para un árbol de decisión?

Sí, es un buen ejemplo para este tipo de modelo.

- Tiene características numéricas y continuas, lo cual encaja perfectamente con las divisiones que hace un árbol.
- Tiene 3 clases bien definidas.
- No tiene valores faltantes.
- Tiene 178 muestras, que es suficiente para entrenar y probar sin que los resultados sean poco confiables.

En otras palabras, el dataset tiene exactamente las condiciones que un árbol de decisión suele manejar bien.

### Variables importantes

- **`color_intensity`**: fue la primera variable que el árbol usó para dividir los datos.
- **`proline`**: aparece en varias ramas y ayuda mucho a distinguir clases.
- **`flavanoides`**: sirve para separar mejor la clase 2 del resto.
- **`alcohol` y `ash`**: aparecen más hacia las ramas profundas y ayudan a afinar la clasificación.

### ¿Agregar otras variables?

Sí, pero no es necesario para que el modelo funcione bien. En un caso real, variables como región, año de cosecha, clima o pH podrían ayudar a distinguir casos más difíciles, pero con el dataset actual el árbol ya obtiene un desempeño muy bueno.

## Estructura del repositorio

```text
├── arbol_decision_wine.py   # Script principal con entrenamiento y evaluación
└── README.md                # Documentación del proyecto
```
