


# 📊 Predicción de Cancelación de Clientes (Churn)

Este proyecto busca analizar y predecir la **cancelación de clientes (churn)** en el sector de telecomunicaciones, utilizando técnicas de **ciencia de datos y machine learning**.  
El objetivo es construir un pipeline reproducible que permita **entrenar, balancear, evaluar e interpretar modelos** para anticipar la pérdida de clientes.

---

## 🚀 Tecnologías utilizadas
- **Python 3.12**
- **Pandas / NumPy** → Manipulación y procesamiento de datos
- **Scikit-learn** → Modelado, validación y selección de variables
- **Imbalanced-learn (SMOTE)** → Balanceo de clases
- **SHAP** → Interpretabilidad de modelos
- **Matplotlib / Seaborn** → Visualización de resultados
- **Git + GitHub** → Control de versiones y colaboración

---

## 📂 Estructura del proyecto
```

├── data/
│   └── processed/
│       ├── datos\_telecomx.csv       # Dataset base utilizado
│       ├── datos\_procesados.pkl     # Datos procesados (generado automáticamente)
│       ├── feature\_names.pkl        # Features seleccionadas
│       ├── preprocesador.pkl        # Preprocesador entrenado
│       ├── train\_ready.csv          # Train set procesado (no se sube al repo)
│       └── test\_ready.csv           # Test set procesado (no se sube al repo)
│
├── notebooks/
│   └── 01\_churn\_pipeline.ipynb      # Jupyter Notebook principal con el análisis
│
├── src/
│   ├── preprocessing.py             # Funciones de preprocesamiento
│   ├── modeling.py                  # Modelos y evaluaciones
│   └── utils.py                     # Funciones auxiliares
│
├── .gitignore
├── requirements.txt
└── README.md

```

---

## 📑 Flujo del proyecto

### 🔹 1. Preprocesamiento de datos
- Carga del dataset `datos_telecomx.csv`.
- Limpieza y transformación de variables.
- División en **train** y **test**.

### 🔹 2. Balanceo de clases
- Se probó **RandomForest con pesos balanceados**.
- Se implementó **SMOTE** para balancear los datos en el entrenamiento.

### 🔹 3. Refinamiento y simplificación
- Selección de variables con **RandomForest + SelectFromModel**.
- Regularización en **Regresión Logística (L1 y L2)**.

### 🔹 4. Interpretabilidad
- Uso de **SHAP** para interpretar los modelos.
- Identificación de las variables más influyentes en la cancelación.

---

## 📊 Resultados preliminares
- Los modelos muestran un rendimiento prometedor en **ROC-AUC**.
- **RandomForest balanceado** y **Logística L1/L2** ofrecen un buen trade-off entre interpretabilidad y performance.
- SHAP confirma las variables más relevantes en la predicción de churn.

---

## ⚠️ Notas importantes
- Los archivos grandes (`train_ready.csv`, `test_ready.csv`, `venv/`) **NO se suben a GitHub** por límite de tamaño.  
- Solo se mantiene en el repositorio el dataset **`datos_telecomx.csv`** y los archivos ligeros necesarios para reproducir el pipeline.

---

## ▶️ Próximos pasos
- Optimización de hiperparámetros con **GridSearch / RandomizedSearch**.
- Comparación de más algoritmos (XGBoost, LightGBM).
- Despliegue de un servicio con **FastAPI**.
- Dashboard interactivo con **Streamlit**.

---

✍️ Desarrollado por **David Aguirre**
```

---
