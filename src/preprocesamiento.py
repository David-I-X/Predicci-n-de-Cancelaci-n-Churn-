import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle
import os


# 🔹 Directorio raíz del proyecto (sube desde /src hasta la raíz del repo)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def cargar_datos(ruta_csv: str):
    """Carga el dataset limpio."""
    ruta_absoluta = os.path.join(BASE_DIR, ruta_csv)
    return pd.read_csv(ruta_absoluta)


def dividir_variables(df: pd.DataFrame, target: str):
    """Separa features y target."""
    X = df.drop(columns=[target, "customerID"], errors="ignore")
    y = df[target]
    return X, y


def dividir_train_test(X, y, test_size=0.2, random_state=42):
    """Divide en train y test con estratificación."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def crear_pipeline(X: pd.DataFrame):
    """Crea el pipeline de preprocesamiento."""
    columnas_numericas = X.select_dtypes(include=["int64", "float64"]).columns
    columnas_categoricas = X.select_dtypes(include=["object", "category", "bool"]).columns

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), columnas_numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas)
        ]
    )

    return preprocesador


def ejecutar_preprocesamiento(ruta_csv, salida_dir=PROCESSED_DIR):
    """Carga, procesa y guarda los datos transformados."""
    df = cargar_datos(ruta_csv)
    X, y = dividir_variables(df, "cancelacion")
    X_train, X_test, y_train, y_test = dividir_train_test(X, y)

    preprocesador = crear_pipeline(X)
    X_train_proc = preprocesador.fit_transform(X_train)
    X_test_proc = preprocesador.transform(X_test)

    # Guardar preprocesador y datos
    os.makedirs(salida_dir, exist_ok=True)
    with open(os.path.join(salida_dir, "preprocesador.pkl"), "wb") as f:
        pickle.dump(preprocesador, f)

    with open(os.path.join(salida_dir, "datos_procesados.pkl"), "wb") as f:
        pickle.dump((X_train_proc, X_test_proc, y_train, y_test), f)

    print(f"✅ Preprocesamiento completado. Datos guardados en {salida_dir}")


def cargar_datos_procesados(salida_dir=PROCESSED_DIR):
    """Carga los datos procesados y el preprocesador."""
    with open(os.path.join(salida_dir, "preprocesador.pkl"), "rb") as f:
        preprocesador = pickle.load(f)

    with open(os.path.join(salida_dir, "datos_procesados.pkl"), "rb") as f:
        X_train_proc, X_test_proc, y_train, y_test = pickle.load(f)

    return X_train_proc, X_test_proc, y_train, y_test, preprocesador


if __name__ == "__main__":
    ejecutar_preprocesamiento("data/processed/datos_telecomx.csv")
