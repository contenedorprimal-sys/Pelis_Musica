"""
Entrenamiento y evaluacion de modelos de Machine Learning.

Enfoques implementados:
  - Supervisado: clasificacion del genero favorito (pelis/series) y de
    influencia en la concentracion (musica).
    Modelos: MultinomialNB, LinearSVC, KNN, Random Forest, MLPClassifier.
  - No supervisado: K-Means clustering + metodo del codo + PCA para
    segmentar usuarios en perfiles de espectador / oyente.

Metricas: accuracy, precision, recall, f1-score y matriz de confusion.
"""
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "docs")


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Modelos supervisados disponibles
# ---------------------------------------------------------------------------

def get_supervised_models(random_state=42):
    """Diccionario de modelos supervisados a comparar."""
    return {
        "Naive Bayes (MultinomialNB)": MultinomialNB(),
        "SVM (LinearSVC)": LinearSVC(random_state=random_state, max_iter=5000),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        "Red Neuronal (MLPClassifier)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=1000, random_state=random_state,
        ),
    }


def train_supervised(X, y, target_classes, model_name=None, test_size=0.25, random_state=42):
    """Entrena y evalua modelos supervisados. Retorna metricas y el mejor modelo."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=(y if len(np.unique(y)) > 1 and min(
            np.bincount(y)) >= 2 else None),
    )

    models = get_supervised_models(random_state)
    results = []
    best_model = None
    best_f1 = -1.0
    best_name = None

    for name, model in models.items():
        if model_name and name != model_name:
            continue
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(
                y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(
                y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            try:
                cv = cross_val_score(model, X_train, y_train, cv=min(
                    3, len(np.unique(y_train))), scoring="f1_weighted")
                cv_mean = float(cv.mean())
                cv_std = float(cv.std())
            except Exception:
                cv_mean = 0.0
                cv_std = 0.0
            results.append({
                "modelo": name,
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "cv_f1_mean": round(cv_mean, 4),
                "cv_f1_std": round(cv_std, 4),
            })
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_name = name
        except Exception as e:
            results.append({"modelo": name, "error": str(e)})

    results_df = pd.DataFrame(results)

    # Matriz de confusion del mejor modelo
    cm = None
    if best_model is not None:
        y_pred_best = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_best)
        # Guardar el mejor modelo y el vectorizador/encoder se guardan desde el caller
        joblib.dump(best_model, os.path.join(
            MODELS_DIR, f"mejor_modelo_{best_name.split()[0].lower()}.joblib"))

    reporte = {
        "modelo": best_model,
        "resultados": results_df,
        "mejor_modelo": best_name,
        "mejor_f1": round(float(best_f1), 4),
        "matriz_confusion": cm,
        "clases": list(target_classes),
        "reporte_clasificacion": classification_report(
            y_test, best_model.predict(X_test),
            labels=sorted(np.unique(y_test)),
            target_names=[str(c) for c in sorted(np.unique(y_test))],
            zero_division=0
        ) if best_model is not None else "",
    }
    return reporte


# ---------------------------------------------------------------------------
# Modelo no supervisado: K-Means + Elbow + PCA
# ---------------------------------------------------------------------------

def elbow_method(X, max_k=10, random_state=42):
    """Metodo del codo: calcula inercia para k = 1..max_k."""
    inercias = []
    ks = range(1, max_k + 1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inercias.append(km.inertia_)
    return list(ks), inercias


def silhouette_scores(X, max_k=10, random_state=42):
    """Coeficiente de silueta para k = 2..max_k."""
    scores = []
    ks = range(2, max_k + 1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        scores.append(silhouette_score(X, labels))
    return list(ks), scores


def train_kmeans(X, n_clusters=3, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if n_clusters > 1 else 0.0
    joblib.dump(km, os.path.join(MODELS_DIR, "kmeans_model.joblib"))
    return km, labels, sil


def reduce_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X.toarray() if hasattr(X, "toarray") else X)
    return pca, X_pca


def plot_elbow(ks, inercias, nombre="elbow"):
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inercias, marker="o")
    plt.xlabel("Numero de clusters (k)")
    plt.ylabel("Inercia (SSE)")
    plt.title("Metodo del Codo - seleccion de k")
    plt.grid(True, alpha=0.3)
    path = os.path.join(REPORT_DIR, f"{nombre}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_clusters(X_pca, labels, n_clusters, nombre="clusters_pca"):
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=labels, cmap="tab10", s=50, alpha=0.8)
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title(f"Segmentacion de usuarios - K-Means (k={n_clusters}) + PCA")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True, alpha=0.3)
    path = os.path.join(REPORT_DIR, f"{nombre}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


if __name__ == "__main__":
    print("Modulo de modelado listo. Importar desde main o notebooks.")
