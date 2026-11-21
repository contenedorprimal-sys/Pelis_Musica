"""
Script principal de entrenamiento del proyecto.

Ejecuta el pipeline completo para ambos datasets:
  1. Limpieza y vectorizacion NLP (data_processing)
  2. Entrenamiento y comparacion de modelos supervisados (modeling)
  3. Entrenamiento K-Means + PCA + metodo del codo (no supervisado)
  4. Guarda modelos, metricas y graficos en /models y /docs

Uso:
    python src/train.py
"""
import os
import sys
import json
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data_processing import process_pelis, process_musica
from src.modeling import (
    train_supervised, train_kmeans, elbow_method, silhouette_scores,
    reduce_pca, plot_elbow, plot_clusters, ensure_dirs,
)


def main():
    ensure_dirs()
    print("=" * 60)
    print("ENTRENAMIENTO DEL PROYECTO - Sistemas Inteligentes")
    print("=" * 60)

    # ---------------------------------------------------------------
    # DATASET 1: PELIS Y SERIES
    # ---------------------------------------------------------------
    print("\n--- Procesando: Pelis y series ---")
    res_p = process_pelis(save_cleaned=True)
    X_p, y_p = res_p["X"], res_p["y"]
    print(f"Muestra: {X_p.shape[0]} encuestados | Features TF-IDF: {X_p.shape[1]}")
    print(f"Clases (genero favorito): {res_p['target_classes']}")

    print("\nEntrenando modelos SUPERVISADOS (Pelis y series)...")
    rep_p = train_supervised(X_p, y_p, res_p["target_classes"])
    print(rep_p["resultados"].to_string(index=False))
    print(f"\nMejor modelo: {rep_p['mejor_modelo']} (F1 = {rep_p['mejor_f1']})")

    # Guardar vectorizador, encoder y mejor modelo con nombre fijo por dataset
    joblib.dump(res_p["tfidf"], os.path.join(BASE_DIR, "models", "tfidf_pelis.joblib"))
    joblib.dump(res_p["label_encoder"], os.path.join(BASE_DIR, "models", "label_encoder_pelis.joblib"))
    joblib.dump(rep_p["modelo"], os.path.join(BASE_DIR, "models", "modelo_pelis.joblib"))
    res_p["df_clean"].to_csv(os.path.join(BASE_DIR, "data", "processed", "pelis_limpio.csv"),
                             index=False, encoding="utf-8")

    # Metricas a JSON
    metricas_p = rep_p["resultados"].to_dict(orient="records")
    with open(os.path.join(BASE_DIR, "docs", "metricas_pelis.json"), "w", encoding="utf-8") as f:
        json.dump({"metricas": metricas_p, "mejor_modelo": rep_p["mejor_modelo"]}, f,
                  ensure_ascii=False, indent=2)

    print("\nEntrenando NO SUPERVISADO (K-Means) - Pelis y series...")
    ks, inercias = elbow_method(X_p, max_k=8)
    plot_elbow(ks, inercias, nombre="elbow_pelis")
    k_opt = 3  # se puede ajustar segun el codo / silueta
    km_p, labels_p, sil_p = train_kmeans(X_p, n_clusters=k_opt)
    pca_p, X_pca_p = reduce_pca(X_p, n_components=2)
    plot_clusters(X_pca_p, labels_p, k_opt, nombre="clusters_pelis")
    print(f"K-Means k={k_opt} | Silueta = {sil_p:.3f}")

    # ---------------------------------------------------------------
    # DATASET 2: MUSICA
    # ---------------------------------------------------------------
    print("\n--- Procesando: La musica y su influencia en el estado de animo ---")
    res_m = process_musica(save_cleaned=True)
    X_m, y_m = res_m["X"], res_m["y"]
    print(f"Muestra: {X_m.shape[0]} encuestados | Features (TF-IDF + horas): {X_m.shape[1]}")
    print(f"Clases (concentracion): {res_m['target_classes']}")

    print("\nEntrenando modelos SUPERVISADOS (Musica)...")
    rep_m = train_supervised(X_m, y_m, res_m["target_classes"])
    print(rep_m["resultados"].to_string(index=False))
    print(f"\nMejor modelo: {rep_m['mejor_modelo']} (F1 = {rep_m['mejor_f1']})")

    joblib.dump(res_m["tfidf"], os.path.join(BASE_DIR, "models", "tfidf_musica.joblib"))
    joblib.dump(res_m["label_encoder"], os.path.join(BASE_DIR, "models", "label_encoder_musica.joblib"))
    joblib.dump(rep_m["modelo"], os.path.join(BASE_DIR, "models", "modelo_musica.joblib"))
    res_m["df_clean"].to_csv(os.path.join(BASE_DIR, "data", "processed", "musica_limpio.csv"),
                             index=False, encoding="utf-8")

    metricas_m = rep_m["resultados"].to_dict(orient="records")
    with open(os.path.join(BASE_DIR, "docs", "metricas_musica.json"), "w", encoding="utf-8") as f:
        json.dump({"metricas": metricas_m, "mejor_modelo": rep_m["mejor_modelo"]}, f,
                  ensure_ascii=False, indent=2)

    print("\nEntrenando NO SUPERVISADO (K-Means) - Musica...")
    ks_m, inercias_m = elbow_method(X_m, max_k=8)
    plot_elbow(ks_m, inercias_m, nombre="elbow_musica")
    k_opt_m = 3
    km_m, labels_m, sil_m = train_kmeans(X_m, n_clusters=k_opt_m)
    pca_m, X_pca_m = reduce_pca(X_m, n_components=2)
    plot_clusters(X_pca_m, labels_m, k_opt_m, nombre="clusters_musica")
    print(f"K-Means k={k_opt_m} | Silueta = {sil_m:.3f}")

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("Modelos en: models/")
    print("Metricas y graficos en: docs/")
    print("Datos limpios en: data/processed/")
    print("=" * 60)


if __name__ == "__main__":
    main()
