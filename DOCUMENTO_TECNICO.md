# Documento Técnico - Sistemas Inteligentes

## 1. Introducción

Este documento describe la arquitectura, tecnologías, herramientas, procesos y metodología utilizada en el proyecto de Ciencia de Datos e Inteligencia Artificial basado en dos encuestas de preferencias de entretenimiento (**Pelis y series** y **La música y su influencia en el estado de ánimo**).

El proyecto cumple los requisitos definidos en `Hacer.md` y se guía de `sugerencias.md` para la adaptación de cada requisito a los datasets concretos.

---

## 2. Arquitectura del sistema

El sistema sigue un flujo modular por capas:

```
Datos crudos (.xlsx) 
        ↓
src/data_processing.py  (Limpieza + NLP + TF-IDF)
        ↓
data/processed/  (CSV limpios)      models/  (artefactos .joblib)
        ↓                              ↓
src/modeling.py  (Entrenamiento ML)   ←
        ↓
docs/  (metricas JSON + graficos PNG)
        ↓
app.py  (Interfaz Streamlit)  ←  modelos + vectorizadores + encoders
        ↓
Docker / Docker Compose  (despliegue en contenedor)
```

### Capas

1. **Capa de datos:** archivos Excel crudos en `data/raw/` y versiones limpias en `data/processed/`.
2. **Capa de procesamiento (`src/data_processing.py`):** limpieza de texto con expresiones regulares, normalización en español, tratamiento de nulos/atípicos, imputación, vectorización TF-IDF y generación de la matriz de características `X` y vector objetivo `y`.
3. **Capa de modelado (`src/modeling.py`):** entrenamiento y comparación de varios modelos supervisados (clasificación) y un modelo no supervisado (K-Means) con método del codo y PCA.
4. **Capa de orquestación (`src/train.py`):** ejecuta el pipeline completo para ambos datasets y guarda artefactos, métricas y gráficos.
5. **Capa de presentación (`app.py`):** aplicación web interactiva en Streamlit con dos pestañas de predicción.
6. **Capa de infraestructura (`Dockerfile`, `docker-compose.yml`):** contenedorización y despliegue.

---

## 3. Tecnologías y herramientas

| Categoría | Tecnología | Versión (referencia) | Uso |
|-----------|------------|----------------------|-----|
| Lenguaje | Python | 3.11 (Docker) / 3.14 (local) | Desarrollo general |
| Manipulación de datos | pandas | 2.2.3 | Carga y transformación de DataFrames |
| Procesamiento numérico | NumPy | 2.2.3 | Arrays y operaciones matemáticas |
| Machine Learning | scikit-learn | 1.6.1 | TF-IDF, modelos, K-Means, PCA, métricas |
| Serialización de modelos | joblib | 1.4.2 | Guardado/carga de modelos y artefactos |
| Lectura de Excel | openpyxl | 3.1.5 | Leer archivos .xlsx |
| NLP | NLTK | 3.9.1 | Stopwords en español |
| Normalización de texto | unidecode | 1.3.3 | Eliminar acentos/tildes |
| Visualización | Matplotlib, Seaborn, Plotly | 3.10 / 0.13.2 / 6.0.1 | Gráficos EDA y clusters |
| Interfaz web | Streamlit | 1.43.0 | App interactiva |
| Contenedorización | Docker + Docker Compose | — | Despliegue aislado |
| Notebooks | Jupyter (opcional) | — | Experimentos en `notebooks/` |

---

## 4. Procesamiento de datos (NLP y limpieza)

### 4.1 Datasets

- **Pelis y series:** 21 filas × 17 columnas. Columnas de texto: película favorita, serie favorita, actriz, actor, última peli vista, última serie, película reciente, serie reciente, serie odiada, película sobrevalorada, película odiada, serie sobrevalorada, actriz sobrevalorada, actor sobrevalorado. Objetivo: género favorito.
- **Música:** 24 filas × 18 columnas. Columnas de texto abiertas (tipo de música, momentos, estado de ánimo, géneros, experiencia, música triste/estresada, energía, concentración, sueño, diferencias con silencio, características de canción, emociones negativas, importancia, artistas). Variable numérica: horas al día. Objetivo: influencia en la concentración.

### 4.2 Limpieza y transformación

1. **Valores nulos y ambiguos:** respuestas como `"No tengo"`, `"Ninguna"`, `"No"`, `"N/A"`, `"Ninguno"` se estandarizan a `NaN` (categoría `Sin_Preferencia` / `Sin_Consumo`).
2. **Valores atípicos de texto:** respuestas fuera de contexto (ej. `"Hamburguesa con papas"` en película favorita) se detectan y se tratan como nulos.
3. **Expresiones regulares (regex):**
   - Eliminar aclaraciones entre paréntesis: `"Los juegos del hambre (todas)"` → `"Los juegos del hambre"`.
   - Normalizar títulos: `"Moana Real Action"` → `"Moana"`, `"Interestellar"` → `"Interestelar"`.
   - Extraer horas numéricas: `"4 horas"` → `4.0`.
4. **Normalización en español:** minúsculas (`.str.lower()`), eliminación de acentos con `unidecode`, remoción de puntuación y colapso de espacios múltiples.
5. **Imputación:** valores numéricos (horas) imputados con la mediana; texto relleno con `"sin_preferencia"` / `"sin_consumo"` para visualización.
6. **Vectorización NLP:** `TfidfVectorizer` con `stop_words` en español (NLTK), `ngram_range=(1,2)`, `max_features`, `min_df=1`. Genera matriz dispersa `X`.
7. **Codificación objetivo:** `LabelEncoder` sobre géneros agrupados (categorías canónicas: Acción, Ciencia Ficción, Terror, Comedia, Drama, Romance, Animación, Distopia, Superhéroes, Thriller, Fantasía, Otro) y binarización de concentración (Ayuda / Distrae / Neutro).

---

## 5. Análisis exploratorio de datos (EDA)

Se genera en `src/train.py` y `docs/`:

- **Univariado:** distribución de frecuencias del género favorito y de la influencia en la concentración; top de películas/series más mencionadas.
- **Bivariado:** relación entre género preferido y contenido sobrevalorado/odiado; horas de escucha vs. efecto en la concentración.
- **Multivariado:** matriz de coocurrencia de preferencias; proyección PCA/t-SNE de la matriz TF-IDF para detectar agrupamientos naturales.

Los gráficos se guardan en `docs/` (`.png`) y las métricas en `docs/metricas_*.json`.

---

## 6. Modelado y entrenamiento

### 6.1 Enfoque supervisado (clasificación)

Se entrenan y comparan cinco modelos sobre la matriz TF-IDF:

- Multinomial Naive Bayes (ideal para matrices TF-IDF dispersas)
- Support Vector Machines (`LinearSVC`)
- K-Nearest Neighbors (KNN)
- Random Forest
- Perceptrón Multicapa (`MLPClassifier`)

**Métricas:** accuracy, precisión, recall, F1-score (promedio *weighted*), matriz de confusión, validación cruzada (F1). Se selecciona el modelo con mejor F1-score.

### 6.2 Enfoque no supervisado (clustering)

- **K-Means:** agrupa encuestados en perfiles según su vector de preferencias.
- **Método del codo:** determina el número óptimo de clusters `k` mediante la inercia.
- **PCA:** reduce la matriz TF-IDF a 2 componentes para visualizar los clusters en un gráfico de dispersión.
- **Métrica:** coeficiente de silueta para evaluar la separación de clusters.

---

## 7. Artefactos generados

| Archivo | Descripción |
|---------|-------------|
| `data/processed/pelis_limpio.csv` | Dataset Pelis y series limpio |
| `data/processed/musica_limpio.csv` | Dataset Música limpio |
| `models/tfidf_pelis.joblib` | Vectorizador TF-IDF (pelis) |
| `models/label_encoder_pelis.joblib` | Codificador de géneros (pelis) |
| `models/modelo_pelis.joblib` | Mejor modelo supervisado (pelis) |
| `models/tfidf_musica.joblib` | Vectorizador TF-IDF (musica) |
| `models/label_encoder_musica.joblib` | Codificador de concentración (musica) |
| `models/modelo_musica.joblib` | Mejor modelo supervisado (musica) |
| `models/kmeans_model.joblib` | Modelo K-Means (no supervisado) |
| `docs/metricas_pelis.json` | Métricas modelos supervisados (pelis) |
| `docs/metricas_musica.json` | Métricas modelos supervisados (musica) |
| `docs/elbow_*.png`, `docs/clusters_*.png` | Gráficos método del codo y clusters PCA |

---

## 8. Despliegue

La aplicación se empaqueta en una imagen Docker basada en `python:3.11-slim`:

- Dependencias del sistema: `libgomp1` (requerido por OpenMP de scikit-learn/NumPy).
- Dependencias de Python instaladas con `--no-cache-dir` y versiones pinchadas en `requirements.txt`.
- Usuario no root (`streamlit`) por seguridad.
- `EXPOSE 8501` y healthcheck al endpoint `_stcore/health` de Streamlit.
- Orquestación mediante `docker-compose.yml` (puerto 8501, reinicio automático).

Comando de despliegue: `docker compose up -d --build`.

---

## 9. Calidad de código

- Estructura modular en carpetas (`src/`, `data/`, `models/`, `docs/`, `app.py`).
- Codificación UTF-8 para caracteres en español (acentos, `ñ`).
- Advertencias controladas (`warnings.filterwarnings("ignore")`).
- Manejo de excepciones en carga de modelos y entrenamiento (estrategia `stratify` condicional, `cross_val_score` en try/except).
- Pausa: el proyecto busca llegar a 0 errores y 0 advertencias en ejecución.
