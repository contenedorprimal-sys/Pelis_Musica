# Sistemas Inteligentes - Pelis, Series y Música

Proyecto integral de **Ciencia de Datos e Inteligencia Artificial** desarrollado para la práctica de la asignatura de Sistemas Inteligentes. Aplica procesamiento de lenguaje natural (NLP), aprendizaje automático supervisado y no supervisado, análisis exploratorio de datos (EDA) y despliegue en Docker sobre dos encuestas de preferencias de entretenimiento.

> **Datos utilizados:**
> - `Pelis y series (Respuestas).xlsx` — 21 encuestados con gustos de películas, series, actores y géneros.
> - `La música y su influencia en el estado de ánimo (respuestas).xlsx` — 24 encuestados con hábitos musicales, estado de ánimo y concentración.

---

## Requisitos cumplidos

| # | Requisito |
|---|-----------|
| 1 | Selección y definición del proyecto (problema, solución, alcance)
| 2 | Documentación (técnica, negocio, README, comandos)
| 3 | Gestión y control del código (estructura ordenada, limpieza)
| 4 | Preparación y procesamiento de datos (nulos, atípicos, regex, NLP, IA) 
| 5 | Análisis exploratorio (univariado, bivariado, multivariado)
| 6 | Modelado supervisado y no supervisado (varios modelos, comparación, métricas)
| 7 | Tableros y visualización (EDA, modelos, resultados)
| 8 | Despliegue e interfaz web interactiva (Streamlit + Docker)

---

## Estructura del repositorio

```
/
├── data/
│   ├── raw/                          # Datos crudos (.xlsx originales)
│   └── processed/                    # Datos limpios (.csv generados)
├── models/                           # Modelos entrenados y artefactos (.joblib)
├── docs/                             # Métricas (.json) y gráficos (.png)
├── src/
│   ├── data_processing.py            # Pipeline NLP + limpieza + TF-IDF
│   ├── modeling.py                   # Modelos supervisados + K-Means + PCA
│   └── train.py                      # Script principal de entrenamiento
├── app.py                            # Aplicacion web Streamlit
├── Dockerfile                        # Imagen Docker de la app
├── docker-compose.yml                # Orquestacion con Docker Compose
├── requirements.txt                  # Dependencias de Python
├── .dockerignore
├── .streamlit/config.toml            # Config de Streamlit (headless)
├── README.md                         # Este archivo
├── DOCUMENTO_TECNICO.md              # Arquitectura, tecnologias, pipeline
└── DOCUMENTO_NEGOCIO.md              # Problema, oportunidad, valor, beneficios
```

---

## Despliegue en Docker (levantar el proyecto)

### Prerrequisitos
- [Docker](https://docs.docker.com/engine/install/) y
- [Docker Compose](https://docs.docker.com/compose/install/) instalados.

### Comandos de ejecución

```bash
# 1. Clonar / situarse en el directorio del proyecto
cd /ruta/al/proyecto/p3

# 2. Construir la imagen y levantar el contenedor
docker compose up -d --build

# 3. Abrir la aplicacion en el navegador
#    http://localhost:8501

# 4. Ver logs
docker compose logs -f streamlit-app

# 5. Parar y eliminar contenedores
docker compose down
```

La aplicación queda disponible en **http://localhost:8501**.

### Comandos alternativos (solo Docker CLI)

```bash
docker build -t sistemas-app:1.0.0 .
docker run -d --name sistemas-app -p 8501:8501 sistemas-app:1.0.0
docker logs -f sistemas-app
docker stop sistemas-app && docker rm sistemas-app
```

---

## Ejecución local (sin Docker)

```bash
# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt

# Descargar stopwords de NLTK (se hace automaticamente al ejecutar)

# Entrenar los modelos (genera data/processed, models y docs)
python src/train.py

# Lanzar la aplicacion web
streamlit run app.py
# Abrir http://localhost:8501
```

---

## Cómo funciona la aplicación

La app (Streamlit) tiene cuatro pestañas:

1. **Pelis y Series** — El usuario ingresa hasta 3 títulos (películas/series) y el modelo predice su **género cinematográfico favorito** (clasificación supervisada entrenada con TF-IDF + varios algoritmos).
2. **Música** — El usuario ingresa géneros favoritos, horas de escucha al día y qué busca sentir; el sistema predice si la música le **Ayuda o Distrae** al concentrarse y le asigna un **perfil de oyente emocional** (K-Means).
3. **Tablero Analítico** — KPIs (muestras, clases, accuracy, F1, CV F1), tabla comparativa de modelos, análisis exploratorio (distribución de la variable objetivo y horas de escucha) y gráficos del pipeline no supervisado (método del codo y clusters con PCA).
4. **Acerca del proyecto** — Información general, datasets y técnicas aplicadas.

---

## Técnicas aplicadas

- **Limpieza de datos:** tratamiento de valores nulos, respuestas ambiguas (`No tengo`, `Ninguna`, `N/A`), valores atípicos de texto (ej. `"Hamburguesa con papas"`).
- **Expresiones regulares (regex):** eliminar aclaraciones entre paréntesis (`(todas)`), normalizar títulos (`Moana Real Action` → `Moana`), extraer horas numéricas (`4 horas` → `4`).
- **NLP:** normalización en español (minúsculas, sin acentos con `unidecode`), stopwords en español, tokenización, vectorización **TF-IDF** (matriz dispersa).
- **Aprendizaje supervisado:** Multinomial Naive Bayes, LinearSVC, KNN, Random Forest, MLPClassifier. Métricas: accuracy, precisión, recall, F1-score, matriz de confusión, validación cruzada.
- **Aprendizaje no supervisado:** K-Means, método del codo (inercia), coeficiente de silueta, PCA para visualización de clusters en 2D.
- **Despliegue:** contenedor Docker (`python:3.11-slim`), Docker Compose, Streamlit headless en puerto 8501.

---

## Autor

Proyecto desarrollado para la práctica de **Sistemas Inteligentes**.

---
# Pelis_Musica
