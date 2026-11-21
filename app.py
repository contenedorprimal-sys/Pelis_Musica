"""
Aplicacion web interactiva (Streamlit) para el proyecto de Sistemas Inteligentes.

Funcionalidades (según sugerencias.md, requisito 8 - Despliegue e interfaz):
  - Pestaña Pelis y series: el usuario ingresa 3 peliculas/series favoritas y el
    sistema predice su genero cinematografico ideal (modelo supervisado entrenado).
  - Pestaña Musica: el usuario ingresa generos favoritos, horas de escucha al dia
    y que busca sentir; el sistema predice si la musica le Ayuda o Distrae al
    concentrarse y le asigna un perfil de oyente (cluster K-Means).
  - Pestaña Tablero Analitico: KPIs y metricas de los modelos, EDA y visualizaciones
    (elbow / clusters K-Means) de ambos datasets.

Se ejecuta con:  streamlit run app.py
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Sistemas Inteligentes - Pelis, Series y Musica",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Sistemas Inteligentes - Prediccion de Preferencias")
st.markdown(
    "Proyecto de Ciencia de Datos e Inteligencia Artificial. "
    "Entrenado con los encuestados de **Pelis y series** y **La musica y su influencia "
    "en el estado de animo**."
)

# ---------------------------------------------------------------------------
# Carga de modelos y artefactos
# ---------------------------------------------------------------------------

@st.cache_resource
def cargar_artefactos(nombre_dataset):
    """Carga vectorizador TF-IDF, label encoder y mejor modelo para un dataset.
    nombre_dataset: 'pelis' o 'musica' (se busca modelo_<nombre>.joblib)."""
    ruta_tfidf = os.path.join(MODELS_DIR, f"tfidf_{nombre_dataset}.joblib")
    ruta_le = os.path.join(MODELS_DIR, f"label_encoder_{nombre_dataset}.joblib")
    ruta_modelo = os.path.join(MODELS_DIR, f"modelo_{nombre_dataset}.joblib")
    tfidf = joblib.load(ruta_tfidf) if os.path.exists(ruta_tfidf) else None
    le = joblib.load(ruta_le) if os.path.exists(ruta_le) else None
    modelo = joblib.load(ruta_modelo) if os.path.exists(ruta_modelo) else None
    return tfidf, le, modelo, ruta_modelo


@st.cache_resource
def cargar_kmeans():
    ruta = os.path.join(MODELS_DIR, "kmeans_model.joblib")
    return joblib.load(ruta) if os.path.exists(ruta) else None


# ---------------------------------------------------------------------------
# PESTAÑA 1: PELIS Y SERIES
# ---------------------------------------------------------------------------

def pestaña_pelis():
    st.header("🎥 Pelis y Series - Prediccion de Genero Favorito")
    st.markdown(
        "Ingresa hasta 3 titulos (peliculas o series) que te gusten. El modelo predice "
        "cual es tu **genero cinematografico favorito**."
    )

    tfidf, le, modelo, _ = cargar_artefactos("pelis")

    if modelo is None or tfidf is None or le is None:
        st.error(
            "No se encontraron los artefactos entrenados para Pelis y series. "
            "Ejecuta primero `python src/train.py`."
        )
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        t1 = st.text_input("Película / serie favorita 1", placeholder="Ej. Matrix")
    with col2:
        t2 = st.text_input("Película / serie favorita 2", placeholder="Ej. Interestelar")
    with col3:
        t3 = st.text_input("Película / serie favorita 3", placeholder="Ej. Breaking Bad")

    if st.button("Predecir genero favorito", key="pred_pelis"):
        titulos = [t for t in (t1, t2, t3) if t and t.strip()]
        if not titulos:
            st.warning("Ingresa al menos un titulo.")
            return
        texto = " ".join(titulos)
        X = tfidf.transform([texto])
        pred = modelo.predict(X)[0]
        genero = le.inverse_transform([pred])[0]
        # Probabilidades si el modelo las soporta
        proba = None
        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X)[0]
            # 'pred' es la etiqueta codificada (entero); las columnas de
            # predict_proba siguen el orden de modelo.classes_, no de le.classes_
            # (que contiene los nombres de los generos).
            clases_modelo = list(getattr(modelo, "classes_", []))
            idx = clases_modelo.index(pred) if pred in clases_modelo else int(pred)
            confianza = proba[idx]
        else:
            confianza = None

        st.success(f"Genero cinematografico predicho: **{genero}**")
        if confianza is not None:
            st.info(f"Confianza del modelo: {confianza*100:.1f}%")
        st.caption(f"Titulos analizados: {', '.join(titulos)}")


# ---------------------------------------------------------------------------
# PESTAÑA 2: MUSICA
# ---------------------------------------------------------------------------

def pestaña_musica():
    st.header("🎵 Musica - Influencia en la Concentracion y Perfil de Oyente")
    st.markdown(
        "Responde 3 preguntas breves. El sistema predice si la musica te **Ayuda o Distrae "
        "al estudiar/trabajar** y te asigna un **perfil de oyente emocional** (K-Means)."
    )

    tfidf, le, modelo, _ = cargar_artefactos("musica")
    kmeans = cargar_kmeans()

    if modelo is None or tfidf is None or le is None:
        st.error(
            "No se encontraron los artefactos entrenados para Musica. "
            "Ejecuta primero `python src/train.py`."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        generos = st.text_area(
            "Generos musicales que escuchas (separados por coma)",
            placeholder="Ej. Electronica, Pop, Corridos, Reggaeton",
        )
        horas = st.number_input(
            "Horas al día que escuchas música", min_value=0, max_value=24, value=2, step=1
        )
    with col2:
        busca = st.text_area(
            "¿Qué buscas sentir al escuchar música?",
            placeholder="Ej. relajarme, concentrarme, motivarme, desahogarme",
        )

    if st.button("Predecir efecto y perfil", key="pred_musica"):
        texto = f"{generos} {busca}".strip()
        if not texto:
            st.warning("Ingresa al menos tus géneros y qué buscas sentir.")
            return
        # Vectorizar texto + anexar horas_dia (misma estructura que el entrenamiento)
        from scipy.sparse import hstack
        X_text = tfidf.transform([texto])
        horas_arr = np.array([[horas]])
        X = hstack([X_text, horas_arr]).tocsr()

        pred = modelo.predict(X)[0]
        efecto = le.inverse_transform([pred])[0]

        # Perfil de oyente via K-Means (si existe)
        perfil = None
        if kmeans is not None:
            cluster = kmeans.predict(X)[0]
            perfiles = {
                0: "Oyente Funcional (estudio / relajacion)",
                1: "Buscador de Desahogo Emocional",
                2: "Oyente Intenso / Motivado por Ritmo",
                3: "Indiferente / No Oyente",
            }
            perfil = perfiles.get(int(cluster), f"Perfil {cluster}")

        st.subheader("Resultado")
        if efecto == "Ayuda":
            st.success(f"Efecto en la concentracion: **{efecto}** ✅")
        elif efecto == "Distrae":
            st.error(f"Efecto en la concentracion: **{efecto}** ❌")
        else:
            st.info(f"Efecto en la concentracion: **{efecto}**")

        st.write(f"Horas de escucha al día: **{horas} h**")
        if perfil:
            st.write(f"Perfil de oyente (K-Means): **{perfil}**")
        st.caption(f"Texto analizado: {texto[:120]}...")


# ---------------------------------------------------------------------------
# PESTAÑA 3: TABLERO ANALITICO
# ---------------------------------------------------------------------------

# Configuracion del tablero: dataset -> columna objetivo y extras de EDA
CONFIG_TABLERO = {
    "pelis": {
        "titulo": "🎬 Películas y Series",
        "objetivo": "genero_agrupado",
        "titulo_eda": "Distribución de género favorito",
        "extra": None,
    },
    "musica": {
        "titulo": "🎵 Música",
        "objetivo": "concentracion_bin",
        "titulo_eda": "Distribución del efecto en la concentración",
        "extra": ("horas_dia", "Horas de escucha al día"),
    },
}

# Paleta compartida con los graficos del pipeline (matplotlib tab10)
PALETA = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Nombres de columnas en la tabla de metricas
COLUMNAS_TABLA = {
    "modelo": "Modelo",
    "accuracy": "Accuracy",
    "precision": "Precisión",
    "recall": "Recall",
    "f1_score": "F1-score",
    "cv_f1_mean": "CV F1 media",
    "cv_f1_std": "CV F1 desv.",
}


def cargar_metricas(nombre):
    """Lee docs/metricas_<nombre>.json (metricas por modelo + mejor modelo)."""
    ruta = os.path.join(BASE_DIR, "docs", f"metricas_{nombre}.json")
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def cargar_dataset_limpio(nombre):
    """Lee data/processed/<nombre>_limpio.csv generado por src/train.py."""
    ruta = os.path.join(BASE_DIR, "data", "processed", f"{nombre}_limpio.csv")
    if not os.path.exists(ruta):
        return None
    return pd.read_csv(ruta, encoding="utf-8")


def _fila_mejor(mejor, dfm):
    """Fila de metricas del mejor modelo (None si no esta en la tabla)."""
    if not mejor or dfm.empty or "modelo" not in dfm.columns:
        return None
    fila = dfm[dfm["modelo"] == mejor]
    return fila.iloc[0] if not fila.empty else None


def _pct(fila, columna):
    """Metrica formateada como porcentaje; '—' si falta o no es numerica."""
    if fila is None or columna not in getattr(fila, "index", []):
        return "—"
    valor = fila[columna]
    if not isinstance(valor, (int, float, np.integer, np.floating)) or pd.isna(valor):
        return "—"
    return f"{valor:.1%}"


def _tabla_metricas(dfm):
    """Tabla de metricas: encabezados en espanol, valores en porcentaje,
    indice oculto y el mejor F1 resaltado."""
    df = dfm.rename(columns=COLUMNAS_TABLA).drop(columns=["error"], errors="ignore")
    pct = [c for c in COLUMNAS_TABLA.values() if c in df.columns and c != "Modelo"]
    styler = df.style.format({c: "{:.1%}" for c in pct}, na_rep="—").hide(axis="index")
    if "F1-score" in df.columns:
        styler = styler.highlight_max(
            subset=["F1-score"], color="#dcebd8", props="font-weight:600"
        )
    return styler


def _layout_grafico(titulo, alto=320):
    """Layout base de los graficos del tablero (mismo estilo para todos)."""
    return dict(
        title=dict(text=titulo, x=0.01, xanchor="left", font=dict(size=15)),
        height=alto,
        margin=dict(l=10, r=24, t=52, b=12),
        font=dict(size=13),
        showlegend=False,
        bargap=0.15,
    )


def _grafico_barras(conteos, titulo, color=PALETA[0]):
    """Barras horizontales: las categorias mas frecuentes quedan arriba."""
    ordenado = conteos.sort_values()  # ascendente -> la mayor arriba
    fig = px.bar(
        y=ordenado.index.astype(str),
        x=ordenado.values,
        orientation="h",
        text=ordenado.values,
    )
    fig.update_traces(
        marker_color=color,
        textposition="outside",
        hovertemplate="%{y}: %{x} encuestados<extra></extra>",
    )
    fig.update_layout(**_layout_grafico(titulo))
    fig.update_xaxes(title="Encuestados", gridcolor="#e6e6e6", rangemode="tozero")
    fig.update_yaxes(title="")
    return fig


def _grafico_horas(serie, titulo, color=PALETA[0]):
    """Histograma de horas diarias con la media marcada por una linea punteada."""
    valores = serie.dropna().astype(float)
    media = float(valores.mean())
    fig = px.histogram(x=valores, nbins=8)
    fig.add_vline(
        x=media, line_dash="dash", line_color=PALETA[3], line_width=1.5,
    )
    fig.update_traces(
        marker_color=color,
        hovertemplate="Horas: %{x}<br>Encuestados: %{y}<extra></extra>",
    )
    fig.update_layout(**_layout_grafico(f"{titulo} — media {media:.1f} h"))
    fig.update_xaxes(title="Horas al día", gridcolor="#e6e6e6")
    fig.update_yaxes(title="Encuestados", showgrid=False)
    return fig


def pestaña_tablero():
    st.header("📊 Tablero Analítico")
    st.markdown(
        "KPIs, métricas y análisis exploratorio (EDA) de los modelos entrenados. "
        "Las métricas provienen de `docs/metricas_*.json` y el EDA de los CSV limpios "
        "generados por `src/train.py` en `data/processed/`."
    )

    for i, (dataset, cfg) in enumerate(CONFIG_TABLERO.items()):
        if i:
            st.divider()
        st.subheader(cfg["titulo"])

        datos = cargar_metricas(dataset)
        df_limpio = cargar_dataset_limpio(dataset)

        if datos is None:
            st.info(
                "No hay métricas guardadas para este dataset. "
                "Ejecuta `python src/train.py` para entrenar los modelos."
            )
            continue

        dfm = pd.DataFrame(datos.get("metricas", []))
        if dfm.empty:
            st.info(
                "El archivo de métricas está vacío. Re-ejecuta `python src/train.py`."
            )
            continue

        mejor = datos.get("mejor_modelo") or "—"
        st.caption(f"Mejor modelo: **{mejor}** · {len(dfm)} modelos evaluados")

        # ---------------- KPIs ----------------
        fila = _fila_mejor(mejor, dfm)
        objetivo = cfg["objetivo"]
        tiene_df = df_limpio is not None and objetivo in df_limpio.columns
        n = len(df_limpio) if df_limpio is not None else None
        n_clases = int(df_limpio[objetivo].nunique()) if tiene_df else None
        n_prueba = int(np.ceil(0.25 * n)) if n else None  # test_size=0.25 en train.py

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Muestras", n if n is not None else "—")
        c2.metric("Clases objetivo", n_clases if n_clases is not None else "—")
        c3.metric("Accuracy", _pct(fila, "accuracy"))
        c4.metric("F1-score", _pct(fila, "f1_score"))
        c5.metric("CV F1 (media ± desv.)", f"{_pct(fila, 'cv_f1_mean')} ± {_pct(fila, 'cv_f1_std')}")
        if n and n_prueba:
            st.caption(
                f"Métricas evaluadas en el 25% de prueba ({n_prueba} de {n} muestras)."
            )

        # ---------------- Metricas por modelo ----------------
        st.markdown("#### Métricas por modelo")
        st.dataframe(_tabla_metricas(dfm), width="stretch", hide_index=True)

        # ---------------- EDA ----------------
        st.markdown("#### Análisis exploratorio (EDA)")

        insights = []
        if tiene_df:
            conteos = df_limpio[objetivo].value_counts()
            if len(conteos):
                insights.append(
                    f"**{conteos.index[0]}** es la categoría más frecuente "
                    f"({conteos.iloc[0]}/{len(df_limpio)}, {conteos.iloc[0] / len(df_limpio):.0%})"
                )
        if cfg["extra"]:
            col_extra = cfg["extra"][0]
            if df_limpio is not None and col_extra in df_limpio.columns:
                media = df_limpio[col_extra].mean()
                if pd.notna(media):
                    insights.append(f"promedio de **{media:.1f} h** de escucha al día")
        if insights:
            st.caption(" · ".join(insights) + ".")

        cols_eda = st.columns(2 if cfg["extra"] else 1)
        with cols_eda[0]:
            if tiene_df:
                conteos = df_limpio[objetivo].value_counts()
                st.plotly_chart(
                    _grafico_barras(conteos, cfg["titulo_eda"]),
                    width="stretch",
                    config={"displayModeBar": False},
                )
            else:
                st.info(f"No se encontró la columna `{objetivo}` en el CSV limpio.")
        if cfg["extra"]:
            with cols_eda[1]:
                col_extra, titulo_extra = cfg["extra"]
                if df_limpio is not None and col_extra in df_limpio.columns:
                    st.plotly_chart(
                        _grafico_horas(df_limpio[col_extra], titulo_extra),
                        width="stretch",
                        config={"displayModeBar": False},
                    )
                else:
                    st.info(f"No se encontró la columna `{col_extra}` en el CSV limpio.")

        # ---------------- Pipeline no supervisado ----------------
        st.markdown("#### Pipeline no supervisado (K-Means)")
        g1, g2 = st.columns(2)
        imagenes = (
            (g1, "elbow", "Método del codo (inercia por k)"),
            (g2, "clusters", "Clusters K-Means proyectados con PCA"),
        )
        for columna, prefijo, titulo in imagenes:
            with columna:
                ruta_img = os.path.join(BASE_DIR, "docs", f"{prefijo}_{dataset}.png")
                if os.path.exists(ruta_img):
                    st.image(ruta_img, caption=titulo, width="stretch")
                else:
                    st.info(
                        f"No se encontró `{prefijo}_{dataset}.png`. "
                        "Ejecuta `python src/train.py` para generarlo."
                    )


# ---------------------------------------------------------------------------
# PESTAÑA 4: ACERCA DE / RESULTADOS
# ---------------------------------------------------------------------------

def pestaña_acerca():
    st.header("ℹ️ Acerca del proyecto")
    st.markdown("""
    ### Objetivo
    Desarrollar un proyecto integral de ciencia de datos e inteligencia artificial que
    abarque desde la identificación de una problemática y el procesamiento de datos
    hasta el entrenamiento de modelos, la visualización de resultados y el despliegue.

    ### Datasets utilizados
    1. **Pelis y series (Respuestas).xlsx** — 21 encuestados con preferencias de
       películas, series, actores y géneros favoritos.
    2. **La música y su influencia en el estado de ánimo (respuestas).xlsx** — 24
       encuestados con hábitos musicales, estado de ánimo y concentración.

    ### Técnicas aplicadas
    - Limpieza y transformación de datos, tratamiento de valores nulos y atípicos.
    - Expresiones regulares (regex) para normalizar títulos y extraer horas.
    - NLP: tokenización, stopwords en español y vectorización TF-IDF.
    - Aprendizaje supervisado: Naive Bayes, SVM, KNN, Random Forest, MLPClassifier.
    - Aprendizaje no supervisado: K-Means + método del codo + PCA.
    - Métricas: accuracy, precisión, recall, F1-score y matriz de confusión.

    ### Despliegue
    La aplicación está containerizada con Docker y se levanta con `docker compose up`.
    """)

    st.info(
        "📊 Los KPIs, métricas por modelo y gráficos del pipeline están en la pestaña "
        "**Tablero Analítico**."
    )


# ---------------------------------------------------------------------------
# NAVEGACION PRINCIPAL
# ---------------------------------------------------------------------------

pestanas = st.tabs(["🎬 Pelis y Series", "🎵 Música", "📊 Tablero Analítico", "ℹ️ Acerca del proyecto"])

with pestanas[0]:
    pestaña_pelis()
with pestanas[1]:
    pestaña_musica()
with pestanas[2]:
    pestaña_tablero()
with pestanas[3]:
    pestaña_acerca()

st.markdown("---")
st.caption("Desarrollado para la práctica de Sistemas Inteligentes — DESPLIEGUE EN DOCKER")
