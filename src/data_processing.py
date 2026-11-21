"""
Pipeline de procesamiento de datos (NLP + limpieza) para los datasets:
  - Pelis y series (Respuestas).xlsx
  - La música y su influencia en el estado de ánimo (respuestas).xlsx

Aplica: tratamiento de valores nulos, expresiones regulares (regex),
normalizacion de texto en espanol, imputacion de datos faltantes y
vectorizacion TF-IDF. Seguiendo sugerencias.md del proyecto.
"""
import re
import os
import warnings
import numpy as np
import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords

# Descargar stopwords en espanol si no estan disponibles
try:
    STOPWORDS_ES = set(stopwords.words("spanish"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS_ES = set(stopwords.words("spanish"))

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Tokens ambiguos / de rechazo en espanol -> se estandarizan como valor faltante
NULL_TOKENS = {
    "no tengo", "ninguna", "ninguno", "no", "n/a", "na", "nan",
    "-", "none", "", "nadie", "ninguna respuesta", "no aplica",
}


# ---------------------------------------------------------------------------
# Agrupamiento de generos (reduce cardinalidad y unifica variantes)
# ---------------------------------------------------------------------------

GENERO_MAP = [
    (r"accion", "Accion"),
    (r"ciencia ficcion|ciencia-ficcion|ficcion cientifica", "Ciencia Ficcion"),
    (r"terror|horror|suspenso", "Terror"),
    (r"comedia|humor", "Comedia"),
    (r"drama", "Drama"),
    (r"romance|amor", "Romance"),
    (r"animacion|anime|dibujo animado", "Animacion"),
    (r"distopia|distopía|apocalipsis", "Distopia"),
    (r"superheroe|super-héroe|marvel|dc comics", "Superheroes"),
    (r"thriller|misterio|intriga", "Thriller"),
    (r"fantasia|magia", "Fantasía"),
    (r"documental", "Documental"),
    (r"musical", "Musical"),
    (r"western|vaquero", "Western"),
]


def agrupar_genero(valor):
    """Mapea variantes de genero a una categoria canonica."""
    if pd.isna(valor):
        return "Sin_Preferencia"
    s = normalize_text(str(valor))
    if not s or s in NULL_TOKENS:
        return "Sin_Preferencia"
    for patron, canonico in GENERO_MAP:
        if re.search(patron, s):
            return canonico
    return "Otro"


def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def normalize_text(text):
    """Normalizacion de texto en espanol: minusculas, sin acentos, sin puntuacion extra."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).strip()
    s = s.lower()
    s = unidecode(s)  # elimina acentos y tildes (Maria -> maria)
    s = re.sub(r"\s+", " ", s)  # colapsa espacios multiples
    return s.strip()


def clean_parentheticals(text):
    """Elimina aclaraciones entre parentesis.
    'Los juegos del hambre (todas)' -> 'Los juegos del hambre'
    'Moana Real Action' se deja igual (se limpia por lista de sinonimos abajo)."""
    if not text:
        return text
    return re.sub(r"\s*\([^)]*\)", "", text).strip()


def remove_punctuation(text):
    if not text:
        return text
    return re.sub(r"[^\w\s]", " ", text)


def standardize_null(text):
    """Reemplaza respuestas ambiguas de rechazo por NaN estandarizado."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan
    s = str(text).strip()
    if s.lower() in NULL_TOKENS:
        return np.nan
    return s


# Sinonimos / variantes de titulos a estandarizar (regex aplicados luego de lower+unidecode)
TITLE_ALIASES = [
    (r"moana\s+real\s+action", "moana"),
    (r"interestellar", "interestelar"),
    (r"interstellar", "interestelar"),
    (r"harry\s+potter\s+\(.*\)", "harry potter"),
    (r"los\s+juegos\s+del\s+hambre\s*\(\s*\)?", "los juegos del hambre"),
    (r"crepusculo", "crepusculo"),
    (r"avengers", "avengers"),
]


def apply_title_aliases(text):
    if not text:
        return text
    s = text
    for pat, repl in TITLE_ALIASES:
        s = re.sub(pat, repl, s)
    return s.strip()


def clean_cell(text):
    """Limpieza completa de una celda de texto: nulos, parentesis, alias, puntuacion."""
    s = standardize_null(text)
    if pd.isna(s):
        return np.nan
    s = normalize_text(s)
    s = clean_parentheticals(s)
    s = apply_title_aliases(s)
    s = remove_punctuation(s)
    s = s.strip()
    return s if s else np.nan


def impute_nulls(df, text_columns):
    """Imputacion de valores nulos en columnas de texto: deja NaN para modelado y
    rellena con 'sin_preferencia' solo para fines de muestra/visualizacion."""
    df_filled = df.copy()
    for col in text_columns:
        df_filled[col] = df_filled[col].fillna("sin_preferencia")
    return df_filled


# ---------------------------------------------------------------------------
# Cargadores de datos crudos
# ---------------------------------------------------------------------------

def load_pelis():
    ruta = os.path.join(RAW_DIR, "Pelis y series (Respuestas).xlsx")
    df = pd.read_excel(ruta)
    return df


def load_musica():
    ruta = os.path.join(RAW_DIR, "La música y su influencia en el estado de ánimo (respuestas).xlsx")
    df = pd.read_excel(ruta)
    return df


# ---------------------------------------------------------------------------
# Procesamiento: PELIS Y SERIES
# ---------------------------------------------------------------------------

PELIS_TEXT_COLS = [
    "2. ¿Cuál es tu película favorita?",
    "3. Dinos tu serie favorita",
    "4. ¿Actriz favorita?",
    "5. ¿Cuál es tu actor favorito?",
    "7. ¿Cuál es la última peli que viste?",
    "8. ¿Cuál fue la última serie que viste?",
    "9. ¿Cuál pelicula reciente te ha gusta mucho?",
    "10. ¿Cuál es tu serie favorita más reciente que viste?",
    "11. ¿Serie que odias?",
    "12. ¿Película que tu consideres que está sobrevalorada?",
    "13. ¿Película que odias?",
    "14. ¿Serie que consideras que está sobrevalorada?",
    "15. ¿Actriz que consideras que está sobrevalorada?",
    "16. ¿Actor que consideras que está sobrevalorado?",
]

PELIS_TARGET_COL = "6. ¿Cuál es el género que te gusta más?"


def process_pelis(df=None, save_cleaned=True):
    """Limpieza y vectorizacion TF-IDF del dataset Pelis y series.

    Retorna un diccionario con:
      df_clean: DataFrame limpio (una fila por encuestado)
      X: matriz TF-IDF (dispersa) del texto combinado de peliculas/series/actores
      y: vector de generos favoritos (LabelEncoder aplicado)
      label_encoder: el LabelEncoder ajustado (para decodificar predicciones)
      tfidf: el TfidfVectorizer ajustado
      feature_names: nombres de los terminos TF-IDF
    """
    if df is None:
        df = load_pelis()
    df = df.copy()

    # Limpieza de columnas de texto con regex + normalizacion
    for col in PELIS_TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(clean_cell)

    # Texto combinado por usuario (fuente de features para NLP)
    df["texto_combinado"] = df[PELIS_TEXT_COLS].apply(
        lambda row: " ".join([str(v) for v in row.values if pd.notna(v)]), axis=1
    )
    df["texto_combinado"] = df["texto_combinado"].apply(lambda t: t if t.strip() else "sin_preferencia")

    # Variable objetivo: genero favorito agrupado (reduce cardinalidad de variantes)
    df["genero_agrupado"] = df[PELIS_TARGET_COL].apply(agrupar_genero)
    df = df.dropna(subset=["genero_agrupado"]).reset_index(drop=True)
    df = df[df["genero_agrupado"] != "Sin_Preferencia"].reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(df["genero_agrupado"].astype(str))

    # Vectorizacion TF-IDF (matriz dispersa de entrada X)
    tfidf = TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        min_df=1,
        stop_words=list(STOPWORDS_ES),  # stopwords en espanol (de, el, la, los...)
    )
    X = tfidf.fit_transform(df["texto_combinado"])
    feature_names = tfidf.get_feature_names_out()

    if save_cleaned:
        ensure_dirs()
        df.to_csv(os.path.join(PROCESSED_DIR, "pelis_limpio.csv"), index=False, encoding="utf-8")

    return {
        "df_clean": df,
        "X": X,
        "y": y,
        "label_encoder": le,
        "tfidf": tfidf,
        "feature_names": feature_names,
        "text_column": "texto_combinado",
        "target_name": "genero_agrupado",
        "target_classes": list(le.classes_),
    }


# ---------------------------------------------------------------------------
# Procesamiento: MUSICA (el otro .xlsx)
# ---------------------------------------------------------------------------

MUSICA_TEXT_COLS = [
    "1. ¿Qué tipo de música escuchas con mayor frecuencia y por qué?",
    "3. ¿En qué momentos del día acostumbras escuchar música y qué actividades realizas mientras la escuchas?",
    "5. ¿Cómo describirías tu estado de ánimo antes y después de escuchar tu música favorita?",
    "6. ¿Qué géneros musicales te hacen sentir feliz, tranquilo, triste o motivado?",
    "7. ¿Puedes describir una experiencia en la que una canción haya cambiado tu estado de ánimo?",
    "8. ¿Qué tipo de música escuchas cuando te sientes triste, estresado o preocupado, y por qué?",
    "9. ¿Cómo influye la música en tu nivel de energía y motivación durante el día?",
    "10. ¿De qué manera influye la música en tu concentración al estudiar, trabajar o realizar tareas?",
    "11. ¿Cómo describirías la influencia de la música en tu descanso o calidad del sueño?",
    "12. ¿Qué diferencias notas entre realizar una actividad con música y realizarla en silencio?",
    "13. ¿Qué características de una canción, como el ritmo, la letra o el volumen, influyen más en tus emociones?",
    "14. ¿Consideras que la música puede ayudar a manejar emociones negativas? Explica tu respuesta con un ejemplo.",
    "15. ¿Cómo describirías la importancia de la música en tu vida y en tu bienestar emocional?",
    "16. ¿Cuáles son los 5 artistas musicales que escuchas con mayor frecuencia ",
]

MUSICA_HOURS_COL = "2. ¿Cuánto tiempo al día dedicas aproximadamente a escuchar música?"
MUSICA_TARGET_COL = "10. ¿De qué manera influye la música en tu concentración al estudiar, trabajar o realizar tareas?"


def extract_hours(text):
    """Extrae numerico de horas desde cadenas como '4 horas', '7 horas', '0'."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan
    m = re.search(r"(\d+(?:[.,]\d+)?)", str(text))
    if m:
        return float(m.group(1).replace(",", "."))
    return np.nan


def process_musica(df=None, save_cleaned=True):
    """Limpieza del dataset Musica. Extrae horas numericas (regex) y vectoriza texto."""
    if df is None:
        df = load_musica()
    df = df.copy()

    # Horas diarias numericas (requisito: expresiones regulares sobre dato mixto)
    if MUSICA_HOURS_COL in df.columns:
        df["horas_dia"] = df[MUSICA_HOURS_COL].apply(extract_hours)
        df["horas_dia"] = df["horas_dia"].fillna(df["horas_dia"].median())

    for col in MUSICA_TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(clean_cell)

    df["texto_combinado"] = df[MUSICA_TEXT_COLS].apply(
        lambda row: " ".join([str(v) for v in row.values if pd.notna(v)]), axis=1
    )
    df["texto_combinado"] = df["texto_combinado"].apply(lambda t: t if t.strip() else "sin_consumo")

    # Variable objetivo: influencia en la concentracion (binaria: ayuda vs distrae)
    def binarize_concentracion(val):
        if pd.isna(val):
            return np.nan
        s = normalize_text(str(val))
        ayuda_palabras = {"ayuda", "concentra", "mejora", "ayudar", "ayuda a concentrar"}
        distrae_palabras = {"distrae", "distraer", "empeora", "no deja", "estreso", "estres"}
        if any(p in s for p in ayuda_palabras) and not any(p in s for p in distrae_palabras):
            return "Ayuda"
        if any(p in s for p in distrae_palabras):
            return "Distrae"
        return "Neutro"

    df["concentracion_bin"] = df[MUSICA_TARGET_COL].apply(binarize_concentracion)
    df_m = df.dropna(subset=["concentracion_bin"]).reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(df_m["concentracion_bin"].astype(str))

    tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=1, stop_words=list(STOPWORDS_ES))
    X_text = tfidf.fit_transform(df_m["texto_combinado"])

    # Combinar texto TF-IDF con variable numerica horas_dia (matriz multivariada)
    horas = df_m[["horas_dia"]].values
    from scipy.sparse import hstack
    X = hstack([X_text, horas]).tocsr()
    feature_names = list(tfidf.get_feature_names_out()) + ["horas_dia"]

    if save_cleaned:
        ensure_dirs()
        df_m.to_csv(os.path.join(PROCESSED_DIR, "musica_limpio.csv"), index=False, encoding="utf-8")

    return {
        "df_clean": df_m,
        "X": X,
        "y": y,
        "label_encoder": le,
        "tfidf": tfidf,
        "feature_names": np.array(feature_names),
        "text_column": "texto_combinado",
        "target_name": MUSICA_TARGET_COL,
        "target_classes": list(le.classes_),
        "horas_col": "horas_dia",
    }


if __name__ == "__main__":
    res_p = process_pelis()
    print("=== PELIS Y SERIES ===")
    print("Filas:", res_p["df_clean"].shape[0], "| Clases genero:", res_p["target_classes"])
    print("Forma X (TF-IDF):", res_p["X"].shape)

    res_m = process_musica()
    print("\n=== MUSICA ===")
    print("Filas:", res_m["df_clean"].shape[0], "| Clases concentracion:", res_m["target_classes"])
    print("Forma X (TF-IDF + horas):", res_m["X"].shape)
