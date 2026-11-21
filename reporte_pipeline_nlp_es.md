# Reporte: Mejores prácticas para un pipeline de NLP en español (texto corto)

**Contexto:** proyecto universitario de ciencia de datos.  
**Dominio:** títulos de películas, series y actores (texto corto en español).  
**Stack:** `pandas`, `re`, `unidecode`, `nltk`, `scikit-learn`.  
**Idioma del reporte:** español.

---

## 1. Normalización de texto en español

Para texto corto (títulos) conviene normalizar de forma determinista antes de cualquier vectorización. Se recomienda aplicar, en orden:

1. Minúsculas (`.lower()`).
2. Eliminación de acentos (`unidecode` o `str.translate` con tabla, o `unicodedata`).
3. Eliminación de signos de puntuación.
4. Colapso de espacios múltiples y trim.

### Opción A: `unidecode` (más completa)

```python
import re
import unidecode

def quitar_acentos_unidecode(texto: str) -> str:
    return unidecode.unidecode(texto)
```

### Opción B: `str.translate` con tabla de acentos (sin dependencias extra)

```python
def quitar_acentos_translate(texto: str) -> str:
    tabla = str.maketrans({
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
        'ã': 'a', 'ñ': 'n', 'õ': 'o', 'ü': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'À': 'A', 'È': 'E', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U',
        'Â': 'A', 'Ê': 'E', 'Î': 'I', 'Ô': 'O', 'Û': 'U',
        'Ã': 'A', 'Ñ': 'N', 'Õ': 'O', 'Ü': 'U',
    })
    return texto.translate(tabla)
```

### Opción C: `unicodedata` (estándar de Python)

```python
import unicodedata

def quitar_acentos_unicodedata(texto: str) -> str:
    normalizado = unicodedata.normalize('NFD', texto)
    return ''.join(c for c in normalizado if unicodedata.category(c) != 'Mn')
```

### Función completa de normalización

```python
def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str) or not texto.strip():
        return ""
    texto = texto.lower()
    texto = unidecode.unidecode(texto)          # o quitar_acentos_translate(texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)      # elimina puntuación
    texto = re.sub(r'\s+', ' ', texto).strip()  # espacios múltiples + bordes
    return texto
```

**Nota:** `unidecode` transliterará caracteres latinos y dejará la `ñ` intacta en la mayoría de versiones; si prefieres conservar `ñ` y `Ñ` sin tocar, usa `str.translate` o `unicodedata`.

---

## 2. Expresiones regulares (regex) útiles para limpiar títulos

Texto corto de cine/series suele traer "ruido" entre paréntesis, menciones de doblaje, versiones "real action" y años sueltos.

### Ejemplos de transformaciones deseadas

| Entrada | Salida esperada |
|---|---|
| `Los juegos del hambre (todas)` | `Los juegos del hambre` |
| `Moana Real Action` | `Moana` |
| `Harry Potter y la piedra filosofal 2001` | `Harry Potter y la piedra filosofal` |
| `Avatar (español latino)` | `Avatar` |

### Regex recomendadas

```python
def limpiar_ruido_titulos(texto: str, quitar_numeros: bool = True) -> str:
    # 1. Aclaraciones entre paréntesis: "(todas)", "(español latino)", etc.
    texto = re.sub(r'\s*\([^)]*\)', ' ', texto)

    # 2. Versiones dobladas / real action (case-insensitive)
    texto = re.sub(r'\b(real|live)\s*action\b', ' ', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\b(doblaje|latino|español|castellano|subtitulado|sub)\b',
                   ' ', texto, flags=re.IGNORECASE)

    # 3. Números sueltos / años sueltos (evita borrar números dentro de palabras)
    if quitar_numeros:
        texto = re.sub(r'\b\d{1,4}\b', ' ', texto)

    # 4. Puntuación restante y colapso de espacios
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto
```

**Reglas clave:**
- `\([^)]*\)` captura todo lo que haya entre paréntesis, incluidos los paréntesis.
- `\b...\b` garantiza palabras completas (evita borrar "action" dentro de "reaction").
- `quitar_numeros=True` es seguro para años (ej. `2001`, `1999`), pero revisa si en tu dataset hay títulos con números significativos (ej. `Rápidos y furiosos 7`); en ese caso, usa `False` o conserva `\b\d+\b` solo cuando estén aislados al final.

---

## 3. Tratamiento de valores nulos / ambiguos en español

En encuestas o formularios universitarios es común encontrar respuestas como `"No tengo"`, `"Ninguna"`, `"No"`, `"N/A"`, `"Ninguno"`. Se recomienda estandarizarlas en una categoría `"Sin_Preferencia"` (o `NaN`, según el modelo).

### Lista de valores ambiguos en español

```python
import numpy as np
import pandas as pd

VALORES_AMBIGUOS_ES = {
    "no tengo", "ninguna", "ninguno", "no", "n/a", "na",
    "nan", "none", "null", "nil", "-", "—", "",
    " ", "s/n", "sin preferencia", "sin_preferencia",
    "no sé", "no se", "ninguna de las anteriores", "nd"
}

def estandarizar_valor_nulo(valor, categoria="Sin_Preferencia"):
    """
    Convierte valores ambiguos/nulos en una categoría estándar.
    Devuelve categoria si el valor es ambiguo; si no, devuelve el valor limpio.
    """
    if pd.isna(valor):
        return categoria
    s = str(valor).strip().lower()
    if s in VALORES_AMBIGUOS_ES or s == "":
        return categoria
    return s  # o str(valor).strip() si quieres conservar mayúsculas originales
```

### Aplicación sobre un DataFrame

```python
df['genero_favorito_limpio'] = df['genero_favorito'].apply(
    lambda v: estandarizar_valor_nulo(v, categoria="Sin_Preferencia")
)

# Alternativa vectorizada con replace (más rápido en datasets grandes)
df['genero_favorito_limpio'] = (
    df['genero_favorito']
    .astype(str)
    .str.strip()
    .str.lower()
    .replace(list(VALORES_AMBIGUOS_ES | {""}), np.nan)
    .fillna("Sin_Preferencia")
)
```

**Recomendación:** para modelos supervisados usa `"Sin_Preferencia"` como clase extra si hay suficientes ejemplos; si son muy pocos, mejor elimina esas filas o agrupa con la clase más frecuente.

---

## 4. Stopwords en español con NLTK

```python
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

stop_es = set(stopwords.words('spanish'))

# Ejemplo de stopwords en español que descarga NLTK (parcial):
# {'a', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con',
#  'conmigo', 'contra', 'cual', 'cuales', 'cualquier', 'cuyo', 'de',
#  'debe', 'debemos', 'desde', 'donde', 'e', 'el', 'ella', 'ellas',
#  'ellos', 'en', 'entre', 'era', 'erais', 'eran', 'eras', 'eres',
#  'es', 'esa', 'esas', 'ese', 'esos', 'esta', 'estas', 'este', 'estos',
#  'esto', 'estoy', 'etc', 'habia', 'habia', 'hasta', 'hay', 'hasta',
#  'le', 'les', 'lo', 'los', 'me', 'mi', 'mis', 'mientras', 'muy',
#  'más', 'mía', 'mías', 'mío', 'míos', 'nada', 'ni', 'no', 'nos',
#  'nosotras', 'nosotros', 'nuestra', 'nuestras', 'nuestro', 'nuestros',
#  'o', 'para', 'pero', 'por', 'porque', 'que', 'quien', 'quienes',
#  'se', 'sea', 'sean', 'seas', 'ser', 'será', 'serán', 'serías',
#  'si', 'siempre', 'siete', 'sino', 'siendo', 'sobre', 'sois',
#  'somos', 'son', 'soy', 'su', 'sus', 'suya', 'suyas', 'suyo',
#  'suyos', 'tá', 'también', 'tanto', 'te', 'tendremos', 'tendrá',
#  'tendrán', 'tendrías', 'tengo', 'tengamos', 'tengan', 'tengas',
#  'tenemos', 'tenido', 'tenidos', 'teniendo', 'tenéis', 'tenía',
#  'teníamos', 'tenían', 'tenías', 'tiene', 'tienen', 'tienes',
#  'toda', 'todas', 'todo', 'todos', 'tu', 'tus', 'tú', 'tuya',
#  'tuyas', 'tuyo', 'tuyos', 'un', 'una', 'uno', 'unos', 'vosotras',
#  'vosotros', 'voy', 'y', 'ya', 'yo'}
```

### Personalización para dominio cine/series

```python
# Opcional: quitar negadoras si el modelo depende de ellas (ej. "sin sangre")
# stop_es.discard("no")
# stop_es.discard("sin")
# stop_es.discard("ni")

# Añadir stopwords del dominio
stop_cine = {
    "pelicula", "película", "películas", "serie", "series",
    "capitulo", "capítulo", "capítulos", "temporada", "temporadas",
    "actor", "actriz", "actores", "actrices", "personaje", "personajes",
    "director", "directora", "doblaje", "episodio", "episodios"
}
stop_es = stop_es | stop_cine
```

**Nota:** NLTK no incluye lemmatizador por defecto para español; si necesitas lematización, considera `spacy` con el modelo `es_core_news_sm`. Para TF-IDF sobre texto corto, solo stopwords suele ser suficiente.

---

## 5. Vectorización TF-IDF (`TfidfVectorizer` de scikit-learn)

Para texto corto en español se recomienda `ngram_range=(1, 2)` para capturar expresiones como `"star wars"`, `"los juegos del hambre"`, `"ciencia ficcion"`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorizar_tfidf(textos_train, textos_test=None, stop_words=list(stop_es)):
    vectorizer = TfidfVectorizer(
        lowercase=False,            # ya normalizamos minúsculas antes
        stop_words=stop_words,
        max_features=5000,          # limita vocabulario; subir si hay muchos títulos
        ngram_range=(1, 2),         # unigramas + bigramas (clave en títulos)
        min_df=2,                   # ignora términos que aparecen en < 2 documentos
        max_df=0.90,                # ignora términos en > 90 % de los documentos
        sublinear_tf=True,          # escala sublineal de frecuencias (1 + log(tf))
        norm='l2',                  # normalización coseno por fila
        use_idf=True
    )

    X_train = vectorizer.fit_transform(textos_train)
    X_test = vectorizer.transform(textos_test) if textos_test is not None else None

    return vectorizer, X_train, X_test
```

### Parámetros típicos y por qué

| Parámetro | Valor sugerido | Razón |
|---|---|---|
| `max_features` | `3000`–`10000` | Controla dimensionalidad y ruido en texto corto. |
| `ngram_range` | `(1, 2)` | Captura títulos compuestos sin explotar el vocabulario. |
| `min_df` | `2` | Elimina hapax legomena (palabras que aparecen una sola vez). |
| `max_df` | `0.85`–`0.95` | Elimina términos muy comunes que no discriminan. |
| `sublinear_tf` | `True` | Reduce el sesgo de títulos largos vs. cortos. |
| `norm` | `'l2'` | Hace comparable el coseno entre vectores. |

**Tip para texto muy corto y con errores de escritura:** añadir char n-grams `(3, 5)` mediante un segundo `TfidfVectorizer(analyzer='char', ngram_range=(3, 5))` y concatenar (o usar `FeatureUnion`).

---

## 6. One-Hot Encoding para variable objetivo categórica (género favorito)

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Opción A: pandas (rápida y legible)
Y = pd.get_dummies(
    df['genero_favorito_limpio'],
    prefix='gen',
    dummy_na=False
)

# Opción B: scikit-learn (mejor para pipelines y producción)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
Y = ohe.fit_transform(df[['genero_favorito_limpio']])
# ohe.categories_ da el orden de las columnas
```

**Recomendación:** si el modelo es un clasificador (no requiere one-hot en `y`), usa directamente las etiquetas strings con `LabelEncoder` o deja `y` como array de strings; `scikit-learn` las codifica internamente. Usa `OneHotEncoder` solo si necesitas la matriz binaria explícita (ej. redes neuronales con salida multiclase o análisis de asociación).

---

## 7. Modelos supervisados recomendados para matrices TF-IDF dispersas

La matriz TF-IDF es **alta dimensión y dispersa** (la mayoría de celdas son cero). No todos los modelos se comportan igual.

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### a) Multinomial Naive Bayes (baseline rápido)

```python
nb = MultinomialNB(alpha=0.1)   # alpha = suavizado de Laplace
nb.fit(X_train, y_train)
```

- **Por qué:** funciona bien con conteos/frecuencias de texto, es muy rápido y requiere poco ajuste. Aunque TF-IDF no es conteo puro, en la práctica suele ser un excelente baseline para clasificación de texto.
- **Contra:** supone independencia entre características (ingenua).

### b) LinearSVC (recomendado para texto disperso)

```python
svc = LinearSVC(
    C=1.0,
    class_weight='balanced',    # útil si hay clases desbalanceadas
    max_iter=5000,
    random_state=42
)
svc.fit(X_train, y_train)
```

- **Por qué:** máquinas de vectores de soporte lineales son el estándar de oro para clasificación de texto de alta dimensión y dispersa. Buen equilibrio entre precisión y velocidad.
- **Contra:** no da probabilidades directamente (usa `CalibratedClassifierCV` si las necesitas).

### c) KNN (usar con precaución)

```python
knn = make_pipeline(
    StandardScaler(with_mean=False),   # con_mean=False para matrices dispersas
    KNeighborsClassifier(n_neighbors=5, metric='cosine', weights='distance')
)
knn.fit(X_train, y_train)
```

- **Por qué:** simple e interpretable; la distancia coseno es apropiada para texto.
- **Contra:** maldad de la dimensionalidad; lento en predicción con muchos datos y alta dimensión. Reduce primero con PCA/TruncatedSVD si usas KNN.

### d) Random Forest

```python
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
```

- **Por qué:** captura interacciones no lineales, es robusto a overfitting si limitas profundidad y ofrece importancia de características.
- **Contra:** en matrices muy dispersas y altas dimensiones puede overfitear más que `LinearSVC`; más lento de entrenar.

### e) MLPClassifier (red neuronal multicapa)

```python
mlp = make_pipeline(
    StandardScaler(with_mean=False),
    MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        max_iter=500
    )
)
mlp.fit(X_train, y_train)
```

- **Por qué:** puede aprender relaciones no lineales complejas; `early_stopping` evita overfitting.
- **Contra:** requiere escalado de entrada, es más lento, necesita más datos y ajuste de hiperparámetros que `LinearSVC` o `NB`.

### Comparativa rápida (sugerencia de orden de prueba)

1. `MultinomialNB` → baseline rápido.
2. `LinearSVC` → suele ser el mejor para texto disperso.
3. `RandomForest` → si sospechas interacciones no lineales.
4. `KNN` → solo si reduces dimensiones primero.
5. `MLP` → último, si tienes datos suficientes y tiempo de ajuste.

---

## 8. Modelo no supervisado: K-Means + método del codo + PCA/t-SNE

Útil para explorar si los títulos se agrupan naturalmente (sin usar el género como etiqueta).

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- Método del codo (inercia vs k) ---
inercia = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    km.fit(X_train)          # X_train es la matriz TF-IDF dispersa
    inercia.append(km.inertia_)

# Gráfico del codo
plt.figure(figsize=(8, 4))
plt.plot(K_range, inercia, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia (suma de distancias al cuadrado)')
plt.title('Método del codo para K-Means')
plt.grid(True, alpha=0.3)
plt.show()

# Nota visual: el "codo" es donde la bajada de inercia se aplana.

# --- Entrenar con el k elegido (ej. k=4) ---
k_optimo = 4   # cambiar según el gráfico del codo
kmeans = KMeans(
    n_clusters=k_optimo,
    init='k-means++',
    n_init=10,
    random_state=42
)
etiquetas_clusters = kmeans.fit_predict(X_train)

# --- Visualización con PCA (recomendado para matrices dispersas grandes) ---
# TruncatedSVD es la versión dispersa de PCA
reductor = TruncatedSVD(n_components=2, random_state=42)
X_2d = reductor.fit_transform(X_train)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=etiquetas_clusters,
                      cmap='viridis', alpha=0.7, s=50)
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title(f'Clusters K-Means (k={k_optimo}) visualizados con TruncatedSVD/PCA')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# --- t-SNE opcional (solo para datasets pequeños/medianos) ---
# t-SNE requiere matriz densa; convierte solo si X_train no es enorme.
# X_dense = X_train.toarray()
# tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
# X_tsne = tsne.fit_transform(X_dense)
```

**Recomendaciones:**
- Usa `TruncatedSVD` (no `PCA` denso) sobre matrices TF-IDF dispersas para evitar consumir mucha RAM.
- El método del codo es subjetivo; complementa con `silhouette_score` si quieres una métrica numérica.
- `n_init=10` evita soluciones locales pobres (por defecto en sklearn moderno es `10` o `'auto'`).

---

## 9. Métricas de evaluación supervisada

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

y_pred = modelo.predict(X_test)   # modelo ya entrenado

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision (weighted):", precision_score(
    y_test, y_pred, average='weighted', zero_division=0
))
print("Recall (weighted):", recall_score(
    y_test, y_pred, average='weighted', zero_division=0
))
print("F1-score (weighted):", f1_score(
    y_test, y_pred, average='weighted', zero_division=0
))

print("\nReporte de clasificación:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
print("Matriz de confusión:\n", cm)
```

### Qué promedio usar en precision/recall/f1

| Promedio | Cuándo usarlo |
|---|---|
| `'macro'` | Todas las clases tienen igual importancia, aunque haya pocas muestras de alguna. |
| `'weighted'` | Hay desbalance de clases; pondera por soporte de cada clase. Recomendado para género favorito si algunos géneros son minoritarios. |
| `'micro'` | Cuando quieres peso global (equivale a accuracy en multiclasificación). |

### Matriz de confusión visual (opcional)

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de confusión')
plt.show()
```

---

## 10. Pipeline completo listo para usar (ejemplo integrado)

```python
# ============================================================
# PIPELINE NLP ESPAÑOL - TÍTULOS DE PELÍCULAS / SERIES / ACTORES
# ============================================================
import re
import numpy as np
import pandas as pd
import unidecode
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ------------------------------------------------------------------
# 0. Descargar stopwords de español
# ------------------------------------------------------------------
nltk.download('stopwords', quiet=True)
STOP_ES = set(stopwords.words('spanish'))
STOP_CINE = {"pelicula", "película", "películas", "serie", "series",
             "capitulo", "capítulo", "temporada", "actor", "actriz",
             "personaje", "director", "doblaje"}
STOP_ES = STOP_ES | STOP_CINE

# ------------------------------------------------------------------
# 1. Funciones de limpieza
# ------------------------------------------------------------------
VALORES_AMBIGUOS = {
    "no tengo", "ninguna", "ninguno", "no", "n/a", "na", "nan",
    "none", "null", "nil", "-", "—", "", " ", "s/n",
    "sin preferencia", "no sé", "no se"
}

def limpiar_titulo(texto: str) -> str:
    if not isinstance(texto, str) or not texto.strip():
        return ""
    texto = texto.lower()
    texto = unidecode.unidecode(texto)          # quita acentos
    texto = re.sub(r'\s*\([^)]*\)', ' ', texto) # quita (todas), (español), etc.
    texto = re.sub(r'\b(real|live)\s*action\b', ' ', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\b(doblaje|latino|español|castellano|subtitulado|sub)\b',
                   ' ', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\b\d{1,4}\b', ' ', texto)  # quita años/números sueltos
    texto = re.sub(r'[^\w\s]', ' ', texto)      # quita puntuación
    texto = re.sub(r'\s+', ' ', texto).strip()  # colapsa espacios
    return texto

def limpiar_categoria(valor) -> str:
    if pd.isna(valor):
        return "Sin_Preferencia"
    s = str(valor).strip().lower()
    if s in VALORES_AMBIGUOS or s == "":
        return "Sin_Preferencia"
    return s

# ------------------------------------------------------------------
# 2. Carga y preparación de datos (ejemplo con DataFrame propio)
# ------------------------------------------------------------------
# df = pd.read_csv("tus_datos.csv")
# df['titulo_limpio'] = df['titulo'].apply(limpiar_titulo)
# df['genero_limpio'] = df['genero_favorito'].apply(limpiar_categoria)

# Para que el ejemplo sea autónomo, creamos un dataset ilustrativo:
data = pd.DataFrame({
    'titulo': [
        'Los juegos del hambre (todas)', 'Moana Real Action',
        'Harry Potter y la piedra filosofal 2001', 'Avatar (español latino)',
        'Rápidos y Furiosos 7', 'Interestelar', 'No tengo',
        'Ninguna', 'N/A', 'El padrino', 'El padrino parte II'
    ],
    'genero_favorito': [
        'Acción', 'Acción', 'Aventura', 'Ciencia ficción',
        'Acción', 'Ciencia ficción', 'no', 'ninguna',
        'n/a', 'Drama', 'Drama'
    ]
})

df = data.copy()
df['titulo_limpio'] = df['titulo'].apply(limpiar_titulo)
df['genero_limpio'] = df['genero_favorito'].apply(limpiar_categoria)

# ------------------------------------------------------------------
# 3. Vectorización TF-IDF
# ------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    lowercase=False,
    stop_words=list(STOP_ES),
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1,          # en datasets pequeños usa 1; en grandes usa 2+
    max_df=0.95,
    sublinear_tf=True,
    norm='l2'
)

X = vectorizer.fit_transform(df['titulo_limpio'])
print("Forma de la matriz TF-IDF:", X.shape)

# ------------------------------------------------------------------
# 4. Codificación de objetivo (etiquetas string directas)
# ------------------------------------------------------------------
y = df['genero_limpio'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 5. Entrenar y evaluar modelos
# ------------------------------------------------------------------
modelos = {
    "MultinomialNB": MultinomialNB(alpha=0.1),
    "LinearSVC": LinearSVC(C=1.0, class_weight='balanced',
                           max_iter=5000, random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=None, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1
    )
}

resultados = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    resultados[nombre] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    print(f"\n=== {nombre} ===")
    print(classification_report(y_test, y_pred, zero_division=0))

# ------------------------------------------------------------------
# 6. (Opcional) One-Hot Encoding de la variable objetivo
# ------------------------------------------------------------------
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
Y_onehot = ohe.fit_transform(df[['genero_limpio']])
print("\nOne-Hot Encoding shape:", Y_onehot.shape)
print("Categorías:", ohe.categories_)
```

---

## 11. Recomendaciones específicas para texto corto en español

1. **Bigramas son obligatorios:** títulos como `"star wars"` o `"ciencia ficcion"` pierden significado si se separan en unigramas. Usa `ngram_range=(1, 2)` como mínimo.
2. **Cuidado con `min_df`:** en datasets universitarios pequeños, `min_df=2` puede dejar el vocabulario vacío. Prueba `min_df=1` y ajusta según el tamaño.
3. **Desbalance de géneros:** algunos géneros (ej. `"Terror"`) aparecerán más que otros. Usa `class_weight='balanced'` en `LinearSVC`/`RandomForest` y métrica `weighted`.
4. **No apliques KNN directamente sobre TF-IDF de miles de dimensiones:** primero reduce con `TruncatedSVD` o usa `LinearSVC`.
5. **Acentos y `ñ`:** quitar acentos ayuda a unificar `"película"` y `"pelicula"`, pero conserva la `ñ` si tu vocabulario la necesita (evita que `"año"` colapse con `"ano"`).
6. **Validación cruzada estratificada:** usa `StratifiedKFold` para que cada fold tenga proporción similar de géneros.
7. **Interpretabilidad:** `LinearSVC` y `MultinomialNB` te permiten inspeccionar coeficientes/probabilidades por palabra; útil para el informe universitario.

---

## Referencias internas del proyecto

- `requirements.txt` sugerido: `pandas`, `scikit-learn`, `nltk`, `unidecode`, `matplotlib`, `seaborn`, `numpy`.
- Todos los fragmentos están listos para copiar y pegar en un notebook Jupyter o script `.py`.
