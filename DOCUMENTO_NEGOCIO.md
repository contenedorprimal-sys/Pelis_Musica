# Documento de Negocio - Sistemas Inteligentes

## 1. Resumen ejecutivo

El proyecto consiste en el desarrollo de un sistema de inteligencia artificial que analiza encuestas de preferencias de entretenimiento (**películas, series, actores, géneros** y **hábitos musicales, estado de ánimo y concentración**) para predecir, de forma automática, el perfil de consumo de cada usuario.

El modelo permite recomendar contenidos personalizados y anticipar cómo influye la música en la concentración de los estudiantes/trabajadores, con aplicación directa en plataformas de streaming, servicios de recomendación y herramientas de productividad.

---

## 2. Problema de negocio

### 2.1 Problemática identificada

Las encuestas recopilan opiniones en **texto libre no estructurado**, lo que genera varios problemas:

- **Respuestas ruidosas o fuera de contexto** (ej. *"Hamburguesa con papas"* en la columna de película favorita).
- **Valores ambiguos y nulos** (*"No tengo"*, *"Ninguna"*, *"No"*, *"N/A"*), que dificultan el análisis automático.
- **Alta dimensionalidad de términos** y variantes de escritura de un mismo título o género (*"Acción"*, *"Acción."*, *"Ciencia Ficción"*, *"Ciencia ficción"*).
- **Falta de estructura para toma de decisiones:** las plataformas no pueden segmentar a sus usuarios ni predecir sus preferencias sin procesar primero estos datos.

### 2.2 Oportunidad de negocio

Existe una oportunidad clara para:

1. **Plataformas de streaming (video y música):** crear recomendaciones personalizadas basadas en los gustos, repudios y perfil emocional de cada usuario.
2. **Herramientas de productividad y estudio:** predecir si escuchar música ayuda o distrae la concentración de un usuario, y sugerir playlists contextuales (enfoque, relajación, gimnasio).
3. **Análisis de mercado:** identificar arquetipos de espectador/oyente (cinéfilo mainstream, amante del terror, oyente funcional para estudio, etc.) para campañas de marketing y adquisición de contenido.

---

## 3. Propuesta de solución

Se desarrolló un pipeline de ciencia de datos que:

1. **Limpia y normaliza** los datos de texto (expresiones regulares, eliminación de acentos, stopwords en español, tratamiento de nulos y atípicos).
2. **Vectoriza el texto** con TF-IDF para convertirlo en matriz numérica usable por modelos de ML.
3. **Entrena modelos de aprendizaje supervisado** (Naive Bayes, SVM, KNN, Random Forest, Red Neuronal MLP) para clasificar el género favorito de un usuario y predecir si la música le ayuda o distrae al concentrarse.
4. **Aplica aprendizaje no supervisado** (K-Means + PCA) para segmentar usuarios en perfiles o arquetipos de espectador/oyente.
5. **Expone los resultados** mediante una interfaz web interactiva (Streamlit) donde un usuario ingresa sus preferencias y obtiene una predicción en tiempo real.

---

## 4. Propuesta de valor

| Dimensión | Valor entregado |
|-----------|-----------------|
| **Personalización** | Recomendaciones de contenido adaptadas al perfil y gustos de cada usuario. |
| **Toma de decisiones basada en datos** | Segmentación automática de usuarios en arquetipos accionables para marketing y adquisición de catálogo. |
| **Productividad** | Predicción del efecto de la música en la concentración, útil para apps de estudio y bienestar. |
| **Experiencia de usuario** | Interfaz web sencilla que brinda predicciones en tiempo real sin requerir conocimientos técnicos. |
| **Escalabilidad** | Arquitectura modular y containerizada (Docker) que permite desplegar en cualquier nube o servidor. |

---

## 5. Beneficios esperados

- **Mayor retención y satisfacción del usuario** al recibir contenido alineado con sus preferencias.
- **Reducción del esfuerzo manual** de análisis de encuestas (el procesamiento NLP automatiza la clasificación de miles de respuestas de texto libre).
- **Mejora en la efectividad de campañas** al dirigir contenido a segmentos específicos (arquetipos de espectador).
- **Herramienta de bienestar digital** que ayuda a usuarios a optimizar su concentración al estudiar o trabajar.
- **Despliegue rápido y reproducible** en entornos de nube (AWS, Azure) o clústeres, gracias a Docker.

---

## 6. Modelos de negocio potenciales

1. **B2B (plataformas de streaming):** licencia del motor de recomendación y segmentación de usuarios.
2. **B2B2C (apps de productividad/estudio):** integración como módulo de predicción de concentración y sugerencia de playlists contextuales.
3. **SaaS (software como servicio):** plataforma web de análisis de encuestas de entretenimiento con tableros y predicciones.

---

## 7. Métricas de éxito

- Precisión (F1-score) del modelo de clasificación de género favorito y de influencia en la concentración.
- Coeficiente de silueta y separación de los clusters de usuarios (perfiles de espectador/oyente).
- Número de usuarios que reciben una predicción válida a través de la interfaz web.
- Tiempo de respuesta de la predicción en tiempo real (< 2 segundos objetivo).
- Adopción del tablero de resultados por parte del equipo de producto/marketing.

---

## 8. Conclusión

El proyecto convierte encuestas de texto libre no estructurado en un activo de datos accionable, aplicando técnicas de NLP, machine learning supervisado y no supervisado, y desplegando el resultado como una aplicación web containerizada. Genera valor tanto para plataformas de entretenimiento (recomendaciones personalizadas y segmentación) como para herramientas de productividad (gestión de la concentración mediante música).
