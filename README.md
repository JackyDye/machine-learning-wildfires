# 🔥 Predicción de Incendios Forestales en EE.UU.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Poetry](https://img.shields.io/badge/Poetry-Dependencias-60A5FA?logo=poetry&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Proyecto final desarrollado en **Talento Tech**, orientado a la clasificación de registros con y sin incendios forestales a partir de variables climáticas y geográficas de Estados Unidos.

El trabajo recorre el proceso completo en un notebook: descarga del dataset, análisis exploratorio, limpieza, construcción de pipelines y comparación de tres modelos de clasificación. El foco está en comprender el desbalance de clases y el equilibrio entre detectar incendios y generar falsas alarmas.

📓 **[Ver el notebook completo](pipeline.ipynb)**

---

## Objetivo

Comparar la capacidad de distintos modelos de clasificación para distinguir registros asociados a incendios forestales y analizar sus limitaciones mediante métricas que van más allá del accuracy.

- Explorar la distribución y calidad de los datos.
- Integrar el mismo preprocesamiento y cada modelo en pipelines de scikit-learn.
- Comparar regresión logística, árbol de decisión y Random Forest.
- Evaluar accuracy, precisión, recall y F1-score para la clase positiva.

## Dataset

Fuente: **[US Wildfire Dataset (2014–2025)](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset)**, disponible en Kaggle.

| Característica | Descripción |
|---|---|
| Cobertura | Estados Unidos; período indicado por la fuente: 2014–2025 |
| Tamaño del dataset analizado | 9.509.925 registros y 19 columnas antes de la limpieza |
| Variable objetivo | `Wildfire`: `Yes` / `No`, transformada a `1` / `0` |
| Variables disponibles | Ubicación, fecha, precipitación, temperatura, humedad, viento e índices ambientales |
| Distribución de clases | Aproximadamente un 5 % de registros positivos |

El CSV se descarga automáticamente al ejecutar el notebook por primera vez y se guarda en `data/Wildfire_Dataset.csv`. La carpeta `data/` se genera durante la descarga y está excluida de Git mediante `.gitignore`.

## ¿Qué contiene el notebook?

1. **Exploración:** estadísticas descriptivas, valores faltantes y distribución de clases.
2. **Limpieza:** conversión de fechas y de la variable objetivo, eliminación de duplicados y revisión de registros con el valor `32767`.
3. **Preparación:** creación de componentes de fecha y selección de 14 predictores.
4. **Entrenamiento:** división estratificada en 70 % para entrenamiento y 30 % para prueba, con `random_state=42`.
5. **Pipelines:** imputación por mediana, estandarización y tres clasificadores con ponderación balanceada de clases.
6. **Comparación inicial:** entrenamiento de regresión logística, árbol de decisión y Random Forest con el mismo conjunto de datos y las mismas métricas.

## Comparación inicial de modelos

Resultados obtenidos sobre el conjunto de prueba. Precisión, recall y F1-score corresponden a la clase positiva: incendio (`1`).

| Modelo | Accuracy | Precisión | Recall | F1-score |
|---|---:|---:|---:|---:|
| Regresión logística | 57,61 % | 7,23 % | 59,32 % | 12,90 % |
| Árbol de decisión | **61,41 %** | **8,66 %** | 65,90 % | **15,30 %** |
| Random Forest | 60,23 % | 8,46 % | **66,37 %** | 15,01 % |

En esta primera comparación no hay un modelo claramente superior en todas las métricas. Random Forest alcanza el mayor recall y detecta la mayor proporción de incendios positivos, mientras que el árbol de decisión obtiene los mejores valores de accuracy, precisión y F1-score.

La diferencia de recall entre ambos modelos es de sólo `0,47` puntos porcentuales. Por eso todavía no es suficiente para elegir un modelo definitivo: falta analizar su estabilidad y observar con mayor detalle los falsos positivos y falsos negativos.

**El principal aprendizaje es metodológico:** en un dataset desbalanceado, una sola métrica no alcanza para evaluar un modelo. Los tres clasificadores mantienen una precisión muy baja para la clase positiva, por lo que su rendimiento actual no es adecuado para un sistema operativo de alertas.

## Estructura del repositorio

```text
machine-learning-wildfires/
├── pipeline.ipynb             # Análisis, entrenamiento y evaluación
├── data/                      # Generada localmente; excluida de Git
│   └── Wildfire_Dataset.csv    # Dataset descargado desde Kaggle
├── .gitattributes             # Configuración de archivos para Git
├── .gitignore                 # Exclusiones de Git
├── LICENSE                    # Licencia del código
├── poetry.lock                # Versiones resueltas de las dependencias
├── poetry.toml                # Entorno virtual dentro del proyecto
├── pyproject.toml             # Configuración y dependencias
└── README.md
```

## Stack tecnológico

| Herramienta | Uso |
|---|---|
| Python 3.12 | Lenguaje del proyecto |
| pandas | Exploración, limpieza y transformación de datos |
| scikit-learn | Preprocesamiento, entrenamiento y evaluación |
| Matplotlib / Seaborn | Visualizaciones y análisis de resultados |
| KaggleHub | Descarga del dataset |
| Jupyter | Ejecución interactiva del notebook |
| Poetry | Gestión del entorno virtual y las dependencias |

---

## Instalación y ejecución

### Requisitos previos

- **[Python 3.12](https://www.python.org/downloads/)** instalado. El proyecto requiere una versión `3.12.x`.
- **[Poetry](https://python-poetry.org/docs/#installation)** instalado; usar una versión reciente compatible con los grupos de dependencias de `pyproject.toml`.
- **[Git](https://git-scm.com/downloads)** para clonar el repositorio.
- Conexión a Internet para instalar dependencias y descargar el dataset por primera vez.

> **Recursos:** el archivo tiene aproximadamente 1 GB. Reservá varios GB de espacio libre para el dataset y el entorno virtual. El notebook carga todos los registros y crea copias y transformaciones en memoria, por lo que puede consumir varios GB de RAM. El tiempo de ejecución depende del equipo y aumenta al entrenar los tres modelos.

### 1. Clonar el repositorio

```bash
git clone https://github.com/JackyDye/machine-learning-wildfires.git
cd machine-learning-wildfires
```

### 2. Preparar el entorno e instalar las dependencias

Desde la carpeta raíz del proyecto:

```bash
poetry env use 3.12
poetry install --with dev
```

Poetry crea el entorno `.venv/` e instala las dependencias del proyecto y las herramientas para ejecutar notebooks, utilizando `poetry.lock`. No hace falta instalar las librerías una por una.

Si Poetry no encuentra Python 3.12, verificá que esté instalado y pasá la ruta de su ejecutable a `poetry env use`.

### 3. Abrir el notebook

```bash
poetry run jupyter lab pipeline.ipynb
```

Jupyter Lab abrirá el navegador. Si no lo hace, abrí la URL local que aparece en la terminal. Mantené esa terminal abierta mientras trabajás.

**Ejecutá este comando desde la raíz del repositorio:** el notebook utiliza `Path.cwd()` para definir dónde guardar y buscar la carpeta `data/`.

### 4. Ejecutar las celdas

Seleccioná el kernel de Python del entorno del proyecto y ejecutá las celdas en orden, o utilizá **Run → Run All Cells** para correr el análisis completo.

- Si el CSV no existe, el notebook lo descarga desde Kaggle y crea `data/` automáticamente.
- Si ya existe en `data/Wildfire_Dataset.csv`, reutiliza el archivo local.
- Las tablas, métricas y gráficos se muestran dentro del notebook.

Para un reinicio completo, utilizá **Kernel → Restart Kernel and Run All Cells**.

También podés abrir `pipeline.ipynb` en VS Code con las extensiones Python y Jupyter, seleccionando como kernel el Python de `.venv/`.

### Problemas frecuentes

| Situación | Qué revisar |
|---|---|
| `poetry` no se reconoce | Verificá su instalación y abrí una nueva terminal para actualizar el `PATH`. |
| Falta una librería o Jupyter | Ejecutá `poetry install --with dev` y comprobá que el notebook use el entorno `.venv/`. |
| El CSV se busca en otra carpeta | Cerrá Jupyter y volvé a iniciarlo desde `machine-learning-wildfires/`. |
| La descarga falla por autenticación o consentimiento | Consultá el acceso al dataset en Kaggle y la [autenticación de KaggleHub](https://github.com/Kaggle/kagglehub#authenticate). |
| El kernel se cierra o aparece `MemoryError` | Revisá la RAM disponible: el análisis carga el dataset completo y genera copias en memoria. |

Referencias: [uso básico de Poetry](https://python-poetry.org/docs/basic-usage/) y [descarga de datasets con KaggleHub](https://github.com/Kaggle/kagglehub#download-dataset).

## Alcance y limitaciones

- **Comparación inicial:** los resultados actuales provienen de una única división entre entrenamiento y prueba y todavía deben validarse con técnicas adicionales.
- **Desbalance de clases:** ponderar las clases durante el entrenamiento no garantiza una buena precisión ni elimina las falsas alarmas.
- **Validación temporal y geográfica pendiente:** la división actual es aleatoria y estratificada; no mide por separado el desempeño en fechas futuras o ubicaciones no vistas.
- **Uso del conjunto de prueba:** esta etapa lo utiliza para una comparación exploratoria. Una selección formal debería usar validación y reservar el test para la evaluación final.
- **Versionado del dataset:** las dependencias están fijadas en `poetry.lock`, pero la descarga solicita la versión disponible en Kaggle sin fijar una versión concreta. Si la fuente cambia, los resultados pueden variar.

## Próximos pasos

La rama `main` conserva la primera versión basada en regresión logística. Esta rama incorpora la comparación inicial de modelos y continuará ampliando su evaluación.

- Incorporar reportes de clasificación y matrices de confusión para cada modelo.
- Aplicar validación cruzada y comparar la estabilidad de los resultados.
- Agregar curvas ROC y precisión-recall junto con AUC y average precision.
- Revisar la selección de variables y considerar una validación temporal o geográfica.
- Extraer funciones reutilizables del notebook y agregar pruebas.

## Licencia

El código se distribuye bajo la [licencia MIT](LICENSE). El dataset pertenece a su fuente original; consultá sus condiciones de uso en la [página de Kaggle](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset).

## 👤 Autor

**Augusto Valles · JackyDye**  
[GitHub](https://github.com/JackyDye) · [LinkedIn](https://www.linkedin.com/in/augusto-valles/)

*Proyecto final de Talento Tech · Clasificación binaria y evaluación de modelos de machine learning.*
