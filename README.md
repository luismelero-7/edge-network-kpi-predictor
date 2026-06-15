# edge-network-kpi-predictor

Repositorio asociado al Trabajo de Fin de Grado **“Diseño y evaluación de un servicio inteligente de predicción de información de red para sistemas de Edge Computing”**.

El proyecto desarrolla y evalúa un servicio de predicción de indicadores de rendimiento de red en entornos de **5G Edge Computing**, utilizando datos generados con el simulador **MintEDGE** y modelos de Machine Learning, incluyendo modelos tabulares y secuenciales.

## Contenido del repositorio

```text
edge-network-kpi-predictor/
├── dataset_builder/
│   ├── build_dataset.py
│   └── data/
│       ├── DATOS_BRUTOS_ONEDRIVE.txt
│       └── DATOS_PROCESADOS_ONEDRIVE.txt
├── notebook/
│   ├── TFG_LuisMeleroJareno_Predictor.ipynb
│   └── utils.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Datos

Los datos utilizados en este trabajo no se incluyen directamente en el repositorio debido a su tamaño. En su lugar, se proporcionan enlaces externos mediante OneDrive.

### Datos brutos

Los datos brutos proceden de las simulaciones generadas con MintEDGE y constituyen la entrada original utilizada para construir el conjunto de datos homogéneo.

El enlace de descarga se encuentra en:

```text
dataset_builder/data/DATOS_BRUTOS_ONEDRIVE.txt
```

Estos datos pueden utilizarse junto con el script `dataset_builder/build_dataset.py` para reconstruir el conjunto de datos procesado.

### Datos procesados

También se proporciona el conjunto de datos procesado utilizado directamente en los experimentos del notebook. Este conjunto corresponde a la versión homogénea generada a partir de los datos brutos.

El enlace de descarga se encuentra en:

```text
dataset_builder/data/DATOS_PROCESADOS_ONEDRIVE.txt
```

Estos datos permiten ejecutar el flujo experimental sin necesidad de reconstruir el conjunto de datos desde los ficheros brutos.

## Construcción del conjunto de datos

El script encargado de construir el conjunto de datos homogéneo se encuentra en:

```text
dataset_builder/build_dataset.py
```

Este script procesa las salidas originales de MintEDGE, homogeneiza los escenarios de simulación y genera el conjunto de datos utilizado posteriormente para el entrenamiento y evaluación de los modelos predictivos.

## Notebook experimental

El notebook principal se encuentra en:

```text
notebook/TFG_LuisMeleroJareno_Predictor.ipynb
```

Este notebook incluye el flujo de experimentación seguido en el trabajo: carga del conjunto de datos, preparación de variables, selección de características, entrenamiento de modelos, evaluación en validación y prueba, comparación de modelos y generación de resultados.

Debido al tamaño del archivo y a que conserva las salidas de ejecución de los experimentos, es posible que GitHub no pueda previsualizarlo correctamente desde el navegador. En ese caso, se recomienda descargarlo mediante la opción Download raw file y abrirlo localmente con Jupyter Notebook, JupyterLab o Visual Studio Code.

## Modelos y resultados

Los experimentos incluyen modelos tabulares, como DecisionTree, RandomForest, XGBoost y LightGBM, y modelos secuenciales, como LSTM y CNN.

Los resultados obtenidos se emplean para construir las tablas y figuras incluidas en la memoria del Trabajo de Fin de Grado.

## Instalación de dependencias

Se recomienda crear un entorno virtual e instalar las dependencias mediante:

```bash
pip install -r requirements.txt
```

## Autor

**Luis Melero Jareño**
| Grado en Ingeniería Informática
| Escuela Superior de Ingeniería Informática
| Universidad de Castilla-La Mancha
| Curso 2025/2026
