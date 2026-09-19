# edge-network-kpi-predictor

> **Trabajo de Fin de Grado** · Grado en Ingeniería Informática · UCLM · Curso 2025/2026

**"Diseño y evaluación de un servicio inteligente de predicción de información de red para sistemas de Edge Computing"**

El proyecto desarrolla y evalúa un servicio de predicción de indicadores de rendimiento de red en entornos de **5G Edge Computing**, utilizando datos generados con el simulador **MintEDGE** y modelos de Machine Learning, incluyendo modelos tabulares y secuenciales.

---

## 📄 Documentación

| Documento | Descripción |
|-----------|-------------|
| [Memoria del TFG](TFG_LuisMeleroJare%C3%B1o.pdf) | Documento completo del trabajo (121 páginas) |
| [Presentación de defensa](Presentacion_TFG_LuisMeleroJare%C3%B1o.pdf) | Presentación utilizada en la defensa |

---

## 📁 Estructura del repositorio

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
├── TFG_LuisMeleroJareño.pdf
├── Presentacion_TFG_LuisMeleroJareño.pdf
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🗃️ Datos

Los datos no se incluyen en el repositorio por su tamaño. Se proporcionan enlaces externos vía OneDrive.

| Tipo | Enlace |
|------|--------|
| Datos brutos | `dataset_builder/data/DATOS_BRUTOS_ONEDRIVE.txt` |
| Datos procesados | `dataset_builder/data/DATOS_PROCESADOS_ONEDRIVE.txt` |

**Datos brutos:** proceden de las simulaciones generadas con MintEDGE y constituyen la entrada original para construir el conjunto de datos homogéneo. Pueden utilizarse junto con `dataset_builder/build_dataset.py` para reconstruir el dataset procesado.

**Datos procesados:** conjunto de datos homogéneo utilizado directamente en los experimentos del notebook. Permite ejecutar el flujo experimental sin necesidad de reconstruir el dataset desde los ficheros brutos.

---

## 🔧 Construcción del conjunto de datos

```text
dataset_builder/build_dataset.py
```

Este script procesa las salidas originales de MintEDGE, homogeneiza los escenarios de simulación y genera el conjunto de datos utilizado para el entrenamiento y evaluación de los modelos predictivos.

---

## 🤖 Modelos evaluados

| Tipo | Modelos |
|------|---------|
| Tabulares | DecisionTree · RandomForest · XGBoost · LightGBM |
| Secuenciales | LSTM · CNN 1D |

Los resultados obtenidos se emplean para construir las tablas y figuras incluidas en la memoria del TFG.

---

## 📓 Notebook experimental

```text
notebook/TFG_LuisMeleroJareno_Predictor.ipynb
```

Incluye el flujo de experimentación completo: carga del conjunto de datos, preparación de variables, selección de características, entrenamiento de modelos, evaluación en validación y prueba, comparación de modelos y generación de resultados.

> Si GitHub no puede previsualizarlo por su tamaño, descárgalo con **Download raw file** y ábrelo localmente con Jupyter Notebook, JupyterLab o Visual Studio Code.

---

## 🚀 Instalación

```bash
pip install -r requirements.txt
```

---

## 👤 Autor

**Luis Melero Jareño**  
Grado en Ingeniería Informática · Intensificación en Ingeniería de Computadores  
Escuela Superior de Ingeniería Informática · Universidad de Castilla-La Mancha · Curso 2025/2026

**Tutores:** Gabriel Cebrián Márquez · Estefanía Coronado Calero