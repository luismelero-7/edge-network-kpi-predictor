from IPython.display import HTML, display


# ─────────────────────────────────────────────
# ESTILOS DE TÍTULOS
# ─────────────────────────────────────────────

def titulo_capitulo(texto, subtexto=""):
    display(HTML(f"""
    <div style="background:#1F4E79;padding:18px 24px;border-radius:10px;margin:30px 0 8px 0;border-left:6px solid #2E75B6">
        <h1 style="color:white;margin:0;font-size:22px;font-weight:600">{texto}</h1>
        {"<p style='color:#9DC3E6;margin:6px 0 0 0;font-size:13px;font-weight:400'>" + subtexto + "</p>" if subtexto else ""}
    </div>
    """))

def titulo_seccion(texto, subtexto=""):
    display(HTML(f"""
    <div style="background:#2E75B6;padding:14px 20px;border-radius:8px;margin:20px 0 8px 0">
        <h2 style="color:white;margin:0;font-size:17px;font-weight:500">{texto}</h2>
        {"<p style='color:#BDD7EE;margin:4px 0 0 0;font-size:12px;font-weight:400'>" + subtexto + "</p>" if subtexto else ""}
    </div>
    """))

def titulo_subseccion(texto, subtexto=""):
    display(HTML(f"""
    <div style="border-left:4px solid #2E75B6;padding:8px 16px;margin:16px 0 8px 0;background:#EEF4FB;border-radius:0 6px 6px 0">
        <h3 style="color:#1F4E79;margin:0;font-size:15px;font-weight:500">{texto}</h3>
        {"<p style='color:#5B9BD5;margin:3px 0 0 0;font-size:12px'>" + subtexto + "</p>" if subtexto else ""}
    </div>
    """))

def nota(texto):
    display(HTML(f"""
    <div style="border-left:4px solid #F0A500;padding:8px 16px;margin:12px 0;background:#FFF8E7;border-radius:0 6px 6px 0">
        <p style="color:#7D5A00;margin:0;font-size:13px;line-height:1.6">📌 {texto}</p>
    </div>
    """))

def descripcion(texto):
    display(HTML(f"""
    <div style="border-left:4px solid #2E75B6;padding:10px 16px;margin:10px 0;background:#EEF4FB;border-radius:0 6px 6px 0">
        <p style="color:#1F4E79;margin:0;font-size:13px;line-height:1.7">{texto}</p>
    </div>
    """))


# ─────────────────────────────────────────────
# PORTADA E ÍNDICE
# ─────────────────────────────────────────────

def portada():
    display(HTML("""
    <div style="background:#1F4E79;padding:30px 24px;border-radius:12px;margin:10px 0 20px 0;border-left:8px solid #2E75B6">
        <h1 style="color:white;margin:0;font-size:26px;font-weight:600">
            Diseño y evaluación de un servicio inteligente de predicción de información de red para sistemas de Edge Computing
        </h1>
        <p style="color:#9DC3E6;margin:12px 0 0 0;font-size:14px">Trabajo de Fin de Grado - Ingeniería Informática · UCLM</p>
        <p style="color:#9DC3E6;margin:4px 0 0 0;font-size:13px">Tecnología Específica de Ingeniería de Computadores</p>
        <p style="color:#9DC3E6;margin:4px 0 0 0;font-size:13px">Autor: Luis Melero Jareño &nbsp;|&nbsp; Tutores: Gabriel Cebrián Márquez · Estefanía Coronado Calero</p>
        <p style="color:#9DC3E6;margin:4px 0 0 0;font-size:13px">Curso 2025-2026</p>
    </div>
    <div style="border-left:4px solid #2E75B6;padding:12px 20px;margin:10px 0;background:#EEF4FB;border-radius:0 8px 8px 0">
        <p style="color:#1F4E79;margin:0;font-size:14px;line-height:1.7">
            Este notebook recoge el pipeline completo de experimentación para el desarrollo del servicio inteligente
            de predicción de KPIs de red. Se incluyen la construcción del dataset homogéneo a partir de simulaciones
            con MintEDGE, la selección de variables, la división del dataset, el entrenamiento y evaluación de modelos
            tabulares y secuenciales, y la comparación final de resultados sobre el conjunto de validación.
        </p>
    </div>
    """))

def indice():
    display(HTML("""
    <div style="margin:20px 0">
        <h2 style="color:#1F4E79;font-size:18px;border-bottom:2px solid #2E75B6;padding-bottom:8px">Índice</h2>
        <div style="color:#2E75B6;font-size:14px;line-height:2.2">
            <div><b>1. Construcción del dataset homogéneo</b></div>
            <div><b>2. Carga y configuración</b></div>
            <div><b>3. Análisis exploratorio del dataset</b></div>
            <div><b>4. Selección de variables y división del dataset</b></div>
            <div style="padding-left:24px">4.1 División del dataset</div>
            <div style="padding-left:24px">4.2 Métodos de selección de variables</div>
            <div><b>5. Modelos tabulares</b></div>
            <div style="padding-left:24px">5.1 Configuración y entrenamiento</div>
            <div style="padding-left:24px">5.2 Resultados sobre validación</div>
            <div style="padding-left:24px">5.3 Ajuste de hiperparámetros</div>
            <div style="padding-left:24px">5.4 Score y selección del mejor modelo</div>
            <div style="padding-left:24px">5.5 Análisis del error</div>
            <div><b>6. Modelos secuenciales</b></div>
            <div style="padding-left:24px">6.1 Generación de secuencias</div>
            <div style="padding-left:24px">6.2 Análisis previo y selección de configuración</div>
            <div style="padding-left:24px">6.3 Entrenamiento completo</div>
            <div style="padding-left:24px">6.4 Score y selección del mejor modelo</div>
            <div style="padding-left:24px">6.5 Análisis del error</div>
            <div><b>7. Guardado de resultados y modelos</b></div>
            <div><b>8. Comparación global de modelos</b></div>
            <div style="padding-left:24px">8.1 Comparativa tabular vs secuencial</div>
            <div style="padding-left:24px">8.2 Ranking global y selección del ganador</div>
            <div style="padding-left:24px">8.3 Persistencia de modelos ganadores</div>
            <div><b>9. Evaluación final sobre el conjunto de test</b></div>
            <div style="padding-left:24px">9.1 Predicción sobre test</div>
            <div style="padding-left:24px">9.2 Comparativa validación vs test</div>
            <div style="padding-left:24px">9.3 Análisis de importancia de variables</div>
            <div style="padding-left:24px">9.4 Discusión de resultados</div>
        </div>
    </div>
    """))