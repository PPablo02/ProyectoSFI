import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Proyecto Final de Manejo de Portafolios", page_icon="📊", layout="wide")

# Título principal
st.title("📊 Proyecto Final de Manejo de Portafolios y Asset Allocation")

# Estilos personalizados para los textos
st.markdown("""
    <style>
    .big-font {
        font-size: 30px !important;
        color: #4CAF50;
        font-weight: bold;
    }
    .section-title {
        font-size: 24px;
        color: #2C3E50;
        font-weight: bold;
    }
    .subsection {
        font-size: 18px;
        color: #34495E;
        font-style: italic;
    }
    .tabs {
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
    }
    .tabs button {
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px 20px;
        margin: 5px;
        cursor: pointer;
        font-size: 18px;
    }
    .tabs button:hover {
        background-color: #e0e0e0;
    }
    .tabs button.active {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Función para renderizar la página de inicio
def pagina_inicio():
    st.header("Introducción")
    st.markdown("<p class='big-font'>Bienvenidos al proyecto final del curso de Manejo de Portafolios y Asset Allocation.</p>", unsafe_allow_html=True)
    st.write("""
        El objetivo es crear un portafolio óptimo usando diferentes modelos y técnicas, 
        incluyendo el modelo de Black-Litterman, y realizar un backtesting de los portafolios obtenidos.
    """)

    # Nombres de los colaboradores
    colaboradores = [
        "Pablo Pineda Pineda",
        "Mariana Vigil Villegas",
        "Adrián Soriano Fuentes",
        "Emmanuel Reyes Hernández"
    ]

    st.markdown("<p class='section-title'>Colaboradores:</p>", unsafe_allow_html=True)
    st.write(", ".join(colaboradores))

# Función para renderizar la página de selección de ETFs
def pagina_etfs():
    st.header("Selección de 5 ETFs")
    st.write("""
        En esta sección puedes seleccionar 5 ETFs que serán parte de tu análisis de portafolios.
    """)
    etfs = []
    for i in range(1, 6):
        etfs.append(st.text_input(f"ETF {i}"))
    st.markdown("<p class='subsection'>Los ETFs seleccionados:</p>", unsafe_allow_html=True)
    st.write(", ".join(etfs))

# Función para renderizar la página de estadísticas de los ETFs
def pagina_stats_etfs():
    st.header("Stats de los ETFs")
    st.write("""
        Aquí puedes ver las estadísticas relacionadas con los ETFs seleccionados.
    """)
    st.markdown("<p class='subsection'>Estadísticas como el rendimiento histórico, volatilidad, etc.</p>", unsafe_allow_html=True)

# Función para renderizar la página de portafolios óptimos y backtesting
def pagina_portafolios():
    st.header("Portafolios Óptimos y Backtesting")
    st.write("""
        En esta sección se realiza la optimización de portafolios y el backtesting para evaluar el rendimiento.
    """)
    st.markdown("<p class='subsection'>Análisis de portafolios óptimos y backtesting de estos.</p>", unsafe_allow_html=True)

# Función para renderizar la página del modelo de Black-Litterman
def pagina_black_litterman():
    st.header("Modelo de Black-Litterman")
    st.write("""
        En esta sección se implementa el modelo de Black-Litterman para obtener la asignación de activos óptima.
    """)
    st.markdown("<p class='subsection'>Implementación y análisis usando el modelo de Black-Litterman.</p>", unsafe_allow_html=True)

# Crear un conjunto de botones como pestañas
tabs = ["Inicio", "Selección de ETFs", "Stats de los ETFs", "Portafolios Óptimos y Backtesting", "Modelo de Black-Litterman"]

# Muestra botones para las pestañas
selected_tab = st.radio("", tabs, index=0, key="tabs", label_visibility="collapsed")

# Mostrar el contenido correspondiente a la pestaña seleccionada
if selected_tab == "Inicio":
    pagina_inicio()
elif selected_tab == "Selección de ETFs":
    pagina_etfs()
elif selected_tab == "Stats de los ETFs":
    pagina_stats_etfs()
elif selected_tab == "Portafolios Óptimos y Backtesting":
    pagina_portafolios()
elif selected_tab == "Modelo de Black-Litterman":
    pagina_black_litterman()
