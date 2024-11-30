import streamlit as st

# Título principal con estilo
st.set_page_config(page_title="Proyecto Final de Manejo de Portafolios", page_icon="📊", layout="wide")
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
    </style>
""", unsafe_allow_html=True)

# Página de inicio
def pagina_inicio():
    st.header("Introducción", anchor="introduccion")
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

# Página de selección de ETFs
def pagina_etfs():
    st.header("Selección de 5 ETFs", anchor="etfs")
    st.write("""
        En esta sección puedes seleccionar 5 ETFs que serán parte de tu análisis de portafolios.
    """)
    etfs = []
    for i in range(1, 6):
        etfs.append(st.text_input(f"ETF {i}"))
    st.markdown("<p class='subsection'>Los ETFs seleccionados:</p>", unsafe_allow_html=True)
    st.write(", ".join(etfs))

# Página de estadísticas de los ETFs
def pagina_stats_etfs():
    st.header("Stats de los ETFs", anchor="stats_etfs")
    st.write("""
        Aquí puedes ver las estadísticas relacionadas con los ETFs seleccionados.
    """)
    st.markdown("<p class='subsection'>Estadísticas como el rendimiento histórico, volatilidad, etc.</p>", unsafe_allow_html=True)

# Página de portafolios óptimos y backtesting
def pagina_portafolios():
    st.header("Portafolios Óptimos y Backtesting", anchor="portafolios")
    st.write("""
        En esta sección se realiza la optimización de portafolios y el backtesting para evaluar el rendimiento.
    """)
    st.markdown("<p class='subsection'>Análisis de portafolios óptimos y backtesting de estos.</p>", unsafe_allow_html=True)

# Página del modelo de Black-Litterman
def pagina_black_litterman():
    st.header("Modelo de Black-Litterman", anchor="black_litterman")
    st.write("""
        En esta sección se implementa el modelo de Black-Litterman para obtener la asignación de activos óptima.
    """)
    st.markdown("<p class='subsection'>Implementación y análisis usando el modelo de Black-Litterman.</p>", unsafe_allow_html=True)

# Barra lateral de navegación
pagina = st.sidebar.radio("Selecciona una página", ["Inicio", "Selección de ETFs", "Stats de los ETFs", "Portafolios Óptimos y Backtesting", "Modelo de Black-Litterman"])

# Redirigir a la página correspondiente
if pagina == "Inicio":
    pagina_inicio()
elif pagina == "Selección de ETFs":
    pagina_etfs()
elif pagina == "Stats de los ETFs":
    pagina_stats_etfs()
elif pagina == "Portafolios Óptimos y Backtesting":
    pagina_portafolios()
elif pagina == "Modelo de Black-Litterman":
    pagina_black_litterman()
