import streamlit as st

# Título principal
st.title("Proyecto Final de Manejo de Portafolios y Asset Allocation")

# Página de inicio
def pagina_inicio():
    st.header("Introducción")
    st.write("""
        Este es el proyecto final del curso de Manejo de Portafolios y Asset Allocation. 
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
    
    st.write("Los nombres de los colaboradores son:")
    st.write(", ".join(colaboradores))

# Página de selección de ETFs
def pagina_etfs():
    st.header("Selección de 5 ETFs")
    st.write("""
        En esta sección puedes seleccionar 5 ETFs que serán parte de tu análisis de portafolios.
    """)
    etfs = []
    for i in range(1, 6):
        etfs.append(st.text_input(f"ETF {i}"))
    st.write("Los ETFs seleccionados son:", etfs)

# Página de estadísticas de los ETFs
def pagina_stats_etfs():
    st.header("Stats de los ETFs")
    st.write("""
        Aquí puedes ver las estadísticas relacionadas con los ETFs seleccionados.
    """)
    st.write("Aquí irían las estadísticas como el rendimiento histórico, volatilidad, etc.")

# Página de portafolios óptimos y backtesting
def pagina_portafolios():
    st.header("Portafolios Óptimos y Backtesting")
    st.write("""
        En esta sección se realiza la optimización de portafolios y el backtesting para evaluar el rendimiento.
    """)
    st.write("Aquí iría el análisis de portafolios óptimos y el backtesting de estos portafolios.")

# Página del modelo de Black-Litterman
def pagina_black_litterman():
    st.header("Modelo de Black-Litterman")
    st.write("""
        En esta sección se implementa el modelo de Black-Litterman para obtener la asignación de activos óptima.
    """)
    st.write("Aquí iría la implementación del modelo de Black-Litterman.")

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
