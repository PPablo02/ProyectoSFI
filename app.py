import streamlit as st

# Título principal
st.title("Proyecto Final de Manejo de Portafolios y Asset Allocation")

# Barra de navegación con pestañas
tabs = st.sidebar.radio("Selecciona una pestaña", ["Introducción", "Selección de ETFs", "Stats de los ETFs", "Portafolios Óptimos y Backtesting", "Modelo de Black-Litterman"])

if tabs == "Introducción":
    # Pestaña de introducción
    st.header("Introducción")
    st.write("""
        Este es el proyecto final del curso de Manejo de Portafolios y Asset Allocation. 
        El objetivo es crear un portafolio óptimo usando diferentes modelos y técnicas, 
        incluyendo el modelo de Black-Litterman, y realizar un backtesting de los portafolios obtenidos.
    """)
    st.write("A continuación, puedes ingresar cuatro nombres:")
    
    # Lista para ingresar los nombres
    names = [st.text_input(f"Nombre {i+1}") for i in range(4)]
    st.write("Los nombres ingresados son:", names)

elif tabs == "Selección de ETFs":
    # Pestaña para seleccionar 5 ETFs
    st.header("Selección de 5 ETFs")
    st.write("""
        En esta sección puedes seleccionar 5 ETFs que serán parte de tu análisis de portafolios.
    """)
    etfs = []
    for i in range(1, 6):
        etfs.append(st.text_input(f"ETF {i}"))
    st.write("Los ETFs seleccionados son:", etfs)

elif tabs == "Stats de los ETFs":
    # Pestaña de estadísticas de los ETFs
    st.header("Stats de los ETFs")
    st.write("""
        Aquí puedes ver las estadísticas relacionadas con los ETFs seleccionados.
    """)
    # Estadísticas básicas (esto puede ser más detallado dependiendo de los datos disponibles)
    st.write("Aquí irían las estadísticas como el rendimiento histórico, volatilidad, etc.")

elif tabs == "Portafolios Óptimos y Backtesting":
    # Pestaña de portafolios óptimos y backtesting
    st.header("Portafolios Óptimos y Backtesting")
    st.write("""
        En esta sección se realiza la optimización de portafolios y el backtesting para evaluar el rendimiento.
    """)
    # Código para la optimización de portafolios y backtesting (aquí puedes integrar tus modelos de optimización)
    st.write("Aquí iría el análisis de portafolios óptimos y el backtesting de estos portafolios.")

elif tabs == "Modelo de Black-Litterman":
    # Pestaña del modelo de Black-Litterman
    st.header("Modelo de Black-Litterman")
    st.write("""
        En esta sección se implementa el modelo de Black-Litterman para obtener la asignación de activos óptima.
    """)
    # Aquí puedes agregar el código que implementa el modelo Black-Litterman
    st.write("Aquí iría la implementación del modelo de Black-Litterman.")

