import streamlit as st
import yfinance as yf
import plotly.express as px

# Título de la aplicación
st.title("Optimización de Portafolios: Selección de Activos")

# Definir los ETFs seleccionados y sus categorías
etfs = {
    "Renta Fija Desarrollada": "TLT",  # Ejemplo: Bonos del Tesoro EE.UU. a Largo Plazo
    "Renta Fija Emergente": "EMB",    # Ejemplo: Bonos Mercados Emergentes
    "Renta Variable Desarrollada": "SPY",  # Ejemplo: S&P 500
    "Renta Variable Emergente": "EEM",    # Ejemplo: Acciones Mercados Emergentes
    "Materias Primas": "GLD"  # Ejemplo: Oro
}

# Información básica y descripciones manuales de los ETFs
etf_descriptions = {
    "TLT": {
        "Exposición": "Bonos del Tesoro de EE.UU. a largo plazo.",
        "Índice": "ICE U.S. Treasury 20+ Year Bond Index",
        "Divisa": "USD",
        "Contribuidores Principales": "Bonos a largo plazo emitidos por el gobierno de EE.UU.",
        "Regiones": "Estados Unidos",
        "Métricas de Riesgo": "Alta duración (>20 años), sensibilidad a tasas de interés.",
        "Estilo": "Renta fija, grado de inversión.",
        "Costo": "0.15%"
    },
    "EMB": {
        "Exposición": "Bonos soberanos de mercados emergentes denominados en dólares.",
        "Índice": "J.P. Morgan EMBI Global Core Index",
        "Divisa": "USD",
        "Contribuidores Principales": "Gobiernos de mercados emergentes como Brasil, México, Turquía.",
        "Regiones": "Mercados emergentes.",
        "Métricas de Riesgo": "Duración moderada, beta con riesgo soberano.",
        "Estilo": "Renta fija, grado de inversión y alto rendimiento.",
        "Costo": "0.39%"
    },
    "SPY": {
        "Exposición": "Acciones de empresas grandes de EE.UU. en el índice S&P 500.",
        "Índice": "S&P 500",
        "Divisa": "USD",
        "Contribuidores Principales": "Apple, Microsoft, Amazon.",
        "Regiones": "Estados Unidos.",
        "Métricas de Riesgo": "Beta ~1 con el mercado, volatilidad moderada.",
        "Estilo": "Blend, Large Cap.",
        "Costo": "0.09%"
    },
    "EEM": {
        "Exposición": "Acciones de mercados emergentes.",
        "Índice": "MSCI Emerging Markets Index",
        "Divisa": "USD",
        "Contribuidores Principales": "Tencent, Samsung, Alibaba.",
        "Regiones": "China, Corea del Sur, Brasil, India.",
        "Métricas de Riesgo": "Beta alto (~1.2), alta volatilidad.",
        "Estilo": "Blend, Emerging Markets.",
        "Costo": "0.68%"
    },
    "GLD": {
        "Exposición": "Oro físico.",
        "Índice": "Precio spot del oro.",
        "Divisa": "USD",
        "Contribuidores Principales": "Lingotes de oro almacenados en bóvedas seguras.",
        "Regiones": "Global.",
        "Métricas de Riesgo": "Bajo beta con acciones (~0), alta correlación con inflación.",
        "Estilo": "Materias primas.",
        "Costo": "0.40%"
    }
}

# Mostrar la información de cada ETF con su serie de tiempo
st.header("Descripción y Series de Tiempo de los ETFs Seleccionados")
for category, etf in etfs.items():
    details = etf_descriptions[etf]

    # Mostrar información en formato de viñetas
    st.markdown(f"### {etf} - {details['Exposición']}")
    st.markdown(f"- **Índice:** {details['Índice']}")
    st.markdown(f"- **Divisa:** {details['Divisa']}")
    st.markdown(f"- **Contribuidores Principales:** {details['Contribuidores Principales']}")
    st.markdown(f"- **Regiones:** {details['Regiones']}")
    st.markdown(f"- **Métricas de Riesgo:** {details['Métricas de Riesgo']}")
    st.markdown(f"- **Estilo:** {details['Estilo']}")
    st.markdown(f"- **Costo:** {details['Costo']}")

    # Descargar y graficar la serie de tiempo
    series = yf.download(etf, start="2010-01-01", end="2023-12-31")["Adj Close"]
    st.subheader(f"Serie de Tiempo: {etf}")
    fig = px.line(series, title=f"Serie de tiempo de {etf}", labels={"index": "Fecha", "value": "Precio Ajustado"})
    st.plotly_chart(fig)

    # Separador para mejorar visualización
    st.markdown("---")
