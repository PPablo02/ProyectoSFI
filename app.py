import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# Título de la aplicación
st.title("Optimización de Portafolios: Selección de Activos")

# Definir los ETFs seleccionados y sus categorías
etfs = {
    "Renta Fija Desarrollada": "TLT",  # Bonos del Tesoro EE.UU. a Largo Plazo
    "Renta Fija Emergente": "EMB",    # Bonos Mercados Emergentes
    "Renta Variable Desarrollada": "SPY",  # S&P 500
    "Renta Variable Emergente": "EEM",    # Acciones Mercados Emergentes
    "Materias Primas": "GLD"  # Oro
}

# Colores personalizados para cada ETF
etf_colors = {
    "TLT": "blue",
    "EMB": "green",
    "SPY": "red",
    "EEM": "orange",
    "GLD": "purple"
}

# Información básica y descripciones manuales de los ETFs
etf_descriptions = {
    "TLT": {
        "Tipo": "Renta Fija Desarrollada",
        "Exposición": "Bonos del Tesoro de EE.UU. a largo plazo.",
        "Índice": "ICE U.S. Treasury 20+ Year Bond Index",
        "Divisa": "USD",
        "Contribuidores Principales": "Bonos a largo plazo emitidos por el gobierno de EE.UU.",
        "Regiones": "Estados Unidos",
        "Métricas de Riesgo": """
            - **Duración:** Alta (>20 años), lo que implica una alta sensibilidad a cambios en las tasas de interés.
            - **Riesgo de Tasa de Interés:** Incrementos en las tasas pueden reducir significativamente el valor de este ETF.
            - **Volatilidad:** Menor que activos de renta variable, pero mayor que bonos de menor duración.
        """,
        "Estilo": "Renta fija, grado de inversión.",
        "Costo": "0.15%",
        "Link Riesgo": "https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf"
    },
    "EMB": {
        "Tipo": "Renta Fija Emergente",
        "Exposición": "Bonos soberanos de mercados emergentes denominados en dólares.",
        "Índice": "J.P. Morgan EMBI Global Core Index",
        "Divisa": "USD",
        "Contribuidores Principales": "Gobiernos de mercados emergentes como Brasil, México, Turquía.",
        "Regiones": "Mercados emergentes.",
        "Métricas de Riesgo": """
            - **Duración:** Moderada (~7 años).
            - **Riesgo de Crédito:** Depende de la calidad crediticia de países como Brasil, Turquía y México.
            - **Riesgo de Liquidez:** Menor liquidez comparada con bonos de mercados desarrollados.
            - **Riesgo Soberano:** Alta exposición a cambios en políticas fiscales y económicas.
        """,
        "Estilo": "Renta fija, grado de inversión y alto rendimiento.",
        "Costo": "0.39%",
        "Link Riesgo": "https://www.ishares.com/us/products/239572/ishares-jp-morgan-usd-emerging-markets-bond-etf"
    },
    "SPY": {
        "Tipo": "Renta Variable Desarrollada",
        "Exposición": "Acciones de empresas grandes de EE.UU. en el índice S&P 500.",
        "Índice": "S&P 500",
        "Divisa": "USD",
        "Contribuidores Principales": "Apple, Microsoft, Amazon.",
        "Regiones": "Estados Unidos.",
        "Métricas de Riesgo": """
            - **Beta:** ~1, refleja alta correlación con el mercado en general.
            - **Volatilidad:** Moderada (~16% anualizada en promedio histórico).
            - **Riesgo de Mercado:** Afectado por fluctuaciones económicas globales.
            - **Diversificación:** Bajo riesgo idiosincrático por exposición a múltiples sectores.
        """,
        "Estilo": "Blend, Large Cap.",
        "Costo": "0.09%",
        "Link Riesgo": "https://www.ssga.com/us/en/intermediary/etfs/funds/spdr-sp-500-etf-trust-spy"
    },
    "EEM": {
        "Tipo": "Renta Variable Emergente",
        "Exposición": "Acciones de mercados emergentes.",
        "Índice": "MSCI Emerging Markets Index",
        "Divisa": "USD",
        "Contribuidores Principales": "Tencent, Samsung, Alibaba.",
        "Regiones": "China, Corea del Sur, Brasil, India.",
        "Métricas de Riesgo": """
            - **Beta:** ~1.2, más volátil que el mercado general.
            - **Riesgo de Moneda:** Fluctuaciones en monedas emergentes frente al dólar.
            - **Riesgo Geopolítico:** Alta exposición a cambios políticos en mercados emergentes.
            - **Volatilidad:** Alta (~20% anualizada en promedio histórico).
        """,
        "Estilo": "Blend, Emerging Markets.",
        "Costo": "0.68%",
        "Link Riesgo": "https://www.ishares.com/us/products/239637/ishares-msci-emerging-markets-etf"
    },
    "GLD": {
        "Tipo": "Materias Primas",
        "Exposición": "Oro físico.",
        "Índice": "Precio spot del oro.",
        "Divisa": "USD",
        "Contribuidores Principales": "Lingotes de oro almacenados en bóvedas seguras.",
        "Regiones": "Global.",
        "Métricas de Riesgo": """
            - **Riesgo de Mercado:** Afectado por tasas de interés, inflación y fluctuaciones del dólar.
            - **Riesgo de Almacenamiento:** Relacionado con los costos y la seguridad del almacenamiento físico.
            - **Volatilidad:** Moderada (~15% anualizada), más baja que la de acciones emergentes.
        """,
        "Estilo": "Materias primas.",
        "Costo": "0.40%",
        "Link Riesgo": "https://www.spdrgoldshares.com/usa/"
    }
}

# Mostrar la información de cada ETF con su serie de tiempo
st.header("Descripción y Series de Tiempo de los ETFs Seleccionados")
for category, etf in etfs.items():
    details = etf_descriptions[etf]

    # Mostrar información en formato de viñetas
    st.markdown(f"<h3 style='color:{etf_colors[etf]}'>{etf} - {details['Tipo']}</h3>", unsafe_allow_html=True)
    st.markdown(f"- **Exposición:** {details['Exposición']}")
    st.markdown(f"- **Índice:** {details['Índice']}")
    st.markdown(f"- **Divisa:** {details['Divisa']}")
    st.markdown(f"- **Contribuidores Principales:** {details['Contribuidores Principales']}")
    st.markdown(f"- **Regiones:** {details['Regiones']}")
    st.markdown(f"- **Métricas de Riesgo:** {details['Métricas de Riesgo']}")
    st.markdown(f"- [Más detalles sobre métricas de riesgo para {etf}]({details['Link Riesgo']})")

    # Descargar y graficar la serie de tiempo
    series = yf.download(etf, start="2010-01-01", end="2023-12-31")["Adj Close"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", line=dict(color=etf_colors[etf]), name=etf))
    st.plotly_chart(fig)

    # Separador para mejorar visualización
    st.markdown("---")

# Mostrar gráfica combinada de todas las series de tiempo
st.header("Comparación de Series de Tiempo de Todos los ETFs")
combined_fig = go.Figure()
for category, etf in etfs.items():
    details = etf_descriptions[etf]
    series = yf.download(etf, start="2010-01-01", end="2023-12-31")["Adj Close"]
    combined_fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", line=dict(color=etf_colors[etf]), name=etf))

st.plotly_chart(combined_fig)
