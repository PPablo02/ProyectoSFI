import streamlit as st
import yfinance as yf
import plotly.express as px

# Título de la aplicación
st.title("Optimización de Portafolios: Selección de Activos")

# Selección de ETFs
etfs = st.multiselect(
    "Selecciona 5 ETFs:",
    options=["SPY", "QQQ", "IWM", "EEM", "TLT", "GLD"],  # Ejemplo de opciones
    default=["SPY", "QQQ", "IWM", "EEM", "TLT"]
)

# Obtener información detallada y mostrar viñetas para cada ETF
if len(etfs) == 5:
    st.header("Descripción de los ETFs Seleccionados")

    # Mostrar información para cada ETF
    for etf in etfs:
        # Descargar información del ETF desde Yahoo Finance
        ticker = yf.Ticker(etf)
        info = ticker.info

        # Información básica
        nombre = info.get("shortName", "N/A")
        divisa = info.get("currency", "N/A")
        costo = info.get("expenseRatio", "N/A")
        costo = f"{costo * 100:.2f}%" if isinstance(costo, float) else "N/A"

        # Campo para agregar un resumen manual
        resumen = st.text_area(f"Resumen para {etf}:", placeholder="Escribe aquí el resumen del ETF.")

        # Mostrar información en formato de viñetas
        st.markdown(f"### {etf} - {nombre}")
        st.markdown(f"- **Divisa:** {divisa}")
        st.markdown(f"- **Costo Anual:** {costo}")
        st.markdown(f"- **Resumen:** {resumen if resumen else 'No se ha ingresado un resumen.'}")
        st.markdown("---")  # Línea separadora para mejor visualización

    # Descarga de datos históricos
    st.header("Series de Tiempo de los ETFs Seleccionados")
    data = {
        etf: yf.download(etf, start="2010-01-01", end="2023-12-31")["Adj Close"]
        for etf in etfs
    }

    # Graficar series de tiempo
    for etf, series in data.items():
        st.subheader(f"Serie de Tiempo: {etf}")
        fig = px.line(series, title=f"Serie de tiempo de {etf}", labels={"index": "Fecha", "value": "Precio Ajustado"})
        st.plotly_chart(fig)
