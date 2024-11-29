# Título de la aplicación
st.title("Optimización de Portafolios: Selección de Activos")

# Selección de ETFs
etfs = st.multiselect(
    "Selecciona 5 ETFs:",
    options=["SPY", "QQQ", "IWM", "EEM", "TLT", "GLD"],  # Ejemplo de opciones
    default=["SPY", "QQQ", "IWM", "EEM", "TLT"]
)

# Obtener información detallada de cada ETF
if len(etfs) == 5:
    st.header("Descripción de los ETFs Seleccionados")

    # Crear un DataFrame vacío para almacenar la información
    etf_details = []

    # Descargar información para cada ETF seleccionado
    for etf in etfs:
        ticker = yf.Ticker(etf)
        info = ticker.info  # Información completa del ETF

        # Extraer datos relevantes
        etf_details.append({
            "Ticker": etf,
            "Nombre": info.get("shortName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Divisa": info.get("currency", "N/A"),
            "Activo Principal": info.get("quoteType", "N/A"),
            "Resumen": info.get("longBusinessSummary", "No disponible"),
            "Costo (%)": info.get("expenseRatio", "N/A") * 100 if info.get("expenseRatio") else "N/A",
        })

    # Convertir la información a un DataFrame
    etf_details_df = pd.DataFrame(etf_details)

    # Mostrar la tabla en Streamlit
    st.dataframe(etf_details_df)

    # Mostrar un resumen textual de cada ETF
    for _, row in etf_details_df.iterrows():
        st.subheader(f"{row['Ticker']} - {row['Nombre']}")
        st.write(f"**Resumen:** {row['Resumen']}")

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
