import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

# --- Configuración de los ETFs ---
tickers = {
    "TLT": {
        "nombre": "iShares 20+ Year Treasury Bond ETF",
        "descripcion": "Este ETF sigue el índice ICE U.S. Treasury 20+ Year Bond Index, compuesto por bonos del gobierno de EE. UU. con vencimientos superiores a 20 años.",
        "sector": "Renta fija",
        "categoria": "Bonos del Tesoro de EE. UU.",
        "exposicion": "Bonos del gobierno de EE. UU. a largo plazo.",
        "moneda": "USD",
        "beta": 0.2,
        "top_holdings": [
            {"symbol": "US Treasury", "holdingPercent": "100%"}
        ],
        "gastos": "0.15%",
        "rango_1y": "120-155 USD",
        "rendimiento_ytd": "5%",
        "duracion": "Larga"
    },
    "EMB": {
        "nombre": "iShares JP Morgan USD Emerging Markets Bond ETF",
        "descripcion": "Este ETF sigue el índice J.P. Morgan EMBI Global Diversified Index, que rastrea bonos soberanos de mercados emergentes en dólares estadounidenses.",
        "sector": "Renta fija",
        "categoria": "Bonos emergentes",
        "exposicion": "Bonos soberanos de mercados emergentes denominados en USD.",
        "moneda": "USD",
        "beta": 0.6,
        "top_holdings": [
            {"symbol": "Brazil 10Yr Bond", "holdingPercent": "10%"},
            {"symbol": "Mexico 10Yr Bond", "holdingPercent": "9%"},
            {"symbol": "Russia 10Yr Bond", "holdingPercent": "7%"}
        ],
        "gastos": "0.39%",
        "rango_1y": "85-105 USD",
        "rendimiento_ytd": "8%",
        "duracion": "Media"
    },
    "SPY": {
        "nombre": "SPDR S&P 500 ETF Trust",
        "descripcion": "Este ETF sigue el índice S&P 500, compuesto por las 500 principales empresas de EE. UU.",
        "sector": "Renta variable",
        "categoria": "Acciones grandes de EE. UU.",
        "exposicion": "Acciones de las 500 empresas más grandes de EE. UU.",
        "moneda": "USD",
        "beta": 1.0,
        "top_holdings": [
            {"symbol": "Apple", "holdingPercent": "6.5%"},
            {"symbol": "Microsoft", "holdingPercent": "5.7%"},
            {"symbol": "Amazon", "holdingPercent": "4.3%"}
        ],
        "gastos": "0.0945%",
        "rango_1y": "360-420 USD",
        "rendimiento_ytd": "15%",
        "duracion": "Baja"
    },
    "VWO": {
        "nombre": "Vanguard FTSE Emerging Markets ETF",
        "descripcion": "Este ETF sigue el índice FTSE Emerging Markets All Cap China A Inclusion Index, que incluye acciones de mercados emergentes en Asia, Europa, América Latina y África.",
        "sector": "Renta variable",
        "categoria": "Acciones emergentes",
        "exposicion": "Mercados emergentes globales.",
        "moneda": "USD",
        "beta": 1.2,
        "top_holdings": [
            {"symbol": "Tencent", "holdingPercent": "6%"},
            {"symbol": "Alibaba", "holdingPercent": "4.5%"},
            {"symbol": "Taiwan Semiconductor", "holdingPercent": "4%"}
        ],
        "gastos": "0.08%",
        "rango_1y": "40-55 USD",
        "rendimiento_ytd": "10%",
        "duracion": "Alta"
    },
    "GLD": {
        "nombre": "SPDR Gold Shares",
        "descripcion": "Este ETF sigue el precio del oro físico.",
        "sector": "Materias primas",
        "categoria": "Oro físico",
        "exposicion": "Oro físico y contratos futuros de oro.",
        "moneda": "USD",
        "beta": 0.1,
        "top_holdings": [
            {"symbol": "Gold", "holdingPercent": "100%"}
        ],
        "gastos": "0.40%",
        "rango_1y": "160-200 USD",
        "rendimiento_ytd": "12%",
        "duracion": "Baja"
    }
}

# --- Funciones Auxiliares ---
def cargar_datos(tickers, inicio, fin):
    """Descarga datos históricos para una lista de tickers desde Yahoo Finance."""
    datos = {}
    for ticker in tickers:
        df = yf.download(ticker, start=inicio, end=fin)
        df['Retornos'] = df['Close'].pct_change()
        datos[ticker] = df
    return datos

def calcular_metricas(df, nivel_VaR=[0.95, 0.975, 0.99]):
    """Calcula métricas estadísticas clave, incluyendo VaR y beta."""
    retornos = df['Retornos'].dropna()

    # Calcular métricas básicas
    media = np.mean(retornos) * 100
    volatilidad = np.std(retornos) * 100
    sesgo = skew(retornos)
    curtosis = kurtosis(retornos)

    # VaR y CVaR
    VaR = {f"VaR {nivel*100}%": np.percentile(retornos, (1 - nivel) * 100) for nivel in nivel_VaR}
    cVaR = {f"CVaR {nivel*100}%": retornos[retornos <= np.percentile(retornos, (1 - nivel) * 100)].mean() for nivel in nivel_VaR}

    # Sharpe Ratio
    sharpe = np.mean(retornos) / np.std(retornos) if np.std(retornos) != 0 else np.nan

    # Beta
    sp500 = yf.download("^GSPC", start=df.index[0], end=df.index[-1])['Adj Close']
    sp500_retornos = sp500.pct_change().dropna()
    retornos_alineados = retornos.reindex(sp500_retornos.index).dropna()
    sp500_retornos_alineados = sp500_retornos.reindex(retornos_alineados.index).dropna()

    # Alinear ambas series utilizando el índice compartido
    indice_comun = retornos_alineados.index.intersection(sp500_retornos_alineados.index)
    retornos_alineados = retornos_alineados.loc[indice_comun]
    sp500_retornos_alineados = sp500_retornos_alineados.loc[indice_comun]
    
    # Convertir a arreglos bidimensionales explícitamente
    retornos_alineados = retornos_alineados.values.flatten()
    sp500_retornos_alineados = sp500_retornos_alineados.values.flatten()
    
    if len(retornos_alineados) > 0 and len(sp500_retornos_alineados) > 0:
        covarianza = np.cov(retornos_alineados, sp500_retornos_alineados)[0, 1]
        var_sp500 = np.var(sp500_retornos_alineados)
        beta = covarianza / var_sp500 if var_sp500 != 0 else np.nan
    else:
        beta = np.nan
        metrics = {
            "Media (%)": media,
            "Volatilidad (%)": volatilidad,
            "Sesgo": sesgo,
            "Curtosis": curtosis,
            **VaR,
            **cVaR,
            "Sharpe Ratio": sharpe,
            "Beta": beta
        }
    
        return pd.DataFrame(metrics, index=["Valor"]).T

# Función para graficar con formato adecuado
def graficar_linea(df, x_column, y_column, title, labels=None):
    """
    Función para crear un gráfico de línea asegurando que los datos de `y_column` sean unidimensionales.

    :param df: DataFrame con los datos.
    :param x_column: Nombre de la columna a usar en el eje X.
    :param y_column: Nombre de la columna a usar en el eje Y.
    :param title: Título del gráfico.
    :param labels: Diccionario para etiquetar los ejes (opcional).
    :return: gráfico interactivo.
    """
    # Asegurarse de que y_column sea un DataFrame de una sola columna y alineado con el índice de x_column
    if isinstance(y_column, pd.Series):
        y_column = y_column.reset_index(drop=True)  # Asegurar que y_column tenga el índice correcto

    # Crear el gráfico de línea
    fig = px.line(df, x=x_column, y=y_column, title=title, labels=labels)
    return fig

# --- Funciones de Optimización de Portafolios ---
def optimizar_portafolio_markowitz(retornos, metodo="min_vol", objetivo=None):
    # Función para optimizar el portafolio según el modelo de Markowitz
    media = retornos.mean()
    cov = retornos.cov()

    # Función para calcular el riesgo del portafolio (volatilidad)
    def riesgo(w):
        return np.sqrt(np.dot(w.T, np.dot(cov, w)))

    # Función para calcular el Sharpe ratio del portafolio
    def sharpe(w):
        return -(np.dot(w.T, media) / np.sqrt(np.dot(w.T, np.dot(cov, w))))

    # Número de activos
    n = len(media)
    
    # Pesos iniciales (distribución igual)
    w_inicial = np.ones(n) / n
    
    # Restricciones: los pesos deben sumar 1
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    
    if metodo == "target" and objetivo is not None:
        restricciones.append({"type": "eq", "fun": lambda w: np.dot(w, media) - objetivo})  # Rendimiento objetivo
        objetivo_funcion = riesgo
    elif metodo == "sharpe":
        objetivo_funcion = sharpe
    else:
        objetivo_funcion = riesgo
    
    # Definir los límites de los pesos (entre 0 y 1)
    limites = [(0, 1) for _ in range(n)]
    
    # Optimización
    resultado = minimize(objetivo_funcion, w_inicial, constraints=restricciones, bounds=limites)
    
    # Devolver los pesos optimizados
    return np.array(resultado.x).flatten()

def black_litterman_optimizar(retornos, P, Q, tau=0.05, metodo="min_vol"):
    """
    Optimiza el portafolio utilizando el modelo de Black-Litterman.
    
    Parameters:
    - retornos: DataFrame con los retornos de cada activo.
    - P: Matriz de views (tamaño k x n).
    - Q: Vector con las expectativas de rendimiento (tamaño k x 1).
    - tau: Parámetro de incertidumbre sobre la media de los activos.
    
    Returns:
    - Pesos del portafolio ajustados por el modelo Black-Litterman.
    """
    media = retornos.mean()
    cov = retornos.cov()
    n = len(media)
    
    # Matriz de incertidumbre sobre las views
    M = np.linalg.inv(np.linalg.inv(tau * cov) + np.dot(np.dot(P.T, np.linalg.inv(np.diag([1]*P.shape[0]))), P))
    
    # Ajustar la media con el modelo Black-Litterman
    ajustada_media = np.dot(M, np.dot(np.linalg.inv(tau * cov), media) + np.dot(np.dot(P.T, np.linalg.inv(np.diag([1]*P.shape[0]))), Q))
    
    # Optimización de Markowitz usando la media ajustada
    return optimizar_portafolio_markowitz(retornos, metodo=metodo)

# --- Configuración de Streamlit ---
st.title("Proyecto de Optimización de Portafolios")

# Crear tabs
tabs = st.tabs(["Introducción", "Selección de ETF's", "Estadísticas de los ETF's", "Portafolios Óptimos", "Backtesting", "Modelo de Black-Litterman"])

# --- Introducción ---
with tabs[0]:
    st.header("Introducción")
    st.write(""" 
    Este proyecto tiene como objetivo analizar y optimizar un portafolio utilizando ETFs en diferentes clases de activos, tales como renta fija, renta variable, y materias primas. A lo largo del proyecto, se evaluará el rendimiento de estos activos a través de diversas métricas financieras y técnicas de optimización de portafolios, como la optimización de mínima volatilidad y la maximización del Sharpe Ratio. 
    """)

# --- Selección de ETF's ---
with tabs[1]:
    st.header("Selección de ETF's")
    hoy = datetime.today().strftime("%Y-%m-%d")
    datos_2010_hoy = cargar_datos(list(tickers.keys()), "2010-01-01", hoy)

    # Crear un DataFrame consolidado con las características de los ETFs
    etf_caracteristicas = pd.DataFrame({
        "Ticker": list(tickers.keys()),
        "Nombre": [info["nombre"] for info in tickers.values()],
        "Sector": [info["sector"] for info in tickers.values()],
        "Categoría": [info["categoria"] for info in tickers.values()],
        "Exposición": [info["exposicion"] for info in tickers.values()],
        "Moneda": [info["moneda"] for info in tickers.values()],
        "Beta": [info["beta"] for info in tickers.values()],
        "Gastos": [info["gastos"] for info in tickers.values()],
        "Rango 1 Año": [info["rango_1y"] for info in tickers.values()],
        "Rendimiento YTD": [info["rendimiento_ytd"] for info in tickers.values()],
        "Duración": [info["duracion"] for info in tickers.values()],
    })

    # Mostrar las características en Streamlit
    st.subheader("Características de los ETFs Seleccionados")
    st.dataframe(etf_caracteristicas)

    # Mostrar la serie de tiempo de cada ETF
    st.subheader("Series de Tiempo de los Precios de Cierre")
    for ticker, info in tickers.items():
        fig = px.line(datos_2010_hoy[ticker],
                      x=datos_2010_hoy[ticker].index,
                      y=datos_2010_hoy[ticker]['Close'].values.flatten(),
                      title=f"Precio de Cierre - {ticker}")
        st.plotly_chart(fig)

# --- Estadísticas de los ETF's ---
with tabs[2]:
    st.header("Estadísticas de los ETF's (2010-2023)")

    # Cargar los datos históricos de 2010 a 2023
    datos_2010_2023 = cargar_datos(list(tickers.keys()), "2010-01-01", "2023-01-01")

    # Función auxiliar para calcular Drawdown y Watermark
    def calcular_drawdown_y_watermark(precios):
        """Calcula el drawdown y el watermark basado en precios."""
        watermark = precios.cummax()  # Máximo acumulado
        drawdown = (precios / watermark) - 1  # Pérdida relativa desde el máximo
        return drawdown, watermark

    # Loop para procesar cada ETF
    for ticker, descripcion in tickers.items():
        st.subheader(f"{descripcion['nombre']} ({ticker})")

        # Datos de rendimientos diarios
        data = datos_2010_2023[ticker].dropna()
        precios = data["Close"]  # Precios del ETF
        retornos = data["Retornos"]

        # Calcular métricas estadísticas
        media = retornos.mean() * 100
        volatilidad = retornos.std() * 100
        sesgo = skew(retornos)
        curtosis = kurtosis(retornos)
        sharpe = media / volatilidad if volatilidad != 0 else np.nan
        sortino = media / retornos[retornos < 0].std() if retornos[retornos < 0].std() != 0 else np.nan
        VaR_95 = np.percentile(retornos, 5)
        CVaR_95 = retornos[retornos <= VaR_95].mean()

        # Calcular Drawdown y Watermark
        drawdown, watermark = calcular_drawdown_y_watermark(precios)

        # 1. Mostrar métricas en tabla
        st.write("### Tabla de Métricas")
        metricas = pd.DataFrame({
            "Métrica": ["Media (%)", "Volatilidad (%)", "Sesgo", "Curtosis", "Sharpe Ratio", "Sortino Ratio", "VaR 95%", "CVaR 95%"],
            "Valor": [media, volatilidad, sesgo, curtosis, sharpe, sortino, VaR_95, CVaR_95],
        })
        st.dataframe(metricas)

        # 2. Gráfica de rendimientos acumulados
        st.write("### Rendimientos Acumulados")
        fig_rendimientos = graficar_linea(
            data, 
            x_column=data.index, 
            y_column=(1 + retornos).cumprod(),  # Rendimientos acumulados
            title=f"Rendimientos Acumulados - {descripcion['nombre']}",
            labels={"x": "Fecha", "y": "Rendimientos Acumulados"}
        )
        st.plotly_chart(fig_rendimientos)

        # 3. Gráfica de distribución de retornos con VaR y CVaR
        st.write("### Distribución de Retornos")
        fig_dist = px.histogram(
            retornos,
            nbins=50,
            title="Distribución de Retornos",
            labels={"value": "Retornos", "index": "Frecuencia"}
        )
        # Añadir líneas para VaR y CVaR
        fig_dist.add_vline(x=VaR_95, line_dash="dash", line_color="red", annotation_text="VaR 95%", annotation_position="top left")
        fig_dist.add_vline(x=CVaR_95, line_dash="dot", line_color="orange", annotation_text="CVaR 95%", annotation_position="top left")
        st.plotly_chart(fig_dist)

        # 4. Serie de tiempo del precio con drawdowns y watermark
        st.write("### Serie de Tiempo del Precio con Drawdowns y Watermark")
        # Convertir 'precios' a unidimensional
    precios_unidimensional = precios.values.flatten()
    
    # Asegurar que drawdown sea un arreglo NumPy unidimensional
    drawdown = drawdown.values.flatten()
    
    # Verificar dimensiones
    if len(precios_unidimensional) != len(drawdown):
        st.error("Las dimensiones de precios_unidimensional y drawdown no coinciden.")
        drawdown = np.resize(drawdown, precios_unidimensional.shape)
    
    # Calcular la curva de drawdown
    drawdown_curve = precios_unidimensional + (drawdown * precios_unidimensional)
    
    # Crear el gráfico
    fig_drawdown = px.line(
        x=data.index,
        y=precios_unidimensional,  # Ahora es unidimensional
        title=f"Precio del ETF - {descripcion['nombre']}",
        labels={"x": "Fecha", "y": "Precio del ETF"}
    )
    
    # Añadir Watermark y Drawdowns como capas
    fig_drawdown.add_scatter(x=data.index, y=watermark, mode="lines", name="Watermark", line=dict(color="blue", dash="dash"))
    fig_drawdown.add_scatter(x=data.index, y=drawdown_curve, mode="lines", name="Drawdown", line=dict(color="red", dash="dot"))
    
    # Mostrar el gráfico
    st.plotly_chart(fig_drawdown)



# --- Portafolios Óptimos ---
with tabs[3]:
    st.header("Portafolios Óptimos (2010-2020)")

    # Descargar datos históricos para el periodo 2010-2020
    datos_2010_2020 = cargar_datos(list(tickers.keys()), "2010-01-01", "2020-01-01")
    retornos_2010_2020 = pd.DataFrame({k: v["Retornos"] for k, v in datos_2010_2020.items()}).dropna()

    # 1. Portafolio de Mínima Volatilidad
    st.subheader("Portafolio de Mínima Volatilidad")
    pesos_min_vol = optimizar_portafolio_markowitz(retornos_2010_2020, metodo="min_vol")
    st.write("Pesos del Portafolio de Mínima Volatilidad:")
    for ticker, peso in zip(tickers.keys(), pesos_min_vol):
        st.write(f"{ticker}: {peso:.2%}")
    fig_min_vol = px.bar(x=list(tickers.keys()), y=pesos_min_vol, title="Pesos - Mínima Volatilidad")
    st.plotly_chart(fig_min_vol)

    # 2. Portafolio de Máximo Sharpe Ratio
    st.subheader("Portafolio de Máximo Sharpe Ratio")
    pesos_sharpe = optimizar_portafolio_markowitz(retornos_2010_2020, metodo="sharpe")
    st.write("Pesos del Portafolio de Máximo Sharpe Ratio:")
    for ticker, peso in zip(tickers.keys(), pesos_sharpe):
        st.write(f"{ticker}: {peso:.2%}")
    fig_sharpe = px.bar(x=list(tickers.keys()), y=pesos_sharpe, title="Pesos - Máximo Sharpe Ratio")
    st.plotly_chart(fig_sharpe)



# --- Backtesting ---
with tabs[4]:
    st.header("Backtesting (2021-2023)")

    # Descargar datos históricos para el periodo 2021-2023
    datos_2021_2023 = cargar_datos(list(tickers.keys()), "2021-01-01", "2023-01-01")
    retornos_2021_2023 = pd.DataFrame({k: v["Retornos"] for k, v in datos_2021_2023.items()}).dropna()


    # Crear DataFrame para guardar los rendimientos acumulados de cada portafolio
    rendimientos_acumulados = pd.DataFrame(index=retornos_2021_2023.index)

    
    #Métricas por cada portafolio, hay que incluir rendimientos acumulados, sesgo, curtosis, VaR, CVAR, sharp, sortino y drowdown
    #Comenzamos con el portafolio de mínima volatilidad
    #Extraemos los rendimientos
    #   Lista de portafolios y sus respectivos pesos
    # Lista de portafolios y sus respectivos pesos
    portafolios = [
        ("Mínima Volatilidad", pesos_min_vol),
        ("Máximo Sharpe Ratio", pesos_sharpe)
    ]

    # Lista para almacenar las métricas de cada portafolio
    metricas_totales = []

    # Bucle para iterar sobre los portafolios
    for nombre, pesos in portafolios:
        # Calculamos los rendimientos para el portafolio actual
        retornos = np.sum(retornos_2021_2023 * pesos, axis=1)

        # Calculamos las estadísticas relevantes
        media = retornos.mean() * 100
        volatilidad = retornos.std() * 100
        sesgo = skew(retornos)
        curtosis = kurtosis(retornos)
        sharpe = media / volatilidad if volatilidad != 0 else np.nan
        sortino = media / retornos[retornos < 0].std() if retornos[retornos < 0].std() != 0 else np.nan
        VaR_95 = np.percentile(retornos, 5)
        CVaR_95 = retornos[retornos <= VaR_95].mean()

        # Guardamos las métricas en un DataFrame
        metricas = pd.DataFrame({
            "Métrica": ["Media (%)", "Volatilidad (%)", "Sesgo", "Curtosis", "Sharpe Ratio", "Sortino Ratio", "VaR 95%", "CVaR 95%"],
            "Valor": [media, volatilidad, sesgo, curtosis, sharpe, sortino, VaR_95, CVaR_95],
            "Portafolio": [nombre] * 8  # Añadimos una columna con el nombre del portafolio
        })

        # Añadimos las métricas del portafolio actual a la lista de métricas
        metricas_totales.append(metricas)

    # Combinamos todas las métricas de los portafolios en un solo DataFrame
    metricas_finales = pd.concat(metricas_totales, ignore_index=True)

    # Mostrar las métricas combinadas
    st.write(metricas_finales)







    # Calcular rendimientos acumulados para cada portafolio
    st.subheader("Rendimientos Acumulados de los Portafolios")
    for nombre, pesos in [
        ("Mínima Volatilidad", pesos_min_vol),
        ("Máximo Sharpe Ratio", pesos_sharpe),
    ]:
        pesos_reshaped = np.array(pesos).reshape(-1, 1)
        rendimientos = retornos_2021_2023.dot(pesos_reshaped)
        rendimientos_acumulados[nombre] = rendimientos.cumsum()
        st.write(f"Rendimientos Acumulados - {nombre}")
        st.line_chart(rendimientos.cumsum())










        # Graficar todos los portafolios en una sola gráfica
    fig_rendimientos = px.line(
        rendimientos_acumulados,
        title="Rendimientos Acumulados - Comparación de Portafolios",
        labels={"value": "Rendimientos Acumulados", "index": "Fecha"}
    )
    st.plotly_chart(fig_rendimientos)


# --- Modelo de Black-Litterman ---
with tabs[5]:
    st.header("Modelo de Optimización Black-Litterman")
    P = np.array([
        [1, 0, 0, 0, 0],  # TLT tiene un rendimiento esperado de 3%
        [0, 1, 0, 0, 0],  # EMB tiene un rendimiento esperado de 6%
        [0, 0, 1, 0, 0],  # SPY tiene un rendimiento esperado de 8%
        [0, 0, 0, 1, 0],  # VWO tiene un rendimiento esperado de 11%
        [0, 0, 0, 0, 1]   # GLD tiene un rendimiento esperado de 4%
    ])
    Q = np.array([0.03, 0.06, 0.08, 0.11, 0.04])
    pesos_black_litterman = black_litterman_optimizar(retornos_2010_2020, P, Q)
    st.write("Pesos del Portafolio Ajustado con el Modelo de Black-Litterman:")
    for ticker, peso in zip(tickers.keys(), pesos_black_litterman):
        st.write(f"{ticker}: {peso:.2%}")
    fig_black_litterman = px.bar(x=list(tickers.keys()), y=pesos_black_litterman, title="Pesos Ajustados - Black-Litterman")
    st.plotly_chart(fig_black_litterman)
