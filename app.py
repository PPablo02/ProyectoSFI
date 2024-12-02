# --- Librerías ---
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from datetime import datetime

# --- Funciones Auxiliares ---
@st.cache_data
def cargar_datos(tickers, inicio, fin):
    """Descarga datos históricos para una lista de tickers desde Yahoo Finance."""
    datos = {}
    for ticker in tickers:
        df = yf.download(ticker, start=inicio, end=fin)
        df['Retornos'] = df['Close'].pct_change()
        datos[ticker] = df
    return datos

# --- Al llamar la función, convertir dict_keys a lista ---
tickers = list(tickers.keys())  # Convertir a lista
datos_2010_2023 = cargar_datos(tickers, "2010-01-01", "2023-01-01")
def obtener_informacion_etf(ticker):
    """Obtiene información detallada del ETF desde Yahoo Finance."""
    etf = yf.Ticker(ticker)
    info = etf.info
    return {
        "nombre": info.get('shortName', 'No disponible'),
        "descripcion": info.get('longBusinessSummary', 'No disponible'),
        "sector": info.get('sector', 'No disponible'),
        "categoria": info.get('category', 'No disponible'),
        "exposicion_geografica": info.get('region', 'No disponible'),
        "composicion": info.get('topHoldings', 'No disponible'),
        "gastos": info.get('expenseRatio', 'No disponible'),
        "rango_1y": info.get('fiftyTwoWeekRange', 'No disponible'),
        "rendimiento_ytd": info.get('ytdReturn', 'No disponible'),
        "moneda": info.get('currency', 'No disponible'),
        "beta": info.get('beta', 'No disponible')
    }

def calcular_metricas(df, nivel_VaR=[0.95, 0.975, 0.99]):
    """Calcula métricas estadísticas clave, incluyendo VaR a diferentes niveles."""
    retornos = df['Retornos'].dropna()
    metrics = {
        "Media (%)": np.mean(retornos) * 100,
        "Volatilidad (%)": np.std(retornos) * 100,
        "Sesgo": skew(retornos),
        "Curtosis": kurtosis(retornos),
        "Beta": df['Retornos'].cov(df['Retornos']) / df['Retornos'].var(),  # Beta simple
        "VaR 95%": np.percentile(retornos, 5),
        "VaR 97.5%": np.percentile(retornos, 2.5),
        "VaR 99%": np.percentile(retornos, 1),
        "CVaR 95%": retornos[retornos <= np.percentile(retornos, 5)].mean(),
        "Sharpe Ratio": np.mean(retornos) / np.std(retornos)
    }
    return pd.DataFrame(metrics, index=["Valor"]).T

def optimizar_portafolio(retornos, metodo="min_vol", objetivo=None):
    """Optimiza el portafolio basado en mínima volatilidad, Sharpe Ratio o un rendimiento objetivo."""
    media = retornos.mean()
    cov = retornos.cov()

    def riesgo(w):
        return np.sqrt(w.T @ cov @ w)

    def sharpe(w):
        return -(w.T @ media) / np.sqrt(w.T @ cov @ w)

    n = len(media)
    w_inicial = np.ones(n) / n
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if metodo == "target":
        restricciones.append({"type": "eq", "fun": lambda w: w.T @ media - objetivo})
        objetivo_funcion = riesgo
    elif metodo == "sharpe":
        objetivo_funcion = sharpe
    else:
        objetivo_funcion = riesgo

    limites = [(0, 1) for _ in range(n)]
    resultado = minimize(objetivo_funcion, w_inicial, constraints=restricciones, bounds=limites)
    return resultado.x

def calcular_drawdown(df):
    """Calcula el drawdown y watermark de una serie de precios."""
    roll_max = df['Close'].cummax()
    daily_drawdown = df['Close'] / roll_max - 1.0
    watermark = df['Close'] / roll_max
    return daily_drawdown, watermark

# --- Configuración de Streamlit ---
st.title("Proyecto de Optimización de Portafolios")

# Crear tabs
tabs = st.tabs(["Introducción", "Selección de ETF's", "Estadísticas de los ETF's", "Portafolios Óptimos", "Backtesting"])

# --- Introducción ---
with tabs[0]:
    st.header("Introducción")
    st.write("""
    Este proyecto tiene como objetivo analizar y optimizar un portafolio utilizando ETFs en diferentes clases de activos, tales como renta fija, renta variable, y materias primas. A lo largo del proyecto, se evaluará el rendimiento de estos activos a través de diversas métricas financieras y técnicas de optimización de portafolios, como la optimización de mínima volatilidad y la maximización del Sharpe Ratio.

    Para lograr esto, se utilizarán datos históricos de rendimientos y se realizarán pruebas de backtesting para validar las estrategias propuestas. Además, se implementará el modelo de optimización Black-Litterman para ajustar los rendimientos esperados en función de perspectivas macroeconómicas.
    """)

# --- Selección de ETF's ---
with tabs[1]:
    st.header("Selección de ETF's")
    
    tickers = {
        "TLT": "Bonos del Tesoro a Largo Plazo (Renta Fija Desarrollada)",
        "EMB": "Bonos Mercados Emergentes (Renta Fija Emergente)",
        "SPY": "S&P 500 (Renta Variable Desarrollada)",
        "EEM": "MSCI Mercados Emergentes (Renta Variable Emergente)",
        "GLD": "Oro Físico (Materias Primas)"
    }
    datos_2010_2023 = cargar_datos(tickers.keys(), "2010-01-01", "2023-01-01")

    for ticker, descripcion in tickers.items():
        st.subheader(f"{ticker} - {descripcion}")
        
        # Obtener información detallada del ETF
        info_etf = obtener_informacion_etf(ticker)
        
        # Mostrar detalles sobre la composición del ETF
        st.write("### Descripción del ETF:")
        st.write(info_etf['descripcion'])
        
        st.write("### Sector de Inversión:")
        st.write(info_etf['sector'])
        
        st.write("### Categoría del ETF:")
        st.write(info_etf['categoria'])
        
        st.write("### Exposición Geográfica:")
        st.write(info_etf['exposicion_geografica'])
        
        st.write("### Composición Top Holdings:")
        st.write(info_etf['composicion'])
        
        st.write("### Relación de Gastos (Expense Ratio):")
        st.write(f"{info_etf['gastos']} %")
        
        st.write("### Rango de Precio (1 Año):")
        st.write(info_etf['rango_1y'])
        
        st.write("### Rendimiento YTD (Año hasta la fecha):")
        st.write(f"{info_etf['rendimiento_ytd']} %")
        
        st.write("### Moneda en la que cotiza:")
        st.write(info_etf['moneda'])
        
        st.write("### Beta del ETF:")
        st.write(info_etf['beta'])
        
        # Graficar el rendimiento histórico
        fig = px.line(datos_2010_2023[ticker], x=datos_2010_2023[ticker].index, y="Close", title=f"Precio de Cierre - {ticker}")
        st.plotly_chart(fig)

        # Calcular y graficar Drawdown y Watermark
        drawdown, watermark = calcular_drawdown(datos_2010_2023[ticker])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=datos_2010_2023[ticker].index, y=drawdown, mode='lines', name='Drawdown'))
        fig.add_trace(go.Scatter(x=datos_2010_2023[ticker].index, y=watermark, mode='lines', name='Watermark'))
        fig.update_layout(title=f"Drawdown y Watermark - {ticker}", xaxis_title="Fecha", yaxis_title="Valor")
        st.plotly_chart(fig)

        # Si se tienen los top holdings, mostrar una tabla
        if isinstance(info_etf['composicion'], list) and len(info_etf['composicion']) > 0:
            st.write("### Principales Activos (Top Holdings):")
            df_composicion = pd.DataFrame(info_etf['composicion'])
            st.dataframe(df_composicion[['symbol', 'holdingPercent']])

# --- Estadísticas de los ETF's ---
with tabs[2]:
    st.header("Estadísticas de los ETF's (2010-2023)")
    for ticker, descripcion in tickers.items():
        st.subheader(f"{ticker} - {descripcion}")
        metricas = calcular_metricas(datos_2010_2023[ticker])
        st.write(pd.DataFrame(metricas, index=["Valor"]).T)

        # Graficar distribución de retornos
        fig = px.histogram(datos_2010_2023[ticker].dropna(), x="Retornos", nbins=50, title=f"Distribución de Retornos - {ticker}")
        st.plotly_chart(fig)

# --- Portafolios Óptimos ---
with tabs[3]:
    st.header("Portafolios Óptimos (2010-2020)")
    datos_2010_2020 = cargar_datos(tickers.keys(), "2010-01-01", "2020-01-01")
    retornos_2010_2020 = pd.DataFrame({k: v["Retornos"] for k, v in datos_2010_2020.items()}).dropna()

    # 1. Mínima Volatilidad
    st.subheader("Portafolio de Mínima Volatilidad")
    pesos_min_vol = optimizar_portafolio(retornos_2010_2020, metodo="min_vol")
    st.write("Pesos Óptimos (Mínima Volatilidad):")
    for ticker, peso in zip(tickers.keys(), pesos_min_vol):
        st.write(f"{ticker}: {peso:.2%}")
    fig = px.bar(x=tickers.keys(), y=pesos_min_vol, title="Pesos - Mínima Volatilidad")
    st.plotly_chart(fig)

    # 2. Máximo Sharpe Ratio
    st.subheader("Portafolio de Máximo Sharpe Ratio")
    pesos_sharpe = optimizar_portafolio(retornos_2010_2020, metodo="sharpe")
    st.write("Pesos Óptimos (Máximo Sharpe Ratio):")
    for ticker, peso in zip(tickers.keys(), pesos_sharpe):
        st.write(f"{ticker}: {peso:.2%}")
    fig = px.bar(x=tickers.keys(), y=pesos_sharpe, title="Pesos - Máximo Sharpe Ratio")
    st.plotly_chart(fig)

    # 3. Mínima Volatilidad con Rendimiento Objetivo
    rendimiento_objetivo = 0.10 / 252  # Rendimiento objetivo anualizado
    st.subheader("Portafolio de Mínima Volatilidad con Rendimiento Objetivo (10% Anual)")
    pesos_target = optimizar_portafolio(retornos_2010_2020, metodo="target", objetivo=rendimiento_objetivo)
    st.write("Pesos Óptimos (Rendimiento Objetivo):")
    for ticker, peso in zip(tickers.keys(), pesos_target):
        st.write(f"{ticker}: {peso:.2%}")
    fig = px.bar(x=tickers.keys(), y=pesos_target, title="Pesos - Rendimiento Objetivo (10%)")
    st.plotly_chart(fig)

# --- Backtesting ---
with tabs[4]:
    st.header("Backtesting (2021-2023)")
    datos_2021_2023 = cargar_datos(tickers.keys(), "2021-01-01", "2023-01-01")
    retornos_2021_2023 = pd.DataFrame({k: v["Retornos"] for k, v in datos_2021_2023.items()}).dropna()

    st.subheader("Rendimientos Acumulados de los Portafolios")
    rendimientos_portafolio = pd.DataFrame(index=retornos_2021_2023.index)
    for nombre, pesos in [("Mínima Volatilidad", pesos_min_vol), 
                          ("Sharpe Ratio", pesos_sharpe), 
                          ("Rendimiento Objetivo", pesos_target)]:
        rendimientos = retornos_2021_2023 @ pesos
        rendimientos_portafolio[nombre] = rendimientos.cumsum()
    
    fig = px.line(rendimientos_portafolio, x=rendimientos_portafolio.index, title="Rendimientos Acumulados de los Portafolios")
    st.plotly_chart(fig)

    # Heatmap de Correlación
    st.subheader("Heatmap de Correlación de los ETFs")
    correlacion = retornos_2021_2023.corr()
    fig = px.imshow(correlacion, text_auto=True, title="Correlación entre los ETFs")
    st.plotly_chart(fig)
