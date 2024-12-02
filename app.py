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

def calcular_metricas(df, nivel_VaR=[0.95, 0.975, 0.99]):
    """Calcula métricas estadísticas clave, incluyendo VaR a diferentes niveles."""
    retornos = df['Retornos'].dropna()
    
    # Calcular la media y volatilidad
    media = np.mean(retornos) * 100  # Convertir a porcentaje
    volatilidad = np.std(retornos) * 100  # Convertir a porcentaje
    
    # Calcular el sesgo y curtosis
    sesgo = skew(retornos)
    curtosis = kurtosis(retornos)
    
    # Calcular el VaR a diferentes niveles
    VaR = {f"VaR {nivel*100}%": np.percentile(retornos, (1 - nivel) * 100) for nivel in nivel_VaR}
    
    # Calcular el CVaR (Conditional VaR)
    cVaR = {f"CVaR {nivel*100}%": retornos[retornos <= np.percentile(retornos, (1 - nivel) * 100)].mean() for nivel in nivel_VaR}
    
    # Calcular el Sharpe Ratio
    sharpe = np.mean(retornos) / np.std(retornos) if np.std(retornos) != 0 else np.nan
    
    # Calcular el Beta (relación entre los retornos del ETF y el mercado)
    sp500 = yf.download("^GSPC", start=df.index[0], end=df.index[-1])['Adj Close']
    sp500_retornos = sp500.pct_change().dropna()
    beta = np.cov(retornos, sp500_retornos)[0][1] / np.var(sp500_retornos)
    
    # Crear un diccionario con todas las métricas
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

# --- Configuración de Streamlit ---
st.title("Proyecto de Optimización de Portafolios")

# Crear tabs
tabs = st.tabs(["Introducción", "Selección de ETF's", "Estadísticas de los ETF's", "Portafolios Óptimos", "Backtesting"])

# --- Introducción ---
with tabs[0]:
    st.header("Introducción")
    st.write("""
    Este proyecto tiene como objetivo analizar y optimizar un portafolio utilizando ETFs en diferentes clases de activos. 
    A continuación, se seleccionan 5 ETFs con características variadas para crear un portafolio diversificado. 
    Se calcularán las métricas de riesgo, el rendimiento de los portafolios y se optimizarán utilizando técnicas avanzadas como la optimización de mínima volatilidad, maximización del Sharpe Ratio y más.
    """)

# --- Selección de ETF's ---
with tabs[1]:
    st.header("Selección de ETF's")

    # Información de los ETFs
    tickers = {
        "TLT": {
            "nombre": "iShares 20+ Year Treasury Bond ETF (Renta Fija Desarrollada)",
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
            "nombre": "iShares JP Morgan USD Emerging Markets Bond ETF (Renta Fija Emergente)",
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
            "nombre": "SPDR S&P 500 ETF Trust (Renta Variable Desarrollada)",
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
            "nombre": "Vanguard FTSE Emerging Markets ETF (Renta Variable Emergente)",
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
            "nombre": "SPDR Gold Shares (Materias Primas)",
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

    # Descargar los datos de los ETFs
    tickers_lista = list(tickers.keys())
    datos_2010_2023 = cargar_datos(tickers_lista, "2010-01-01", "2023-01-01")

    # Mostrar información de cada ETF
    for ticker, info in tickers.items():
        st.subheader(f"{info['nombre']}")
        
        st.write(f"### Descripción del ETF:")
        st.write(info['descripcion'])
        
        st.write(f"### Sector de Inversión:")
        st.write(info['sector'])
        
        st.write(f"### Categoría del ETF:")
        st.write(info['categoria'])
        
        st.write(f"### Exposición del ETF:")
        st.write(info['exposicion'])
        
        st.write(f"### Moneda de Denominación:")
        st.write(info['moneda'])
        
        st.write(f"### Beta del ETF:")
        st.write(info['beta'])
        
        st.write(f"### Costos (Expense Ratio):")
        st.write(info['gastos'])
        
        st.write(f"### Rango de Precio (1 Año):")
        st.write(info['rango_1y'])
        
        st.write(f"### Rendimiento YTD (Año hasta la fecha):")
        st.write(info['rendimiento_ytd'])
        
        st.write(f"### Duración del ETF:")
        st.write(info['duracion'])
        
        st.write("### Principales Contribuidores (Top Holdings):")
        st.write(pd.DataFrame(info['top_holdings']))
        
        # Graficar el rendimiento histórico
        fig = px.line(datos_2010_2023[ticker], x=datos_2010_2023[ticker].index, y=datos_2010_2023[ticker]['Close'].values.flatten(), title=f"Precio de Cierre - {info['nombre']}")
        st.plotly_chart(fig)

# --- Estadísticas de los ETF's ---
with tabs[2]:
    st.header("Estadísticas de los ETF's (2010-2023)")
    for ticker, descripcion in tickers.items():
        st.subheader(f"{descripcion['nombre']}")
        metricas = calcular_metricas(datos_2010_2023[ticker])
        st.write(pd.DataFrame(metricas, index=["Valor"]).T)

        # Graficar distribución de retornos
        fig = px.histogram(datos_2010_2023[ticker].dropna(), x="Retornos", nbins=50, title=f"Distribución de Retornos - {descripcion['nombre']}")
        st.plotly_chart(fig)

# --- Portafolios Óptimos ---
with tabs[3]:
    st.header("Portafolios Óptimos (2010-2020)")
    datos_2010_2020 = cargar_datos(tickers_lista, "2010-01-01", "2020-01-01")
    retornos_2010_2020 = pd.DataFrame({k: v["Retornos"] for k, v in datos_2010_2020.items()}).dropna()

    # 1. Mínima Volatilidad
    st.subheader("Portafolio de Mínima Volatilidad")
    pesos_min_vol = optimizar_portafolio(retornos_2010_2020, metodo="min_vol")
    st.write("Pesos Óptimos (Mínima Volatilidad):")
    for ticker, peso in zip(tickers_lista, pesos_min_vol):
        st.write(f"{ticker}: {peso:.2%}")
    fig = px.bar(x=tickers_lista, y=pesos_min_vol, title="Pesos - Mínima Volatilidad")
    st.plotly_chart(fig)

    # 2. Máximo Sharpe Ratio
    st.subheader("Portafolio de Máximo Sharpe Ratio")
    pesos_sharpe = optimizar_portafolio(retornos_2010_2020, metodo="sharpe")
    st.write("Pesos Óptimos (Máximo Sharpe Ratio):")
    for ticker, peso in zip(tickers_lista, pesos_sharpe):
        st.write(f"{ticker}: {peso:.2%}")
    fig = px.bar(x=tickers_lista, y=pesos_sharpe, title="Pesos - Máximo Sharpe Ratio")
    st.plotly_chart(fig)

    # 3. Mínima Volatilidad con Rendimiento Objetivo
    rendimiento_objetivo = 0.10 / 252  # Rendimiento objetivo anualizado
    st.subheader("Portafolio de Mínima Volatilidad con Rendimiento Objetivo (10% Anual)")
    pesos_target = optimizar_portafolio(retornos_2010_2020, metodo="target", objetivo=rendimiento_objetivo)
    st.write("Pesos Óptimos (Rendimiento Objetivo):")
    for ticker, peso in zip(tickers_lista, pesos_target):
        st.write(f"{ticker}: {peso:.2%}")
    fig = px.bar(x=tickers_lista, y=pesos_target, title="Pesos - Rendimiento Objetivo (10%)")
    st.plotly_chart(fig)

# --- Backtesting ---
with tabs[4]:
    st.header("Backtesting (2021-2023)")
    datos_2021_2023 = cargar_datos(tickers_lista, "2021-01-01", "2023-01-01")
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
