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

    # Función para calcular la volatilidad del portafolio
    def riesgo(w):
        return np.sqrt(np.dot(w.T, np.dot(cov, w)))

    # Función para calcular el Sharpe ratio del portafolio
    def sharpe(w):
        return -(np.dot(w.T, media) / np.sqrt(np.dot(w.T, np.dot(cov, w))))

    n = len(media)
    w_inicial = np.ones(n) / n  # Inicializamos pesos de manera equitativa
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]  # Suma de pesos debe ser 1

    if metodo == "target":
        restricciones.append({"type": "eq", "fun": lambda w: np.dot(w, media) - objetivo})  # Objetivo de rendimiento
        objetivo_funcion = riesgo
    elif metodo == "sharpe":
        objetivo_funcion = sharpe
    else:
        objetivo_funcion = riesgo

    limites = [(0, 1) for _ in range(n)]  # Pesos entre 0 y 1
    resultado = minimize(objetivo_funcion, w_inicial, constraints=restricciones, bounds=limites)
    
    # Asegurarse de que el resultado es un array unidimensional
    return np.array(resultado.x).flatten()

def calcular_metricas(df, nivel_VaR=[0.95, 0.975, 0.99]):
    """Calcula métricas estadísticas clave, incluyendo VaR a diferentes niveles."""
    retornos = df['Retornos'].dropna()  # Retornos del ETF

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

    # Calcular el Beta
    sp500 = yf.download("^GSPC", start=df.index[0], end=df.index[-1])['Adj Close']
    sp500_retornos = sp500.pct_change().dropna()  # Retornos del índice S&P 500

    # Alinear fechas entre retornos del ETF y del índice S&P 500
    retornos_alineados = retornos.reindex(sp500_retornos.index).dropna()
    sp500_retornos_alineados = sp500_retornos.reindex(retornos_alineados.index).dropna()

    if len(retornos_alineados) > 0 and len(sp500_retornos_alineados) > 0:
        covarianza = np.cov(retornos_alineados, sp500_retornos_alineados)[0, 1]
        var_sp500 = np.var(sp500_retornos_alineados)
        beta = covarianza / var_sp500 if var_sp500 != 0 else np.nan
    else:
        beta = np.nan  # No se puede calcular beta si no hay datos suficientes

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
    # Mantén aquí tu sección actual de selección de ETFs

# --- Estadísticas de los ETF's ---
with tabs[2]:
    st.header("Estadísticas de los ETF's (2010-2023)")
    for ticker, descripcion in tickers.items():
        st.subheader(f"{descripcion['nombre']}")
        metricas = calcular_metricas(datos_2010_2023[ticker])
        st.write(metricas)

# --- Librerías ---
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

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
        return np.sqrt(np.dot(w.T, np.dot(cov, w)))

    def sharpe(w):
        return -(np.dot(w.T, media) / np.sqrt(np.dot(w.T, np.dot(cov, w))))

    n = len(media)
    w_inicial = np.ones(n) / n  # Inicializamos pesos equitativamente
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]  # Suma de pesos = 1

    if metodo == "target":
        restricciones.append({"type": "eq", "fun": lambda w: np.dot(w, media) - objetivo})  # Rendimiento objetivo
        objetivo_funcion = riesgo
    elif metodo == "sharpe":
        objetivo_funcion = sharpe
    else:
        objetivo_funcion = riesgo

    limites = [(0, 1) for _ in range(n)]  # Pesos entre 0 y 1
    resultado = minimize(objetivo_funcion, w_inicial, constraints=restricciones, bounds=limites)
    return np.array(resultado.x).flatten()

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

    if len(retornos_alineados) > 0 and len(sp500_retornos_alineados) > 0:
        covarianza = np.cov(retornos_alineados, sp500_retornos_alineados)[0, 1]
        var_sp500 = np.var(sp500_retornos_alineados)
        beta = covarianza / var_sp500 if var_sp500 != 0 else np.nan
    else:
        beta = np.nan

    # Métricas completas
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
    Este proyecto analiza y optimiza portafolios utilizando ETFs diversificados. 
    Incluye métricas de riesgo, optimización de portafolios, y backtesting.
    """)

# --- Selección de ETF's ---
with tabs[1]:
    st.header("Selección de ETF's")
    tickers = {
        "TLT": {"nombre": "iShares 20+ Year Treasury Bond ETF", "descripcion": "Bonos del Tesoro de EE.UU. a largo plazo."},
        "EMB": {"nombre": "iShares JP Morgan USD Emerging Markets Bond ETF", "descripcion": "Bonos de mercados emergentes."},
        "SPY": {"nombre": "SPDR S&P 500 ETF Trust", "descripcion": "Acciones de las 500 empresas más grandes de EE.UU."},
        "VWO": {"nombre": "Vanguard FTSE Emerging Markets ETF", "descripcion": "Acciones de mercados emergentes."},
        "GLD": {"nombre": "SPDR Gold Shares", "descripcion": "Oro físico."}
    }

    for ticker, info in tickers.items():
        st.subheader(f"{info['nombre']} ({ticker})")
        st.write(info["descripcion"])

# --- Estadísticas de los ETF's ---
with tabs[2]:
    st.header("Estadísticas de los ETF's (2010-2023)")
    datos_2010_2023 = cargar_datos(tickers.keys(), "2010-01-01", "2023-01-01")

    for ticker, info in tickers.items():
        st.subheader(f"{info['nombre']} ({ticker})")
        metricas = calcular_metricas(datos_2010_2023[ticker])
        st.write(metricas)

            # Graficar distribución de retornos
            fig = px.histogram(datos_2010_2023[ticker]['Retornos'].dropna(),
                               x="Retornos", nbins=50,
                               title=f"Distribución de Retornos - {info['nombre']}")
            st.plotly_chart(fig)
    

# --- Portafolios Óptimos ---
with tabs[3]:
    st.header("Portafolios Óptimos (2010-2020)")
    datos_2010_2020 = cargar_datos(tickers.keys(), "2010-01-01", "2020-01-01")
    retornos_2010_2020 = pd.DataFrame({k: v["Retornos"] for k, v in datos_2010_2020.items()}).dropna()

    # Mínima Volatilidad
    st.subheader("Portafolio de Mínima Volatilidad")
    pesos_min_vol = optimizar_portafolio(retornos_2010_2020, metodo="min_vol")
    fig = px.bar(x=tickers.keys(), y=pesos_min_vol, title="Pesos - Mínima Volatilidad")
    st.plotly_chart(fig)

# --- Backtesting ---
with tabs[4]:
    st.header("Backtesting (2021-2023)")
    datos_2021_2023 = cargar_datos(tickers.keys(), "2021-01-01", "2023-01-01")
    retornos_2021_2023 = pd.DataFrame({k: v["Retornos"] for k, v in datos_2021_2023.items()}).dropna()

    rendimientos_portafolio = pd.DataFrame(index=retornos_2021_2023.index)
    rendimientos_portafolio["Mínima Volatilidad"] = np.dot(retornos_2021_2023, pesos_min_vol)

    fig = px.line(rendimientos_portafolio.cumsum(), 
                  title="Rendimientos Acumulados - Portafolios")
    st.plotly_chart(fig)
        # Graficar distribución de retornos
    fig = px.histogram(datos_2010_2023[ticker]['Retornos'].dropna(), 
                    x="Retornos", nbins=50, 
                    title=f"Distribución de Retornos - {descripcion['nombre']}")
    st.plotly_chart(fig)

# --- Portafolios Óptimos ---
with tabs[3]:
    st.header("Portafolios Óptimos (2010-2020)")
    

# --- Backtesting ---
with tabs[4]:
    st.header("Backtesting (2021-2023)")
    
