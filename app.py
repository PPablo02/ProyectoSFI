# Librerías
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, kurtosis
from datetime import datetime

# --- Funciones auxiliares ---

# Función para cargar los datos de los ETFs del tab1
def ventana1(etfs, start_date="2010-01-01"):
    end_date = datetime.today().strftime('%Y-%m-%d')  # Fecha actual
    data = yf.download(etfs, start=start_date, end=end_date)["Adj Close"]
    returns = data.pct_change().dropna()
    return data, returns

# --- Funciones para el tab2 ---

def calcular_metricas(returns):
    # Cálculos básicos
    media = returns.mean()
    sesgo = skew(returns)
    curtosis = kurtosis(returns)
    
    # VaR y CVaR (Niveles de confianza al 95%, 97.5% y 99%)
    var_95 = np.percentile(returns, 5)
    var_975 = np.percentile(returns, 2.5)
    var_99 = np.percentile(returns, 1)
    
    cvar_95 = returns[returns <= var_95].mean()
    cvar_975 = returns[returns <= var_975].mean()
    cvar_99 = returns[returns <= var_99].mean()
    
    # Sharpe y Sortino
    sharpe_ratio = media / returns.std()
    sortino_ratio = media / returns[returns < 0].std()
    
    # Drawdown
    cumulative_returns = (returns + 1).cumprod() - 1
    max_drawdown = (cumulative_returns - cumulative_returns.cummax()).min()
    
    # Watermark (máximo valor alcanzado y el punto más bajo de drawdown)
    watermark = cumulative_returns.cummax()
    lowest_point = cumulative_returns.min()
    
    # Resultados
    return {
        'Media': media,
        'Sesgo': sesgo,
        'Curtosis': curtosis,
        'VaR (95%)': var_95,
        'VaR (97.5%)': var_975,
        'VaR (99%)': var_99,
        'CVaR (95%)': cvar_95,
        'CVaR (97.5%)': cvar_975,
        'CVaR (99%)': cvar_99,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Drawdown Máximo': max_drawdown,
        'Watermark Máximo': watermark.iloc[-1],
        'Punto más Bajo del Drawdown': lowest_point
    }

# --- Función para cargar datos ---
def ventana2(etfs, start_date="2010-01-01", end_date="2023-12-31"):
    data = yf.download(etfs, start=start_date, end=end_date)["Adj Close"]
    returns = data.pct_change().dropna()
    return data, returns
    
# --- Streamlit UI ---
st.title("Proyecto de Optimización de Portafolios")

# Crear tabs
tabs = st.tabs(["Introducción", "Selección de ETF's", "Estadísticas de los ETF's", "Portafolios Óptimos", "Backtesting", "Modelo Black-Litterman"])

# --- Introducción ---
with tabs[0]:
    st.header("Introducción")
    st.write("""
    Este proyecto tiene como objetivo analizar y optimizar un portafolio utilizando ETFs en diferentes clases de activos, tales como renta fija, renta variable, y materias primas. A lo largo del proyecto, se evaluará el rendimiento de estos activos a través de diversas métricas financieras y técnicas de optimización de portafolios, como la optimización de mínima volatilidad y la maximización del Sharpe Ratio.
    
    Para lograr esto, se utilizarán datos históricos de rendimientos y se realizarán pruebas de backtesting para validar las estrategias propuestas. Además, se implementará el modelo de optimización Black-Litterman para ajustar los rendimientos esperados en función de perspectivas macroeconómicas.
    
    Los integrantes de este proyecto son:
    - Emmanuel Reyes Hernández
    - Adrián Fuentes Soriano
    - Pablo Pineda Pineda
    - Mariana Vigil Villegas
    """)

# --- Selección de ETFs ---
with tabs[1]:
    st.header("Selección de ETF's")
    
    st.write("""
    En esta sección se seleccionarán 5 ETFs con características variadas para construir un portafolio balanceado. 
    Los ETFs cubren diferentes clases de activos, como renta fija, renta variable y materias primas, 
    y están denominados en la misma divisa (USD). A continuación, se describen los ETFs seleccionados y sus características.
    """)

    # Lista de ETFs seleccionados con atributos
    etfs = {
        "LQD": {"nombre": "iShares iBoxx $ Investment Grade Corporate Bond ETF", 
                "tipo": "Renta Fija Desarrollada", 
                "índice": "iBoxx $ Liquid Investment Grade Index", 
                "moneda": "USD", 
                "métricas_riesgo": {"Duración": 6.8, "Beta": 0.12, "Volatilidad": 5.5, "Drawdown Máximo": 8.2}, 
                "principales_contribuidores": ["Apple", "Microsoft", "Amazon"], 
                "países_invertidos": "Estados Unidos", 
                "estilo": "Investment Grade", 
                "grado_inversión": "Alto", 
                "costo": 0.14,
                "exposición": "Bonos corporativos de grado de inversión",
                "url": "https://www.blackrock.com/us/products/239726/"},
        "VWOB": {"nombre": "Vanguard Emerging Markets Government Bond ETF", 
                 "tipo": "Renta Fija Emergente", 
                 "índice": "Bloomberg Barclays EM USD Govt 10-30 Year Bond Index", 
                 "moneda": "USD", 
                 "métricas_riesgo": {"Duración": 8.4, "Beta": 0.75, "Volatilidad": 9.2, "Drawdown Máximo": 15.3}, 
                 "principales_contribuidores": ["Brasil", "Rusia", "India"], 
                 "países_invertidos": "Mercados Emergentes", 
                 "estilo": "Mercados Emergentes",
                 "grado_inversión": "Medio", 
                 "costo": 0.36,
                 "exposición": "Bonos del gobierno en mercados emergentes",
                 "url": "https://investor.vanguard.com/etf/profile/VWOB"},
        "SPY": {"nombre": "SPDR S&P 500 ETF Trust", 
                "tipo": "Renta Variable Desarrollada", 
                "índice": "S&P 500", 
                "moneda": "USD", 
                "métricas_riesgo": {"Duración": "N/A", "Beta": 1.00, "Volatilidad": 14.2, "Drawdown Máximo": 18.5}, 
                "principales_contribuidores": ["Apple", "Microsoft", "Nvidia"], 
                "países_invertidos": "Estados Unidos",
                "estilo": "Large Cap, Growth", 
                "grado_inversión": "Alto", 
                "costo": 0.09,
                "exposición": "Acciones de grandes empresas en EE. UU.",
                "url": "https://www.ssga.com/us/en/individual/etfs/fund-spdr-sp-500-etf-trust-spy"},
        "EEM": {"nombre": "iShares MSCI Emerging Markets ETF", 
                "tipo": "Renta Variable Emergente", 
                "índice": "MSCI Emerging Markets Index", 
                "moneda": "USD", 
                "métricas_riesgo": {"Duración": "N/A", "Beta": 1.12, "Volatilidad": 20.5, "Drawdown Máximo": 25.7}, 
                "principales_contribuidores": ["China", "Taiwán", "India"], 
                "países_invertidos": "Mercados Emergentes",
                "estilo": "Mercados Emergentes",
                "grado_inversión": "Medio", 
                "costo": 0.68,
                "exposición": "Acciones de empresas en mercados emergentes",
                "url": "https://www.ishares.com/us/products/etf-investments#!type=ishares&style=ishares&view=keyFacts&fund=EEM"},
        "DBC": {"nombre": "Invesco DB Commodity Index Tracking Fund", 
                "tipo": "Materias Primas", 
                "índice": "DBIQ Optimum Yield Diversified Commodity Index", 
                "moneda": "USD", 
                "métricas_riesgo": {"Duración": "N/A", "Beta": 0.80, "Volatilidad": 18.0, "Drawdown Máximo": 30.0}, 
                "principales_contribuidores": ["Petróleo", "Oro", "Cobre"], 
                "países_invertidos": "Diversos", 
                "estilo": "Diversificación de Materias Primas", 
                "grado_inversión": "N/A", 
                "costo": 0.79,
                "exposición": "Diversificación en commodities",
                "url": "https://www.invesco.com/portal/site/us/investors/etfs/product-detail?productId=DBC"}
    }

    # Mostrar información de los ETFs
    for etf, data in etfs.items():
        st.write(f"**{data['nombre']}**")
        st.write(f"- Tipo: {data['tipo']}")
        st.write(f"- Índice: {data['índice']}")
        st.write(f"- Moneda: {data['moneda']}")
        st.write(f"- Costo: {data['costo']}%")
        st.write(f"- Exposición: {data['exposición']}")
        st.write(f"- Grado de Inversión: {data['grado_inversión']}")
        st.write(f"- URL: [{data['url']}]({data['url']})")

# --- Estadísticas de los ETF's ---
with tabs[2]:
    st.header("Estadísticas de los ETF's")
    
    # Cargar los ETFs seleccionados
    selected_etfs = ["LQD", "VWOB", "SPY", "EEM", "DBC"]
    data, returns = ventana1(selected_etfs)
    
    # Mostrar métricas
    st.write("### Métricas estadísticas de los ETFs seleccionados")
    metrics = calcular_metricas(returns)
    st.write(pd.DataFrame(metrics, index=["Valor"]))
