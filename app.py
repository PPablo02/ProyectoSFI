# Librerías
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize
from datetime import datetime


# --- Funciones auxiliares ---

    # --- Funciones para el tab1 ---

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
                "principales_contribuidores": ["Petróleo Crudo", "Oro", "Cobre"], 
                "países_invertidos": "Global",
                "estilo": "Comodities", 
                "grado_inversión": "Bajo", 
                "costo": 0.89,
                "exposición": "Comodities y materias primas como petróleo, oro, cobre",
                "url": "https://www.invesco.com/portal/site/us/etfs/dbc"}
    }

    # Mostrar las características de cada ETF
    for etf, detalles in etfs.items():
        st.subheader(f"{detalles['nombre']} ({etf})")
        
        st.write(f"**Tipo**: {detalles['tipo']}")
        st.write(f"**Índice que sigue**: {detalles['índice']}")
        st.write(f"**Moneda de denominación**: {detalles['moneda']}")
        st.write(f"**Principales contribuyentes**: {', '.join(detalles['principales_contribuidores'])}")
        st.write(f"**País(es) donde invierte**: {detalles['países_invertidos']}")
        st.write(f"**Exposición**: {detalles['exposición']}")
        st.write(f"**Riesgo**: Duración = {detalles['métricas_riesgo']['Duración']} años, Beta = {detalles['métricas_riesgo']['Beta']}, Volatilidad = {detalles['métricas_riesgo']['Volatilidad']}%, Drawdown Máximo = {detalles['métricas_riesgo']['Drawdown Máximo']}%")
        st.write(f"**Estilo**: {detalles['estilo']}")
        st.write(f"**Grado de inversión**: {detalles['grado_inversión']}")
        st.write(f"**Costo de gestión anual**: {detalles['costo']}%")
        
        # Cargar los datos del ETF
        data, returns = ventana1([etf], start_date="2010-01-01")
        
        # Último precio de cierre
        ultimo_precio_cierre = data.iloc[-1][etf]
        st.write(f"**Último precio de cierre de {etf}**: ${ultimo_precio_cierre:.2f}")
        
        # Crear gráfico interactivo de la serie de tiempo del ETF
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data[etf], mode='lines', name=f"{etf} Precio Ajustado de Cierre"))
        
        # Configuración del gráfico
        fig.update_layout(
            title=f"Serie de Tiempo del ETF: {detalles['nombre']}",
            xaxis_title='Fecha',
            yaxis_title='Precio Ajustado de Cierre',
            template='plotly_dark'
        )
        
        # Mostrar gráfico interactivo
        st.plotly_chart(fig)
        
        # Enlace para más detalles del ETF
        st.markdown(f"[Más detalles sobre este ETF]({detalles['url']})")

    # --- Mostrar todas las series de tiempo en una sola gráfica ---
    st.subheader("Serie de Tiempo Combinada de Todos los ETFs")
    
    # Crear gráfico con todas las series de tiempo
    fig_all = go.Figure()

    for etf, detalles in etfs.items():
        data, returns = ventana1([etf], start_date="2010-01-01")
        fig_all.add_trace(go.Scatter(x=data.index, y=data[etf], mode='lines', name=detalles['nombre']))
    
    fig_all.update_layout(
        title="Serie de Tiempo Combinada de Todos los ETF's",
        xaxis_title='Fecha',
        yaxis_title='Precio Ajustado de Cierre',
        template='plotly_dark'
    )

    # Mostrar gráfico combinado
    st.plotly_chart(fig_all)


# --- Tab 2: Cálculo de Estadísticas ---
import seaborn as sns

# --- Tab 2: Cálculo de Estadísticas ---
with tabs[2]:
    st.header("Estadísticas de los Activos")
    
    st.write("""
    En esta sección, se calcularán varias métricas estadísticas de los 5 ETFs seleccionados. Los rendimientos diarios de cada ETF serán utilizados para calcular la media, el sesgo, la curtosis, el VaR (Value at Risk), el CVaR (Conditional Value at Risk), el Sharpe Ratio, el Sortino Ratio, y el Drawdown.
    """)

    # Definir los ETFs seleccionados
    etfs = ["LQD", "VWOB", "SPY", "EEM", "DBC"]
    data, returns = ventana2(etfs, start_date="2010-01-01", end_date=datetime.now().strftime("%Y-%m-%d"))
    
    # Crear un dataframe para almacenar los resultados de las métricas
    resultados = pd.DataFrame(columns=["Media", "Sesgo", "Curtosis", "VaR (95%)", "VaR (97.5%)", "VaR (99%)",
                                      "CVaR (95%)", "CVaR (97.5%)", "CVaR (99%)", "Sharpe Ratio", "Sortino Ratio",
                                      "Drawdown Máximo", "Watermark Máximo", "Punto más Bajo del Drawdown"], 
                              index=etfs)

    # Crear un dataframe para almacenar los rendimientos diarios
    rendimientos_diarios = returns
    
    # Calcular las métricas para cada ETF y llenar los dataframes
    for etf in etfs:
        resultados.loc[etf] = calcular_metricas(returns[etf])
    
    # Mostrar las métricas calculadas
    st.subheader("Métricas de Riesgo y Rendimiento de los ETFs")
    st.write(resultados)
    
    # Mostrar los rendimientos diarios de los ETFs
    st.subheader("Rendimientos Diarios de los ETFs")
    st.write(rendimientos_diarios)

    # Graficar por separado los rendimientos acumulados de cada ETF
    st.subheader("Rendimientos Acumulados (Por ETF)")
    for etf in etfs:
        cumulative_returns = (returns[etf] + 1).cumprod() - 1
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cumulative_returns.index, cumulative_returns, label=etf, color="blue")
        ax.set_title(f"Rendimientos Acumulados de {etf} (2010-2023)")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Rendimiento Acumulado")
        ax.legend(loc="best")
        st.pyplot(fig)

    # Graficar la distribución de los rendimientos para cada ETF
    st.subheader("Distribución de los Rendimientos Diarios (Por ETF)")
    for etf in etfs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(returns[etf], bins=50, alpha=0.7, color="green", label=etf)
        ax.set_title(f"Distribución de los Rendimientos Diarios de {etf}")
        ax.set_xlabel("Rendimiento Diario")
        ax.set_ylabel("Frecuencia")
        ax.legend(loc="best")
        st.pyplot(fig)

    # Graficar los Watermarks de cada ETF
    st.subheader("Watermarks (Por ETF)")
    for etf in etfs:
        cumulative_returns = (returns[etf] + 1).cumprod() - 1
        watermark = cumulative_returns.cummax()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(watermark.index, watermark, label=f"Watermark {etf}", color="purple")
        ax.set_title(f"Watermark de {etf}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Watermark (Valor Máximo)")
        ax.legend(loc="best")
        st.pyplot(fig)

    # Graficar los Drawdowns de cada ETF
    st.subheader("Drawdowns (Por ETF)")
    for etf in etfs:
        cumulative_returns = (returns[etf] + 1).cumprod() - 1
        drawdown = cumulative_returns - cumulative_returns.cummax()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(drawdown.index, drawdown, label=f"Drawdown {etf}", color="red")
        ax.set_title(f"Drawdown de {etf}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Drawdown")
        ax.legend(loc="best")
        st.pyplot(fig)

    # Graficar los rendimientos acumulados de todos los ETFs
    st.subheader("Rendimientos Acumulados de Todos los ETFs")
    cumulative_returns_all = (returns + 1).cumprod() - 1
    fig, ax = plt.subplots(figsize=(10, 6))
    for etf in etfs:
        ax.plot(cumulative_returns_all.index, cumulative_returns_all[etf], label=etf)
    ax.set_title("Rendimientos Acumulados de Todos los ETFs (2010-2023)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Rendimiento Acumulado")
    ax.legend(loc="best")
    st.pyplot(fig)

    # Heatmap de la matriz de covarianzas entre ETFs
    st.subheader("Mapa de Calor de la Matriz de Covarianzas entre ETFs")
    cov_matrix = returns.cov()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    ax.set_title("Matriz de Covarianzas entre ETFs")
    st.pyplot(fig)


# --- Portafolios Óptimos ---
with tabs[3]:
     st.header("Portafolios Óptimos")
    
# --- Backtesting ---
with tabs[4]:
     st.header("Backtesting")
    
# --- Modelo Black-Litterman ---
with tabs[5]:
     st.header("Modelo Black-Litterman")
    
