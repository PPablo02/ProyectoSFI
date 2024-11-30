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

def cargar_datos(etfs, start_date="2010-01-01", end_date="2023-12-31"):
        """Cargar datos ajustados de cierre y calcular rendimientos diarios."""
        data = yf.download(etfs, start=start_date, end=end_date)["Adj Close"]
        rendimientos = data.pct_change().dropna()
        return data, rendimientos

def calcular_metricas(rendimientos):
        """Calcular métricas estadísticas para un DataFrame de rendimientos."""
        estadisticas = pd.DataFrame(index=rendimientos.columns)

        # Media diaria
        estadisticas['Media diaria'] = rendimientos.mean()

        # Sesgo
        estadisticas['Sesgo'] = rendimientos.skew()

        # Exceso de curtosis
        estadisticas['Exceso de Curtosis'] = rendimientos.kurtosis()

        # VaR y CVaR para distintos niveles de confianza
        niveles_confianza = [0.95, 0.975, 0.99]
        for nivel in niveles_confianza:
            estadisticas[f'VaR {int(nivel*100)}%'] = rendimientos.quantile(1 - nivel)
            estadisticas[f'CVaR {int(nivel*100)}%'] = rendimientos[rendimientos < rendimientos.quantile(1 - nivel)].mean()

        # Sharpe Ratio (suponiendo una tasa libre de riesgo diaria de 0)
        estadisticas['Sharpe Ratio'] = rendimientos.mean() / rendimientos.std()

        # Sortino Ratio
        downside = rendimientos[rendimientos < 0].std()
        estadisticas['Sortino Ratio'] = rendimientos.mean() / downside

        # Drawdown Máximo
        cumulative_returns = (1 + rendimientos).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        estadisticas['Drawdown Máximo'] = drawdown.min()

        return estadisticas

def graficar_drawdown(data, rendimientos, nombre_etf):
        """Graficar serie de tiempo del ETF con drawdowns y watermarks."""
        cumulative_returns = (1 + rendimientos).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max

        plt.figure(figsize=(12, 6))
        plt.plot(data.index, cumulative_returns, label="Retornos acumulados")
        plt.plot(data.index, running_max, label="Watermark", linestyle="--")
        plt.fill_between(data.index, cumulative_returns.squeeze(), running_max.squeeze(), 
                         color='red', alpha=0.3, label="Drawdown")
        plt.title(f"Drawdown y Watermarks para {nombre_etf}")
        plt.xlabel("Fecha")
        plt.ylabel("Valor acumulado")
        plt.legend()
        st.pyplot(plt)

def graficar_rendimientos(rendimientos, nombre_etf):
        """Graficar rendimientos diarios para un ETF."""
        plt.figure(figsize=(12, 6))
        plt.plot(rendimientos.index, rendimientos, label="Rendimientos diarios")
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        plt.title(f"Rendimientos Diarios para {nombre_etf}")
        plt.xlabel("Fecha")
        plt.ylabel("Rendimiento Diario")
        plt.legend()
        st.pyplot(plt)


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
with tabs[2]:
    st.title("Estadísticas de los Activos")

    st.write("""
    En este apartado se calculan los rendimientos diarios y métricas clave de los ETFs seleccionados. 
    Además, se visualizan los rendimientos diarios y se destacan los drawdowns y watermarks 
    para cada ETF desde 2010 hasta 2023.
    """)

    # --- Descargar datos y calcular métricas ---
    etfs = ["LQD", "VWOB", "SPY", "EEM", "DBC"]
    data, rendimientos = cargar_datos(etfs)

    # Calcular métricas para cada ETF
    estadisticas = calcular_metricas(rendimientos)
    st.subheader("Métricas Estadísticas de los ETFs")
    st.dataframe(estadisticas)

    # --- Graficar resultados para cada ETF ---
    st.subheader("Visualizaciones por ETF")
    for etf in etfs:
        st.write(f"### {etf}")
        
        # Graficar rendimientos diarios
        st.write("#### Rendimientos Diarios")
        graficar_rendimientos(rendimientos[etf], etf)

        # Graficar Drawdowns y Watermarks
        st.write("#### Drawdowns y Watermarks")
        graficar_drawdown(data[[etf]], rendimientos[etf], etf)
    
# --- Portafolios Óptimos ---
with tabs[3]:
     st.header("Portafolios Óptimos")
    
# --- Backtesting ---
with tabs[4]:
     st.header("Backtesting")
    
# --- Modelo Black-Litterman ---
with tabs[5]:
     st.header("Modelo Black-Litterman")
    
