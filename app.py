# Librerías
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize
from datetime import datetime


# --- Streamlit UI ---
st.title("Proyecto de Optimización de Portafolios")

# Crear tabs
tabs = st.tabs(["Introducción", "Selección de ETFs", "Estadísticas de Activos", "Portafolios Óptimos", "Backtesting", "Modelo Black-Litterman"])

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
    st.header("Selección de ETFs")
    
    st.write("""
    En esta sección se seleccionarán 5 ETFs con características variadas para construir un portafolio balanceado. 
    Los ETFs cubren diferentes clases de activos, como renta fija, renta variable y materias primas, 
    y están denominados en la misma divisa (USD). A continuación, se describen los ETFs seleccionados y sus características.
    """)

    # Función para cargar los datos de los ETFs (renombrada como ventana1)
    def ventana1(etfs, start_date="2010-01-01"):
        end_date = datetime.today().strftime('%Y-%m-%d')  # Fecha actual
        data = yf.download(etfs, start=start_date, end=end_date)["Adj Close"]
        returns = data.pct_change().dropna()
        return data, returns

    # Lista de ETFs seleccionados
    etfs = {
        "LQD": {"name": "iShares iBoxx $ Investment Grade Corporate Bond ETF", 
                "type": "Renta Fija Desarrollada", 
                "index": "iBoxx $ Liquid Investment Grade Index", 
                "currency": "USD", 
                "risk_metrics": {"Duration": 6.8, "Beta": 0.12}, 
                "contributors": ["Apple", "Microsoft", "Amazon"], 
                "style": "Investment Grade", 
                "cost": 0.14},
        "VWOB": {"name": "Vanguard Emerging Markets Government Bond ETF", 
                 "type": "Renta Fija Emergente", 
                 "index": "Bloomberg Barclays EM USD Govt 10-30 Year Bond Index", 
                 "currency": "USD", 
                 "risk_metrics": {"Duration": 8.4, "Beta": 0.75}, 
                 "contributors": ["Brazil", "Russia", "India"], 
                 "style": "Emerging Market", 
                 "cost": 0.36},
        "SPY": {"name": "SPDR S&P 500 ETF Trust", 
                "type": "Renta Variable Desarrollada", 
                "index": "S&P 500", 
                "currency": "USD", 
                "risk_metrics": {"Duration": "N/A", "Beta": 1.00}, 
                "contributors": ["Apple", "Microsoft", "Tesla"], 
                "style": "Large Cap, Growth", 
                "cost": 0.09},
        "EEM": {"name": "iShares MSCI Emerging Markets ETF", 
                "type": "Renta Variable Emergente", 
                "index": "MSCI Emerging Markets Index", 
                "currency": "USD", 
                "risk_metrics": {"Duration": "N/A", "Beta": 1.12}, 
                "contributors": ["China", "Taiwan", "India"], 
                "style": "Emerging Market", 
                "cost": 0.68},
        "DBC": {"name": "Invesco DB Commodity Index Tracking Fund", 
                "type": "Materias Primas", 
                "index": "DBIQ Optimum Yield Diversified Commodity Index", 
                "currency": "USD", 
                "risk_metrics": {"Duration": "N/A", "Beta": 0.80}, 
                "contributors": ["Crude Oil", "Gold", "Copper"], 
                "style": "Commodity", 
                "cost": 0.89}
    }

    # Mostrar las características de cada ETF
    for etf, details in etfs.items():
        st.subheader(f"{details['name']} ({etf})")
        st.write(f"**Tipo**: {details['type']}")
        st.write(f"**Índice que sigue**: {details['index']}")
        st.write(f"**Moneda de denominación**: {details['currency']}")
        st.write(f"**Principales contribuyentes**: {', '.join(details['contributors'])}")
        st.write(f"**Riesgo**: Duración = {details['risk_metrics']['Duration']} años, Beta = {details['risk_metrics']['Beta']}")
        st.write(f"**Estilo**: {details['style']}")
        st.write(f"**Costo de gestión anual**: {details['cost']}%")
        
        # Cargar los datos del ETF
        data, returns = ventana1([etf], start_date="2010-01-01")
        
        # Último precio de cierre
        last_close_price = data.iloc[-1][etf]
        st.write(f"**Último precio de cierre de {etf}**: ${last_close_price:.2f}")
        
        # Crear gráfico interactivo de la serie de tiempo del ETF
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data[etf], mode='lines', name=f"{etf} Precio Ajustado de Cierre"))
        
        # Configuración del gráfico
        fig.update_layout(
            title=f"Serie de Tiempo del ETF: {details['name']}",
            xaxis_title='Fecha',
            yaxis_title='Precio Ajustado de Cierre',
            template='plotly_dark'
        )
        
        # Mostrar gráfico interactivo
        st.plotly_chart(fig)
        
# --- Estadísticas de Activos ---
with tabs[2]:
    st.header("Estadísticas de Activos")
    
    stats = {etf: calculate_statistics(returns[etf]) for etf in etfs}
    for etf in etfs:
        st.subheader(f"Estadísticas de {etf}")
        st.write(stats[etf])
    
    # Mostrar gráficos de distribución de rendimientos
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for etf in etfs:
        ax.hist(returns[etf], bins=50, alpha=0.5, label=etf)
    ax.set_title("Distribución de Rendimientos Diarios")
    ax.legend()
    st.pyplot(fig)

# --- Portafolios Óptimos ---
with tabs[3]:
    st.header("Portafolios Óptimos")
    
    # Optimizar portafolios con mínima volatilidad y máximo Sharpe ratio
    optimal_weights_min_vol = optimize_min_volatility(returns)
    optimal_weights_max_sharpe = optimize_max_sharpe(returns)
    
    st.write("Pesos del Portafolio de Mínima Volatilidad:", optimal_weights_min_vol)
    st.write("Pesos del Portafolio con Máximo Sharpe Ratio:", optimal_weights_max_sharpe)
    
    # Mostrar gráficos de rendimiento de los portafolios optimizados
    portfolio_min_vol = backtest_portfolio(returns, optimal_weights_min_vol)
    portfolio_max_sharpe = backtest_portfolio(returns, optimal_weights_max_sharpe)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=portfolio_min_vol, mode='lines', name='Mínima Volatilidad'))
    fig.add_trace(go.Scatter(x=data.index, y=portfolio_max_sharpe, mode='lines', name='Máximo Sharpe Ratio'))
    fig.update_layout(title="Backtesting de Portafolios Óptimos")
    st.plotly_chart(fig)

# --- Backtesting ---
with tabs[4]:
    st.header("Backtesting")
    
    # Mostrar el rendimiento acumulado del portafolio optimizado
    st.write("Evaluando los portafolios de 2021 a 2023...")
    
    # Realizar backtesting con los datos de 2021 a 2023
    data_backtest, returns_backtest = load_data(etfs, start_date="2021-01-01", end_date="2023-12-31")
    
    portfolio_min_vol_backtest = backtest_portfolio(returns_backtest, optimal_weights_min_vol)
    portfolio_max_sharpe_backtest = backtest_portfolio(returns_backtest, optimal_weights_max_sharpe)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_backtest.index, y=portfolio_min_vol_backtest, mode='lines', name='Mínima Volatilidad'))
    fig.add_trace(go.Scatter(x=data_backtest.index, y=portfolio_max_sharpe_backtest, mode='lines', name='Máximo Sharpe Ratio'))
    fig.update_layout(title="Backtesting de Portafolios 2021-2023")
    st.plotly_chart(fig)

# --- Modelo Black-Litterman ---
with tabs[5]:
    st.header("Modelo Black-Litterman")
    st.write("Implementación del modelo de optimización Black-Litterman para ajustar los rendimientos esperados.")
    st.write("Aquí puedes agregar tus perspectivas sobre los activos y cómo el modelo ajusta los rendimientos esperados.")
