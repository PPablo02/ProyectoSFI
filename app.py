import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize

# Función para cargar los datos de los ETFs
def load_data(etfs, start_date, end_date):
    data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()  # Calcular rendimientos diarios
    return data, returns

# Función para calcular estadísticas de los activos
def calculate_statistics(returns):
    stats = {}
    stats['mean'] = returns.mean()
    stats['skew'] = skew(returns)
    stats['kurtosis'] = kurtosis(returns)
    stats['VaR_95'] = returns.quantile(0.05)  # VaR al 5%
    stats['CVaR_95'] = returns[returns <= returns.quantile(0.05)].mean()  # CVaR al 5%
    stats['Sharpe'] = returns.mean() / returns.std()  # Asumiendo tasa libre de riesgo 0%
    stats['Sortino'] = returns.mean() / returns[returns < 0].std()  # Sortino ratio
    stats['drawdown'] = (returns.cumsum().min())  # Drawdown acumulado
    return stats

# Función para optimizar el portafolio con mínima volatilidad
def optimize_min_volatility(returns):
    cov_matrix = returns.cov()
    mean_returns = returns.mean()
    
    # Función de volatilidad
    def objective(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Restricción de que la suma de los pesos sea 1
    def constraint(weights):
        return np.sum(weights) - 1
    
    # Número de activos
    num_assets = len(mean_returns)
    
    # Pesos iniciales (equitativos)
    init_weights = np.ones(num_assets) / num_assets
    
    # Restricciones y límites
    cons = [{'type': 'eq', 'fun': constraint}]
    bounds = [(0, 1) for _ in range(num_assets)]
    
    # Optimización para mínima volatilidad
    result = minimize(objective, init_weights, method='SLSQP', constraints=cons, bounds=bounds)
    return result.x

# Función para optimizar el portafolio con máximo Sharpe ratio
def optimize_max_sharpe(returns):
    cov_matrix = returns.cov()
    mean_returns = returns.mean()
    
    # Función para calcular el Sharpe ratio negativo
    def objective(weights):
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility  # Maximizar el Sharpe ratio
    
    # Restricción de que la suma de los pesos sea 1
    def constraint(weights):
        return np.sum(weights) - 1
    
    # Número de activos
    num_assets = len(mean_returns)
    
    # Pesos iniciales (equitativos)
    init_weights = np.ones(num_assets) / num_assets
    
    # Restricciones y límites
    cons = [{'type': 'eq', 'fun': constraint}]
    bounds = [(0, 1) for _ in range(num_assets)]
    
    # Optimización para máximo Sharpe ratio
    result = minimize(objective, init_weights, method='SLSQP', constraints=cons, bounds=bounds)
    return result.x

# Función para hacer backtesting
def backtest_portfolio(returns, weights):
    portfolio_returns = np.dot(returns, weights)
    return portfolio_returns.cumsum()  # Rendimiento acumulado

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
    st.write("Selecciona 5 ETFs para formar un portafolio balanceado.")
    
    # Listado de ETFs
    etfs = ["LQD", "VWOB", "SPY", "EEM", "DBC"]
    st.write("Los ETFs seleccionados son:")
    st.write(etfs)
    
    # Cargar datos
    data, returns = load_data(etfs, start_date="2010-01-01", end_date="2023-12-31")
    st.write("Datos cargados correctamente.")

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
