import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import skew, kurtosis
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.black_litterman import BlackLittermanModel
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la app
st.set_page_config(page_title="Portafolios Óptimos", layout="wide")

# Títulos
st.title("Gestión de Portafolios y Asset Allocation")
st.sidebar.title("Menú")

# Funciones auxiliares
def cargar_datos(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    return data

def calcular_estadisticas(data):
    rendimientos = data.pct_change().dropna()
    stats = {
        "Media": rendimientos.mean(),
        "Sesgo": rendimientos.apply(skew),
        "Exceso de Curtosis": rendimientos.apply(kurtosis),
        "VaR (95%)": rendimientos.quantile(0.05),
        "CVaR (95%)": rendimientos[rendimientos < rendimientos.quantile(0.05)].mean(),
    }
    return pd.DataFrame(stats)

def optimizar_portafolios(data, objetivo):
    rendimientos = data.pct_change().dropna()
    mu = mean_historical_return(rendimientos)
    S = CovarianceShrinkage(rendimientos).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    if objetivo == "Mínima Volatilidad":
        pesos = ef.min_volatility()
    elif objetivo == "Máximo Sharpe Ratio":
        pesos = ef.max_sharpe()
    elif objetivo == "Rendimiento Objetivo":
        ef.efficient_return(target_return=0.10)
        pesos = ef.clean_weights()
    return pesos, ef.portfolio_performance(verbose=True)

# Paso 1: Selección de Activos
st.sidebar.header("Paso 1: Selección de Activos")
tickers = st.sidebar.text_input("Introduce los símbolos de 5 ETFs separados por comas:", value="SPY,EEM,TLT,GLD,LQD")

# Paso 2: Cargar Datos
if st.sidebar.button("Cargar Datos"):
    start_date = "2010-01-01"
    end_date = "2023-01-01"
    tickers_list = [ticker.strip() for ticker in tickers.split(",")]
    datos = cargar_datos(tickers_list, start_date, end_date)
    st.write("### Previsualización de los datos cargados")
    st.write(datos.head())

    # Paso 3: Estadísticas
    st.write("### Estadísticas de los Activos")
    stats = calcular_estadisticas(datos)
    st.dataframe(stats)

    # Paso 4: Optimización de Portafolios
    st.write("### Portafolios Óptimos")
    objetivo = st.selectbox("Selecciona el objetivo de optimización:", ["Mínima Volatilidad", "Máximo Sharpe Ratio", "Rendimiento Objetivo"])
    if st.button("Optimizar Portafolio"):
        pesos, rendimiento = optimizar_portafolios(datos, objetivo)
        st.write(f"Pesos óptimos para el portafolio ({objetivo}):")
        st.json(pesos)
        st.write("Rendimiento del portafolio:")
        st.write(f"Retorno esperado: {rendimiento[0]:.2%}, Riesgo: {rendimiento[1]:.2%}, Sharpe Ratio: {rendimiento[2]:.2f}")

    # Paso 5: Visualización
    st.write("### Visualización de Datos")
    st.line_chart(datos)

# Función de backtesting
def backtesting_portafolio(data, pesos, start_test, end_test):
    rendimientos = data.pct_change().dropna()
    portafolio_retorno = (rendimientos * pesos).sum(axis=1)
    acumulado = (1 + portafolio_retorno).cumprod()
    rendimientos_anuales = portafolio_retorno.resample('Y').sum()

    # Filtrar datos de backtesting
    rendimientos_backtest = portafolio_retorno[start_test:end_test]
    acumulado_backtest = acumulado[start_test:end_test]
    
    # Métricas de backtesting
    metrics = {
        "Rendimiento anual promedio": rendimientos_backtest.mean() * 252,
        "Volatilidad anual": rendimientos_backtest.std() * np.sqrt(252),
        "Sharpe Ratio": (rendimientos_backtest.mean() * 252) / (rendimientos_backtest.std() * np.sqrt(252)),
        "Máximo Drawdown": (acumulado_backtest / acumulado_backtest.cummax() - 1).min()
    }
    return metrics, acumulado_backtest

# Paso 5: Backtesting
if "datos" in locals() and "pesos" in locals():
    st.write("### Backtesting de Portafolios")
    start_test = "2021-01-01"
    end_test = "2023-01-01"
    metrics, acumulado_backtest = backtesting_portafolio(datos, pesos, start_test, end_test)
    
    # Mostrar métricas de backtesting
    st.write("#### Métricas del Portafolio Optimizado")
    st.json(metrics)
    
    # Comparativa contra el S&P500 y portafolio equitativo
    st.write("#### Comparativa contra Benchmark")
    datos_benchmark = cargar_datos(["SPY"], start_test, end_test)
    rendimientos_benchmark = datos_benchmark.pct_change().dropna()
    acumulado_benchmark = (1 + rendimientos_benchmark).cumprod()

    pesos_equitativos = np.ones(len(tickers_list)) / len(tickers_list)
    _, acumulado_equitativo = backtesting_portafolio(datos, pesos_equitativos, start_test, end_test)

    # Gráfica comparativa
    st.line_chart(pd.DataFrame({
        "Portafolio Optimizado": acumulado_backtest,
        "S&P 500": acumulado_benchmark["SPY"],
        "Portafolio Equitativo": acumulado_equitativo
    }))

# Paso 6: Visualización adicional
st.write("### Visualización adicional")
st.write("#### Heatmap de Correlaciones")
correlaciones = datos.pct_change().corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlaciones, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Modelo de Black-Litterman
# Función para obtener pesos del benchmark (basado en capitalización de mercado)
def calcular_priori_benchmark(data, benchmark_ticker):
    benchmark_data = cargar_datos([benchmark_ticker], "2010-01-01", "2023-01-01")
    rendimientos_benchmark = benchmark_data.pct_change().dropna()
    
    # Pesos del benchmark proporcional a la correlación con el benchmark
    rendimientos_activos = data.pct_change().dropna()
    correlaciones = rendimientos_activos.corrwith(rendimientos_benchmark[benchmark_ticker])
    pesos_benchmark = correlaciones / correlaciones.sum()  # Normalizar a 100%
    return pesos_benchmark

# Paso 7: Modelo de Black-Litterman con pesos del benchmark
if "datos" in locals():
    st.write("### Modelo de Black-Litterman con Benchmark")
    
    # Selección del benchmark
    benchmark_ticker = st.text_input("Introduce el símbolo del índice de referencia (ejemplo: SPY para S&P 500):", value="SPY")
    
    if st.button("Calcular Pesos del Benchmark"):
        try:
            # Calcular pesos del benchmark
            pesos_benchmark = calcular_priori_benchmark(datos, benchmark_ticker)
            st.write("Pesos iniciales basados en el benchmark:")
            st.json(pesos_benchmark.to_dict())
            
            # Distribución a priori basada en el benchmark
            mu, S, _ = calcular_priori(datos, tickers_list)
            st.write("Rendimientos esperados a priori:", mu)
            
            # Entrada de Views
            st.write("#### Perspectivas del Usuario (Views)")
            st.write("Ingresa tus perspectivas sobre los rendimientos esperados de los ETFs seleccionados.")
            views = {}
            for ticker in tickers_list:
                views[ticker] = st.number_input(f"Rendimiento esperado para {ticker} (%):", value=0.0, step=0.1) / 100
            
            # Entrada de qué tanta confianza (confidencias) tienes
            st.write("#### Confianza en las Perspectivas")
            st.write("Especifica tu confianza en cada perspectiva (valores bajos indican alta incertidumbre).")
            confidencias = []
            for ticker in tickers_list:
                omega = st.number_input(f"Confianza para {ticker}:", value=0.1, step=0.1)
                confidencias.append(omega)
            
            # Calcular pesos óptimos con Black-Litterman
            if st.button("Calcular Portafolio Óptimo con Black-Litterman"):
                pesos_bl, rendimiento_bl, riesgo_bl, sharpe_bl = aplicar_black_litterman(mu, S, pesos_benchmark, views, confidencias)
                
                # Mostrar resultados
                st.write("#### Pesos Óptimos con Black-Litterman")
                st.json(pesos_bl)
                st.write(f"Rendimiento esperado: {rendimiento_bl:.2%}")
                st.write(f"Riesgo esperado: {riesgo_bl:.2%}")
                st.write(f"Sharpe Ratio: {sharpe_bl:.2f}")
        
        except Exception as e:
            st.error(f"Error al calcular pesos del benchmark: {e}")

# Validación de datos del benchmark
def validar_datos_benchmark(data, benchmark_data):
    if data.index.equals(benchmark_data.index):
        return True
    # Alinear fechas si no coinciden
    benchmark_data = benchmark_data.reindex(data.index, method="ffill").dropna()
    return not benchmark_data.empty, benchmark_data

# Visualización de los pesos
def graficar_pesos(pesos_iniciales, pesos_ajustados, labels):
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(labels))
    
    ax.bar(index, pesos_iniciales, bar_width, label="Pesos Iniciales (Benchmark)", alpha=0.7)
    ax.bar(index + bar_width, pesos_ajustados, bar_width, label="Pesos Ajustados (Black-Litterman)", alpha=0.7)
    
    ax.set_xlabel("Activos")
    ax.set_ylabel("Pesos")
    ax.set_title("Comparativa de Pesos del Portafolio")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    
    return fig

# Incorporar fallback a pesos equitativos
if "datos" in locals():
    st.write("### Modelo de Black-Litterman Mejorado")
    
    # Selección del benchmark
    benchmark_ticker = st.text_input("Introduce el símbolo del índice de referencia (ejemplo: SPY para S&P 500):", value="SPY")
    
    if st.button("Calcular Pesos del Benchmark"):
        try:
            # Descargar y validar datos del benchmark
            benchmark_data = cargar_datos([benchmark_ticker], "2010-01-01", "2023-01-01")
            valido, benchmark_data = validar_datos_benchmark(datos, benchmark_data)
            
            if not valido:
                st.warning("No se pudieron validar los datos del benchmark. Usando pesos equitativos.")
                pesos_benchmark = np.ones(len(tickers_list)) / len(tickers_list)
            else:
                # Calcular pesos del benchmark
                pesos_benchmark = calcular_priori_benchmark(datos, benchmark_ticker)
            
            st.write("Pesos iniciales calculados:")
            st.json(pesos_benchmark.to_dict())
            
            # Distribución a priori basada en el benchmark o equitativos
            mu, S, _ = calcular_priori(datos, tickers_list)
            
            # Entrada de Views
            st.write("#### Perspectivas del Usuario (Views)")
            st.write("Ingresa tus perspectivas sobre los rendimientos esperados de los ETFs seleccionados.")
            views = {}
            for ticker in tickers_list:
                views[ticker] = st.number_input(f"Rendimiento esperado para {ticker} (%):", value=0.0, step=0.1) / 100
            
            # Entrada de Confidencias
            st.write("#### Confianza en las Perspectivas")
            confidencias = []
            for ticker in tickers_list:
                omega = st.number_input(f"Confianza para {ticker}:", value=0.1, step=0.1)
                confidencias.append(omega)
            
            # Calcular pesos óptimos con Black-Litterman
            if st.button("Calcular Portafolio Óptimo con Black-Litterman"):
                pesos_bl, rendimiento_bl, riesgo_bl, sharpe_bl = aplicar_black_litterman(mu, S, pesos_benchmark, views, confidencias)
                
                # Mostrar resultados
                st.write("#### Pesos Óptimos con Black-Litterman")
                st.json(pesos_bl)
                st.write(f"Rendimiento esperado: {rendimiento_bl:.2%}")
                st.write(f"Riesgo esperado: {riesgo_bl:.2%}")
                st.write(f"Sharpe Ratio: {sharpe_bl:.2f}")
                
                # Visualización comparativa
                st.write("#### Comparación de Pesos")
                labels = tickers_list
                fig = graficar_pesos(list(pesos_benchmark.values()), list(pesos_bl.values()), labels)
                st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error al calcular pesos del benchmark: {e}")

# Función para calcular rendimiento acumulado y métricas
def calcular_rendimiento_acumulado(data, pesos, start_date, end_date):
    rendimientos = data.pct_change().dropna()
    portafolio_retorno = (rendimientos * pesos).sum(axis=1)
    acumulado = (1 + portafolio_retorno).cumprod()
    acumulado = acumulado[start_date:end_date]
    return acumulado

# Función para graficar comparativas de rendimiento
def graficar_comparativa_rendimiento(rendimientos_dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, rend in rendimientos_dict.items():
        ax.plot(rend, label=label)
    ax.set_title("Comparativa de Rendimientos Acumulados")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Rendimiento Acumulado")
    ax.legend()
    ax.grid(True)
    return fig

if "datos" in locals() and "pesos_bl" in locals():
    st.write("### Comparativa de Rendimientos entre Portafolios")
    
    # Fechas de backtesting
    start_test = "2021-01-01"
    end_test = "2023-01-01"
    
    # Calcular rendimientos acumulados
    st.write("#### Calculando rendimientos acumulados...")
    try:
        # Portafolio ajustado (Black-Litterman)
        acumulado_bl = calcular_rendimiento_acumulado(datos, list(pesos_bl.values()), start_test, end_test)
        
        # Portafolio inicial (Benchmark)
        acumulado_benchmark = calcular_rendimiento_acumulado(datos, list(pesos_benchmark.values()), start_test, end_test)
        
        # Benchmark adicional (S&P 500)
        sp500_data = cargar_datos(["SPY"], start_test, end_test)
        rendimientos_sp500 = sp500_data.pct_change().dropna()
        acumulado_sp500 = (1 + rendimientos_sp500["SPY"]).cumprod()
        
        # Graficar comparativa
        rendimientos_dict = {
            "Portafolio Ajustado (Black-Litterman)": acumulado_bl,
            "Portafolio Inicial (Benchmark)": acumulado_benchmark,
            "S&P 500": acumulado_sp500
        }
        fig = graficar_comparativa_rendimiento(rendimientos_dict)
        st.pyplot(fig)
        
        # Métricas de comparación
        st.write("#### Métricas Clave")
        for nombre, rend in rendimientos_dict.items():
            rendimiento_anual = rend.pct_change().mean() * 252
            volatilidad_anual = rend.pct_change().std() * np.sqrt(252)
            sharpe_ratio = rendimiento_anual / volatilidad_anual
            st.write(f"**{nombre}:**")
            st.write(f"- Rendimiento anual promedio: {rendimiento_anual:.2%}")
            st.write(f"- Volatilidad anual: {volatilidad_anual:.2%}")
            st.write(f"- Sharpe Ratio: {sharpe_ratio:.2f}")
            st.write("---")
    
    except Exception as e:
        st.error(f"Error al calcular comparativa de rendimientos: {e}")

import plotly.graph_objects as go

# Función para graficar comparativa de rendimiento con Plotly
def graficar_comparativa_rendimiento_plotly(rendimientos_dict):
    fig = go.Figure()
    
    # Añadir líneas para cada portafolio
    for label, rend in rendimientos_dict.items():
        fig.add_trace(go.Scatter(x=rend.index, y=rend, mode='lines', name=label))
    
    # Personalizar la gráfica
    fig.update_layout(
        title="Comparativa de Rendimientos Acumulados",
        xaxis_title="Fecha",
        yaxis_title="Rendimiento Acumulado",
        template="plotly_dark",
        hovermode="closest",
        xaxis_rangeslider_visible=True
    )
    
    return fig

if "datos" in locals() and "pesos_bl" in locals():
    st.write("### Comparativa Interactiva de Rendimientos entre Portafolios")
    
    # Fechas de backtesting
    start_test = "2021-01-01"
    end_test = "2023-01-01"
    
    # Calcular rendimientos acumulados
    st.write("#### Calculando rendimientos acumulados...")
    try:
        # Portafolio ajustado (Black-Litterman)
        acumulado_bl = calcular_rendimiento_acumulado(datos, list(pesos_bl.values()), start_test, end_test)
        
        # Portafolio inicial (Benchmark)
        acumulado_benchmark = calcular_rendimiento_acumulado(datos, list(pesos_benchmark.values()), start_test, end_test)
        
        # Benchmark adicional (S&P 500)
        sp500_data = cargar_datos(["SPY"], start_test, end_test)
        rendimientos_sp500 = sp500_data.pct_change().dropna()
        acumulado_sp500 = (1 + rendimientos_sp500["SPY"]).cumprod()
        
        # Graficar comparativa interactiva
        rendimientos_dict = {
            "Portafolio Ajustado (Black-Litterman)": acumulado_bl,
            "Portafolio Inicial (Benchmark)": acumulado_benchmark,
            "S&P 500": acumulado_sp500
        }
        fig = graficar_comparativa_rendimiento_plotly(rendimientos_dict)
        st.plotly_chart(fig)
        
        # Métricas de comparación
        st.write("#### Métricas Clave")
        for nombre, rend in rendimientos_dict.items():
            rendimiento_anual = rend.pct_change().mean() * 252
            volatilidad_anual = rend.pct_change().std() * np.sqrt(252)
            sharpe_ratio = rendimiento_anual / volatilidad_anual
            st.write(f"**{nombre}:**")
            st.write(f"- Rendimiento anual promedio: {rendimiento_anual:.2%}")
            st.write(f"- Volatilidad anual: {volatilidad_anual:.2%}")
            st.write(f"- Sharpe Ratio: {sharpe_ratio:.2f}")
            st.write("---")
    
    except Exception as e:
        st.error(f"Error al calcular comparativa de rendimientos: {e}")








