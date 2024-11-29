import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

# Configuración inicial de la página
st.set_page_config(page_title="Gestión de Portafolios Óptimos",
                   page_icon="📈",
                   layout="wide")

# Título y descripción
st.markdown("""
# 📊 **Gestión de Portafolios y Optimización**
Una herramienta interactiva para analizar activos, calcular métricas de riesgo y construir portafolios óptimos utilizando modelos avanzados como Black-Litterman.
""")

# 1. Selección de los 5 ETFs iniciales
DEFAULT_ETFS = ["SPY", "EEM", "TLT", "HYG", "GLD"]
tickers = st.sidebar.multiselect("Selecciona los 5 ETFs", options=DEFAULT_ETFS, default=DEFAULT_ETFS)

# Variables globales
data = {}

# Funciones auxiliares
def calcular_estadisticas(precios):
    rendimientos = precios.pct_change().dropna()
    media = rendimientos.mean()
    volatilidad = rendimientos.std()
    sesgo = skew(rendimientos)
    curtosis = kurtosis(rendimientos, fisher=False)
    VaR = rendimientos.quantile(0.05)
    CVaR = rendimientos[rendimientos <= VaR].mean()
    sharpe = media / volatilidad
    sortino = media / rendimientos[rendimientos < 0].std()
    drawdown = (precios / precios.cummax() - 1).min()
    
    return {
        "Media (%)": media * 100,
        "Volatilidad (%)": volatilidad * 100,
        "Sesgo": sesgo,
        "Curtosis": curtosis,
        "VaR 5% (%)": VaR * 100,
        "CVaR (%)": CVaR * 100,
        "Sharpe Ratio": sharpe,
        "Sortino": sortino,
        "Drawdown": drawdown
    }

def optimizar_portafolio(rendimientos, objetivo="min_vol", retorno_objetivo=0.1):
    n_activos = rendimientos.shape[1]
    pesos_iniciales = np.ones(n_activos) / n_activos
    limites = [(0, 1) for _ in range(n_activos)]
    restricciones = [{'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1}]
    
    if objetivo == "min_vol":
        resultado = minimize(lambda pesos: calcular_volatilidad_pesos(pesos, rendimientos),
                             pesos_iniciales, method='SLSQP', bounds=limites, constraints=restricciones)
    elif objetivo == "max_sharpe":
        resultado = minimize(lambda pesos: -calcular_rendimiento_pesos(pesos, rendimientos) /
                                        calcular_volatilidad_pesos(pesos, rendimientos),
                             pesos_iniciales, method='SLSQP', bounds=limites, constraints=restricciones)
    elif objetivo == "min_vol_target":
        restricciones.append({'type': 'eq', 'fun': lambda pesos: calcular_rendimiento_pesos(pesos, rendimientos) - retorno_objetivo})
        resultado = minimize(lambda pesos: calcular_volatilidad_pesos(pesos, rendimientos),
                             pesos_iniciales, method='SLSQP', bounds=limites, constraints=restricciones)
    else:
        raise ValueError("Objetivo no reconocido.")
    
    return resultado.x

def calcular_rendimiento_pesos(pesos, rendimientos):
    return np.sum(pesos * rendimientos.mean())

def calcular_volatilidad_pesos(pesos, rendimientos):
    cov_matrix = rendimientos.cov()
    return np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))

# Funciones del modelo Black-Litterman
def calcular_prior(rendimientos, market_cap, tau):
    # Asumimos un benchmark de asignación equitativa
    w_mkt = np.ones(len(rendimientos.columns)) / len(rendimientos.columns)  # Pesos equitativos
    cov = rendimientos.cov() * 252  # Matriz de covarianza anualizada
    pi = tau * np.dot(cov, w_mkt)  # Rendimientos implícitos
    return pi, cov

def integrar_views(pi, cov, views, tau, varianza_views):
    P = []  # Matriz de views
    Q = []  # Rendimientos esperados según views
    omega = []  # Matriz de incertidumbre (diagonal)

    for view in views:
        activo, valor = view
        idx = tickers_list.index(activo)
        P_row = np.zeros(len(tickers_list))
        P_row[idx] = 1  # Asumimos views absolutas
        P.append(P_row)
        Q.append(float(valor) / 100)
        omega.append(varianza_views)
    
    P = np.array(P)
    Q = np.array(Q)
    omega = np.diag(omega)

    # Black-Litterman posterior
    inv_cov = np.linalg.inv(tau * cov)
    inv_omega = np.linalg.inv(omega)
    posterior_mean = np.linalg.inv(inv_cov + np.dot(P.T, np.dot(inv_omega, P))).dot(
        inv_cov.dot(pi) + np.dot(P.T, np.dot(inv_omega, Q))
    )
    posterior_cov = np.linalg.inv(inv_cov + np.dot(P.T, np.dot(inv_omega, P)))

    return posterior_mean, posterior_cov

def calcular_pesos_optimos(mean_returns, cov):
    inv_cov = np.linalg.inv(cov)
    pesos = np.dot(inv_cov, mean_returns) / np.sum(np.dot(inv_cov, mean_returns))
    return pesos

# 1. Selección de los 5 ETFs
with st.sidebar:
    st.header("Selección de ETFs")
    tickers_list = tickers  # Usar los ETFs seleccionados por el usuario
    fecha_inicio = st.date_input("Fecha de inicio", value=pd.to_datetime("2010-01-01"))
    fecha_fin = st.date_input("Fecha de fin", value=pd.to_datetime("today").date())

    if st.button("Cargar datos"):
        data = {ticker: yf.download(ticker, start=fecha_inicio, end=fecha_fin)['Adj Close'] for ticker in tickers_list}
        st.success("Datos cargados exitosamente.")
        st.dataframe(pd.DataFrame(data))

# 2. Definir los Views del Modelo Black-Litterman
views = [
    {"activo": "LQD", "rendimiento_esperado": 0.03},  # 3% para LQD
    {"activo": "LEMB", "rendimiento_esperado": 0.06},  # 6% para LEMB
    {"activo": "VTI", "rendimiento_esperado": 0.07},   # 7% para VTI
    {"activo": "EEM", "rendimiento_esperado": 0.09},   # 9% para EEM
    {"activo": "GLD", "rendimiento_esperado": 0.05},   # 5% para GLD
]

# 3. Pestaña: Análisis de Activos
with st.tabs("📈 Análisis de Activos")[0]:
    st.header("📈 Análisis de Activos")
    if data:
        for ticker in tickers_list:
            st.subheader(f"Análisis de {ticker}")
            fig = px.line(data[ticker], title=f"Serie de Tiempo - {ticker}")
            st.plotly_chart(fig)

# 4. Pestaña: Optimización Black-Litterman
with st.tabs("🧠 Black-Litterman")[0]:
    st.header("🧠 Modelo Black-Litterman")
    if data:
        tau = st.slider("Tau (confianza en el mercado)", 0.01, 1.0, 0.05)
        varianza_views = st.slider("Varianza de los views (incertidumbre)", 0.01, 1.0, 0.25)

        # Aplicar el Modelo Black-Litterman
        pi, cov = calcular_prior(data, market_cap=10000, tau=tau)
        posterior_mean, posterior_cov = integrar_views(pi, cov, views, tau, varianza_views)
        pesos_optimos = calcular_pesos_optimos(posterior_mean, posterior_cov)

        st.write("Pesos ajustados Black-Litterman:", pesos_optimos)

# 5. Pestaña: Comparación de Resultados
with st.tabs("📊 Visualización de Resultados")[0]:
    st.header("📊 Visualización de Resultados")
    if data:
        # 1. Comparación de Rendimientos de ETFs vs Benchmark
        rendimiento_etfs = {
            'Portafolio Black-Litterman': rendimiento_portafolio,
            'Benchmark (Asignación Equitativa)': rendimiento_benchmark,
        }
        
        for ticker in tickers_list:
            rendimiento_etfs[ticker] = calcular_rendimiento_acumulado(np.ones(len(tickers_list)) / len(tickers_list), rendimientos)

        # Gráfico de barras de comparación de rendimientos
        fig_comparacion = px.bar(
            x=list(rendimiento_etfs.keys()),
            y=list(rendimiento_etfs.values()),
            labels={'x': 'Portafolios y ETFs', 'y': 'Rendimiento Acumulado (%)'},
            title="Comparación del Rendimiento: Portafolio vs Benchmark vs ETFs"
        )

        fig_comparacion.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_comparacion)

        # 2. Matriz de Correlación entre los Rendimientos
        correlacion = rendimientos.corr()  # Matriz de correlación
        fig_correlacion = go.Figure(data=go.Heatmap(
            z=correlacion.values,
            x=tickers_list,
            y=tickers_list,
            colorscale='Viridis',
            colorbar=dict(title="Correlación")
        ))

        fig_correlacion.update_layout(title="Matriz de Correlación entre los Rendimientos de los ETFs")
        st.plotly_chart(fig_correlacion)

        # 3. Histograma de Rendimientos Diarios para cada ETF
        for ticker in tickers_list:
            rend = data[ticker].pct_change().dropna()
            fig_hist = px.histogram(rend, nbins=30, title=f"Distribución de Rendimientos Diarios - {ticker}", 
                                    labels={'value': 'Rendimiento Diario'}, template="plotly_dark")
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist)

        # 4. Gráfico de la "Efficient Frontier"
        # Calculamos los rendimientos y la volatilidad de diferentes combinaciones de portafolios
        num_portafolios = 1000
        resultados = np.zeros((3, num_portafolios))
        for i in range(num_portafolios):
            pesos = np.random.random(len(tickers_list))
            pesos /= np.sum(pesos)
            resultados[0, i] = calcular_rendimiento_pesos(pesos, rendimientos)
            resultados[1, i] = calcular_volatilidad_pesos(pesos, rendimientos)
            resultados[2, i] = resultados[0, i] / resultados[1, i]  # Sharpe Ratio

        # Graficamos la frontera eficiente
        fig_efficient_frontier = go.Figure()
        fig_efficient_frontier.add_trace(go.Scatter(x=resultados[1], y=resultados[0], mode='markers', 
                                                   marker=dict(color=resultados[2], colorscale='Viridis', size=8, opacity=0.7),
                                                   name='Portafolios Aleatorios'))
        fig_efficient_frontier.update_layout(title="Frontera Eficiente", 
                                             xaxis_title="Volatilidad (Riesgo)", 
                                             yaxis_title="Rendimiento Esperado")
        st.plotly_chart(fig_efficient_frontier)

        # 5. Visualización de los Pesos del Portafolio Black-Litterman
        fig_pesos = px.pie(names=tickers_list, values=pesos_optimos, title="Distribución de Pesos del Portafolio Black-Litterman")
        fig_pesos.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pesos)





