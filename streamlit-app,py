"""
QUANT DASHBOARD - Análisis de Distribuciones y Fat Tails
=========================================================

Dashboard interactivo para análisis estadístico de activos financieros
Uso: streamlit run quant_dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime
import requests
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="Quant Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE DATOS
# ============================================================================

@st.cache_data(ttl=300)
def obtener_datos_binance(symbol, interval, limit):
    """Obtiene datos de Binance"""
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].astype(float)
        df.set_index('timestamp', inplace=True)
        return df[['close']]
    except Exception as e:
        st.error(f"Error: {e}")
        return None

@st.cache_data(ttl=300)
def obtener_datos_yahoo(ticker, periodo):
    """Obtiene datos de Yahoo Finance"""
    try:
        df = yf.Ticker(ticker).history(period=periodo)
        if df.empty:
            raise ValueError("Sin datos")
        return df[['Close']].rename(columns={'Close': 'close'})
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def calcular_retornos(df):
    """Calcula retornos logarítmicos"""
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    return df.dropna()

def analizar_distribucion(returns):
    """Análisis estadístico"""
    media = returns.mean()
    std = returns.std()
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    _, p_value = stats.normaltest(returns)
    
    return {
        'N': len(returns),
        'Media (%)': media * 100,
        'Std (%)': std * 100,
        'Vol Anual (%)': std * np.sqrt(252) * 100,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'P-value': p_value,
        'Normal': 'Sí' if p_value >= 0.05 else 'No'
    }, media, std

def analizar_fat_tails(returns, mean, std, sigma=2):
    """Análisis de fat tails"""
    umbral_sup = mean + sigma * std
    umbral_inf = mean - sigma * std
    
    eventos_pos = returns[returns > umbral_sup]
    eventos_neg = returns[returns < umbral_inf]
    
    pct_esperado = stats.norm.sf(sigma) * 100
    pct_real_pos = (len(eventos_pos) / len(returns)) * 100
    pct_real_neg = (len(eventos_neg) / len(returns)) * 100
    
    return {
        f'Eventos +{sigma}σ': len(eventos_pos),
        '% Cola +': pct_real_pos,
        f'Eventos -{sigma}σ': len(eventos_neg),
        '% Cola -': pct_real_neg,
        '% Esperado': pct_esperado,
        'Factor +': pct_real_pos / pct_esperado if pct_esperado > 0 else 0,
        'Factor -': pct_real_neg / pct_esperado if pct_esperado > 0 else 0,
        'Peor (%)': returns.min() * 100,
        'Mejor (%)': returns.max() * 100
    }

def plot_distribucion(returns, media, std, nombre, color):
    """Gráfico de distribución"""
    fig = go.Figure()
    
    hist_data = returns * 100
    
    fig.add_trace(go.Histogram(
        x=hist_data,
        histnorm='probability density',
        name='Datos Reales',
        marker_color=color,
        opacity=0.7,
        nbinsx=50
    ))
    
    x = np.linspace(returns.min(), returns.max(), 200) * 100
    gaussian = stats.norm.pdf(x/100, media, std) / 100
    
    fig.add_trace(go.Scatter(
        x=x,
        y=gaussian,
        mode='lines',
        name='Normal',
        line=dict(color='red', width=3)
    ))
    
    # Líneas ±2σ
    fig.add_vline(x=(media + 2*std)*100, line_dash="dash", 
                  line_color="yellow", annotation_text="+2σ")
    fig.add_vline(x=(media - 2*std)*100, line_dash="dash", 
                  line_color="yellow", annotation_text="-2σ")
    
    fig.update_layout(
        title=f'{nombre} - Distribución de Retornos',
        xaxis_title='Retornos Diarios (%)',
        yaxis_title='Densidad',
        template='plotly_dark',
        height=400
    )
    
    return fig

def plot_qq(returns, nombre):
    """Q-Q Plot"""
    fig = go.Figure()
    
    (osm, osr), (slope, intercept, r) = stats.probplot(returns, dist="norm")
    
    fig.add_trace(go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        name='Datos',
        marker=dict(color='lightblue', size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=osm,
        y=slope * osm + intercept,
        mode='lines',
        name='Línea teórica',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f'{nombre} - Q-Q Plot',
        xaxis_title='Cuantiles Teóricos',
        yaxis_title='Cuantiles Observados',
        template='plotly_dark',
        height=400
    )
    
    return fig

# ============================================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================================

st.sidebar.title("⚙️ Configuración")

# Activos predefinidos
ACTIVOS_CRYPTO = {
    'BTC/USDT': 'BTCUSDT',
    'ETH/USDT': 'ETHUSDT',
    'BNB/USDT': 'BNBUSDT',
    'SOL/USDT': 'SOLUSDT',
}

ACTIVOS_TRADICIONALES = {
    'S&P 500': '^GSPC',
    'Nasdaq 100': '^NDX',
    'Dow Jones': '^DJI',
    'Russell 2000': '^RUT',
    'VIX': '^VIX',
    'Gold': 'GC=F',
    'Crude Oil': 'CL=F',
}

ACTIVOS_FUTURES = {
    'E-mini Nasdaq (NQ)': 'NQ=F',
    'E-mini S&P (ES)': 'ES=F',
    'E-mini Dow (YM)': 'YM=F',
}

st.sidebar.subheader("📈 Activo 1")
tipo_activo_1 = st.sidebar.selectbox(
    "Tipo",
    ['Crypto', 'Índices', 'Futuros'],
    key='tipo1'
)

if tipo_activo_1 == 'Crypto':
    activo_1_nombre = st.sidebar.selectbox("Seleccionar", list(ACTIVOS_CRYPTO.keys()), key='a1')
    activo_1 = {'tipo': 'binance', 'symbol': ACTIVOS_CRYPTO[activo_1_nombre]}
elif tipo_activo_1 == 'Índices':
    activo_1_nombre = st.sidebar.selectbox("Seleccionar", list(ACTIVOS_TRADICIONALES.keys()), key='a1')
    activo_1 = {'tipo': 'yahoo', 'ticker': ACTIVOS_TRADICIONALES[activo_1_nombre]}
else:
    activo_1_nombre = st.sidebar.selectbox("Seleccionar", list(ACTIVOS_FUTURES.keys()), key='a1')
    activo_1 = {'tipo': 'yahoo', 'ticker': ACTIVOS_FUTURES[activo_1_nombre]}

st.sidebar.subheader("📉 Activo 2 (Comparación)")
comparar = st.sidebar.checkbox("Comparar con otro activo")

if comparar:
    tipo_activo_2 = st.sidebar.selectbox(
        "Tipo",
        ['Crypto', 'Índices', 'Futuros'],
        key='tipo2'
    )
    
    if tipo_activo_2 == 'Crypto':
        activo_2_nombre = st.sidebar.selectbox("Seleccionar", list(ACTIVOS_CRYPTO.keys()), key='a2')
        activo_2 = {'tipo': 'binance', 'symbol': ACTIVOS_CRYPTO[activo_2_nombre]}
    elif tipo_activo_2 == 'Índices':
        activo_2_nombre = st.sidebar.selectbox("Seleccionar", list(ACTIVOS_TRADICIONALES.keys()), key='a2')
        activo_2 = {'tipo': 'yahoo', 'ticker': ACTIVOS_TRADICIONALES[activo_2_nombre]}
    else:
        activo_2_nombre = st.sidebar.selectbox("Seleccionar", list(ACTIVOS_FUTURES.keys()), key='a2')
        activo_2 = {'tipo': 'yahoo', 'ticker': ACTIVOS_FUTURES[activo_2_nombre]}

st.sidebar.subheader("📅 Periodo")
periodo_map = {
    '1 mes': ('1mo', 30),
    '3 meses': ('3mo', 90),
    '6 meses': ('6mo', 180),
    '1 año': ('1y', 365),
    '2 años': ('2y', 730)
}
periodo_sel = st.sidebar.select_slider(
    "Seleccionar periodo",
    options=list(periodo_map.keys()),
    value='6 meses'
)
periodo_yahoo, periodo_binance = periodo_map[periodo_sel]

st.sidebar.subheader("🎯 Análisis Fat Tails")
umbral_sigma = st.sidebar.slider(
    "Umbral (σ)",
    min_value=1.0,
    max_value=3.0,
    value=2.0,
    step=0.5
)

# ============================================================================
# HEADER
# ============================================================================

st.title("📊 QUANT DASHBOARD")
st.markdown("### Análisis de Distribuciones y Fat Tails")
st.markdown("---")

# ============================================================================
# OBTENER Y PROCESAR DATOS
# ============================================================================

with st.spinner('Descargando datos...'):
    # Activo 1
    if activo_1['tipo'] == 'binance':
        df1 = obtener_datos_binance(activo_1['symbol'], '1d', periodo_binance)
    else:
        df1 = obtener_datos_yahoo(activo_1['ticker'], periodo_yahoo)
    
    if df1 is not None:
        df1 = calcular_retornos(df1)
        returns_1 = df1['returns'].values
        stats_1, media_1, std_1 = analizar_distribucion(returns_1)
        fat_tails_1 = analizar_fat_tails(returns_1, media_1, std_1, umbral_sigma)
    
    # Activo 2
    if comparar:
        if activo_2['tipo'] == 'binance':
            df2 = obtener_datos_binance(activo_2['symbol'], '1d', periodo_binance)
        else:
            df2 = obtener_datos_yahoo(activo_2['ticker'], periodo_yahoo)
        
        if df2 is not None:
            df2 = calcular_retornos(df2)
            returns_2 = df2['returns'].values
            stats_2, media_2, std_2 = analizar_distribucion(returns_2)
            fat_tails_2 = analizar_fat_tails(returns_2, media_2, std_2, umbral_sigma)

# ============================================================================
# VISUALIZACIONES
# ============================================================================

if df1 is not None:
    # Métricas principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Observaciones", stats_1['N'])
    with col2:
        st.metric("📈 Media Diaria", f"{stats_1['Media (%)']:.3f}%")
    with col3:
        st.metric("📉 Volatilidad", f"{stats_1['Std (%)']:.3f}%")
    with col4:
        st.metric("🎲 Curtosis", f"{stats_1['Kurtosis']:.2f}")
    with col5:
        fat_factor = fat_tails_1['Factor +']
        st.metric("🔥 Factor Fat Tail", f"{fat_factor:.2f}x")
    
    st.markdown("---")
    
    # Gráficos principales
    if comparar and df2 is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 {activo_1_nombre}")
            fig1 = plot_distribucion(returns_1, media_1, std_1, activo_1_nombre, '#3b82f6')
            st.plotly_chart(fig1, use_container_width=True)
            
            fig_qq1 = plot_qq(returns_1, activo_1_nombre)
            st.plotly_chart(fig_qq1, use_container_width=True)
        
        with col2:
            st.subheader(f"📊 {activo_2_nombre}")
            fig2 = plot_distribucion(returns_2, media_2, std_2, activo_2_nombre, '#f59e0b')
            st.plotly_chart(fig2, use_container_width=True)
            
            fig_qq2 = plot_qq(returns_2, activo_2_nombre)
            st.plotly_chart(fig_qq2, use_container_width=True)
    else:
        st.subheader(f"📊 {activo_1_nombre}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = plot_distribucion(returns_1, media_1, std_1, activo_1_nombre, '#3b82f6')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig_qq1 = plot_qq(returns_1, activo_1_nombre)
            st.plotly_chart(fig_qq1, use_container_width=True)
    
    st.markdown("---")
    
    # Estadísticas detalladas
    st.subheader("📋 Estadísticas Detalladas")
    
    if comparar and df2 is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {activo_1_nombre}")
            stats_df1 = pd.DataFrame([stats_1]).T
            stats_df1.columns = ['Valor']
            st.dataframe(stats_df1, use_container_width=True)
            
            st.markdown("**Fat Tails Analysis**")
            ft_df1 = pd.DataFrame([fat_tails_1]).T
            ft_df1.columns = ['Valor']
            st.dataframe(ft_df1, use_container_width=True)
        
        with col2:
            st.markdown(f"#### {activo_2_nombre}")
            stats_df2 = pd.DataFrame([stats_2]).T
            stats_df2.columns = ['Valor']
            st.dataframe(stats_df2, use_container_width=True)
            
            st.markdown("**Fat Tails Analysis**")
            ft_df2 = pd.DataFrame([fat_tails_2]).T
            ft_df2.columns = ['Valor']
            st.dataframe(ft_df2, use_container_width=True)
        
        # Comparación directa
        st.markdown("---")
        st.subheader("⚖️ Comparación Directa")
        
        comp_data = {
            'Métrica': list(stats_1.keys()),
            activo_1_nombre: list(stats_1.values()),
            activo_2_nombre: list(stats_2.values())
        }
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
    else:
        stats_df1 = pd.DataFrame([stats_1]).T
        stats_df1.columns = ['Valor']
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(stats_df1, use_container_width=True)
        
        with col2:
            st.markdown("**Fat Tails Analysis**")
            ft_df1 = pd.DataFrame([fat_tails_1]).T
            ft_df1.columns = ['Valor']
            st.dataframe(ft_df1, use_container_width=True)
    
    # Interpretación
    st.markdown("---")
    st.subheader("💡 Interpretación")
    
    with st.expander("📖 Ver guía de interpretación"):
        st.markdown("""
        **Curtosis:**
        - Curtosis > 0: Distribución leptocúrtica (colas más gordas que normal)
        - Curtosis < 0: Distribución platicúrtica (colas más delgadas)
        - Curtosis = 0: Distribución normal
        
        **Fat Tails:**
        - Factor > 1: Más eventos extremos de lo esperado
        - Factor < 1: Menos eventos extremos
        - Factor = 1: Coincide con distribución normal
        
        **P-value (Normalidad):**
        - p < 0.05: Se rechaza normalidad
        - p ≥ 0.05: No se puede rechazar normalidad
        
        **Skewness:**
        - > 0: Asimetría positiva (cola derecha más larga)
        - < 0: Asimetría negativa (cola izquierda más larga)
        - = 0: Simétrica
        """)
    
    # Alertas
    if stats_1['Kurtosis'] > 3:
        st.warning(f"⚠️ **{activo_1_nombre}** muestra curtosis muy alta ({stats_1['Kurtosis']:.2f}), indicando FAT TAILS significativos. Los modelos que asumen normalidad subestimarán el riesgo.")
    
    if fat_tails_1['Factor +'] > 2:
        st.error(f"🔥 **{activo_1_nombre}** tiene {fat_tails_1['Factor +']:.1f}x más eventos extremos que lo esperado en distribución normal!")
    
    if comparar and df2 is not None:
        vol_ratio = stats_2['Std (%)'] / stats_1['Std (%)']
        st.info(f"📊 **{activo_2_nombre}** es **{vol_ratio:.2f}x** más volátil que **{activo_1_nombre}**")

else:
    st.error("❌ No se pudieron obtener datos. Verifica la conexión a internet.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Quant Dashboard v1.0 | Datos: Binance API + Yahoo Finance</p>
    <p>⚠️ Este dashboard es solo para análisis. No constituye asesoría financiera.</p>
</div>
""", unsafe_allow_html=True)
