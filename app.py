import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="AI 참모 v3.7 (마지막 승부)", page_icon="🧠", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    html, body, [data-testid="stSidebar"], p, h1, h2, h3, h4, div { font-family: 'Nanum Gothic', sans-serif !important; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; color: #ff00ff; }
</style>
""", unsafe_allow_html=True)

# 2. 데이터 가져오기 (Kraken)
@st.cache_data(ttl=900)
def get_analysis_data(tf):
    try:
        ex = ccxt.kraken({'enableRateLimit': True})
        ohlcv = ex.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=200)
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df.dropna()
    except Exception as e:
        st.error(f"⚠️ 거래소 연결 에러: {e}")
        return None

# 3. XGBoost 예측
def predict_next_price(df):
    df_train = df.copy()
    df_train['target'] = df_train['Close'].shift(-1)
    df_train = df_train.dropna()
    X = np.array(range(len(df_train))).reshape(-1, 1)
    y = df_train['Close']
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    return model.predict(np.array([[len(df)]]))[0]

# 4. 제미나이 브리핑 (멀티 엔드포인트 자동 시도)
def get_ai_briefing(df, pred, tf_name):
    if "GEMINI_API_KEY" not in st.secrets:
        return "❌ Secrets에 'GEMINI_API_KEY'를 넣어주세요."
    
    api_key = st.secrets["GEMINI_API_KEY"]
    latest = df.iloc[-1]
    prompt = f"비트코인 {tf_name} 기준, 현재 {latest['Close']:.1f}$, AI예측 {pred:.1f}$. 짧게 3줄 전략 짜줘."
    
    # [핵심] 시도해볼 모델 후보 리스트 (2026년 기준 최신순)
    models_to_try = [
        "v1beta/models/gemini-1.5-flash",
        "v1/models/gemini-1.5-flash",
        "v1beta/models/gemini-1.5-pro",
        "v1beta/models/gemini-pro"
    ]
    
    last_error = ""
    for model_path in models_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/{model_path}:generateContent?key={api_key}"
            res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10)
            
            if res.status_code == 200:
                return res.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                last_error = f"{model_path} 실패({res.status_code})"
                continue # 다음 모델로 시도
        except:
            continue
            
    return f"❌ 모든 모델 접속 실패. ({last_error}) API 키가 'Generative Language API' 권한이 있는지 확인이 필요합니다."

# 5. UI 구성
st.title("🧠 AI 비트코인 참모 v3.7")
st.subheader(f"🕒 현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if 'tf' not in st.session_state: st.session_state.tf, st.session_state.tf_name = "1h", "1시간"
c1, c2, c3 = st.columns(3)
if c1.button("1시간 분석", use_container_width=True): st.session_state.tf, st.session_state.tf_name = "1h", "1시간"
if c2.button("4시간 분석", use_container_width=True): st.session_state.tf, st.session_state.tf_name = "4h", "4시간"
if c3.button("1일 분석", use_container_width=True): st.session_state.tf, st.session_state.tf_name = "1d", "1일"

df = get_analysis_data(st.session_state.tf)
if df is not None:
    pred = predict_next_price(df)
    cur = df['Close'].iloc[-1]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("현재가", f"${cur:,.1f}")
    m2.metric(f"{st.session_state.tf_name} 후 예측", f"${pred:,.1f}", f"{pred-cur:+.1f}$")
    m3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    
    # 💬 브리핑 영역
    st.info(f"💬 **AI 참모 실시간 브리핑**\n\n{get_ai_briefing(df, pred, st.session_state.tf_name)}")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="BTC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#ff00ff')), row=2, col=1)
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)
