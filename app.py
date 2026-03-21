import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="AI 참모 v4.0 (지능형 엔진)", page_icon="🧠", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    html, body, div, p, h1, h2, h3, h4 { font-family: 'Nanum Gothic', sans-serif !important; }
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
        st.error(f"⚠️ 데이터 로딩 실패: {e}")
        return None

# 3. XGBoost 예측
def predict_next_price(df):
    df_train = df.copy()
    df_train['target'] = df_train['Close'].shift(-1)
    df_train = df_train.dropna()
    X = np.array(range(len(df_train))).reshape(-1, 1)
    y = df_train['Close']
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X, y)
    return model.predict(np.array([[len(df)]]))[0]

# 4. 제미나이 지능형 엔진 (사용 가능한 모델 자동 검색)
def get_ai_briefing(df, pred, tf_name):
    if "GEMINI_API_KEY" not in st.secrets:
        return "❌ Secrets에 'GEMINI_API_KEY'가 없습니다."
        
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # [핵심] 현재 API 키로 사용 가능한 모든 모델 목록 가져오기
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if not available_models:
            return "❌ 사용 가능한 Gemini 모델이 없습니다. API 키 권한을 확인해 주세요."
            
        # 가장 최신 모델(보통 목록의 앞이나 뒤) 또는 Gemini 3/2 시리즈 우선 선택
        target_model = ""
        for m in ["models/gemini-3-flash", "models/gemini-2.0-flash", "models/gemini-1.5-flash"]:
            if m in available_models:
                target_model = m
                break
        
        if not target_model:
            target_model = available_models[0] # 없으면 그냥 첫 번째 모델 사용
            
        model = genai.GenerativeModel(target_model)
        latest = df.iloc[-1]
        prompt = f"비트코인 {tf_name} 기준 현재 {latest['Close']:.1f}$, AI예측 {pred:.1f}$, RSI {latest['RSI']:.1f}. 친절하게 3줄 전략 짜줘."
        
        response = model.generate_content(prompt)
        return f"({target_model} 분석 중)\n\n" + response.text
        
    except Exception as e:
        return f"❌ [지능형 엔진 오류] {str(e)}"

# 5. UI 구성
st.title("🧠 AI 비트코인 참모 v4.0")
st.subheader(f"🕒 현재 시간(KST): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
    
    st.info(f"💬 **AI 참모 실시간 브리핑**\n\n{get_ai_briefing(df, pred, st.session_state.tf_name)}")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="BTC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#ff00ff')), row=2, col=1)
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)
