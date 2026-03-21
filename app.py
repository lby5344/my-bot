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
st.set_page_config(page_title="AI 참모 v3.9 (2026 최적화)", page_icon="🧠", layout="wide")
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

# 4. 제미나이 브리핑 (2026년형 자동 모델 매칭)
def get_ai_briefing(df, pred, tf_name):
    try:
        if "GEMINI_API_KEY" not in st.secrets:
            return "❌ Secrets에 'GEMINI_API_KEY'를 넣어주세요."
            
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # [2026년 대응] 시도해볼 모델 명칭 리스트
        # 1.5-flash가 안되면 2.0-flash나 일반 pro로 넘어갑니다.
        model_names = ['gemini-2.0-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-flash', 'gemini-pro']
        
        latest = df.iloc[-1]
        prompt = f"비트코인 {tf_name} 기준 현재 {latest['Close']:.1f}$, AI예측 {pred:.1f}$, RSI {latest['RSI']:.1f}. 친절하게 3줄 전략 짜줘."
        
        response_text = ""
        for m_name in model_names:
            try:
                model = genai.GenerativeModel(m_name)
                response = model.generate_content(prompt)
                response_text = response.text
                if response_text:
                    break # 성공하면 탈출!
            except:
                continue # 실패하면 다음 모델로
        
        if not response_text:
            return "❌ [404 해결불가] 구글 서버에서 사용 가능한 모델을 찾지 못했습니다. API 키 권한을 확인해 주세요."
            
        return response_text
        
    except Exception as e:
        return f"❌ [오류 발생] {str(e)}"

# 5. UI 구성 (생략 없이 전체 포함)
st.title("🧠 AI 비트코인 참모 v3.9")
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
