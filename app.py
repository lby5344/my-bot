import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta

# 1. 페이지 설정 및 스타일
st.set_page_config(page_title="AI 참모 v3.3 (USD 버전)", page_icon="🧠", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    html, body, [data-testid="stSidebar"], p, h1, h2, h3, h4, div { font-family: 'Nanum Gothic', sans-serif !important; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; color: #ff00ff; }
</style>
""", unsafe_allow_html=True)

# 2. 데이터 가져오기 (Bybit 사용 - 달러 기준)
@st.cache_data(ttl=300)
def get_analysis_data(tf):
    # 바이낸스 대신 차단 없는 바이비트 사용
    ex = ccxt.bybit()
    ohlcv = ex.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=200)
    df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9) # KST
    
    # RSI 계산
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.dropna()

# 3. XGBoost 예측
def predict_next_price(df):
    df_train = df.copy()
    df_train['target'] = df_train['Close'].shift(-1)
    df_train = df_train.dropna()
    X = np.array(range(len(df_train))).reshape(-1, 1)
    y = df_train['Close']
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X, y)
    next_idx = np.array([[len(df)]])
    return model.predict(next_idx)[0]

# 4. 제미나이 브리핑 (최신 모델 gemini-1.5-flash 적용)
def get_ai_briefing(df, pred, tf_name):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        latest = df.iloc[-1]
        diff = pred - latest['Close']
        status = "상승" if diff > 0 else "하락"
        
        prompt = f"""
        당신은 암호화폐 전문 'AI 참모'입니다.
        - 분석 시간대: {tf_name}
        - 현재 BTC 가격: ${latest['Close']:,.1f}
        - AI 예측 가격: ${pred:,.1f} ({status} 예상)
        - 현재 RSI: {latest['RSI']:.1f}
        위 데이터를 바탕으로 마누라님께 딱 3줄로 핵심 투자 전략을 브리핑하세요. 말투는 신뢰감 있고 친절하게.
        """
        # 모델명을 최신 flash 버전으로 변경하여 연결성 확보
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}).json()
        return response['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"🤖 브리핑 준비 중입니다... (API 확인 필요: {str(e)})"

# 5. 메인 UI
st.title("🧠 AI 비트코인 참모 v3.3")

# 현재 시간 표시 추가
now_kst = datetime.now()
st.subheader(f"🕒 현재 시간 (KST): {now_kst.strftime('%Y-%m-%d %H:%M:%S')}")
st.write("---")

# 시간대 선택
col_btn1, col_btn2, col_btn3 = st.columns(3)
if 'tf' not in st.session_state:
    st.session_state.tf, st.session_state.tf_name = "1h", "1시간"

if col_btn1.button("🔄 1시간 분석", use_container_width=True):
    st.session_state.tf, st.session_state.tf_name = "1h", "1시간"
if col_btn2.button("🔄 4시간 분석", use_container_width=True):
    st.session_state.tf, st.session_state.tf_name = "4h", "4시간"
if col_btn3.button("🔄 1일 분석", use_container_width=True):
    st.session_state.tf, st.session_state.tf_name = "1d", "1일"

with st.spinner(f'{st.session_state.tf_name} 데이터 분석 중...'):
    df = get_analysis_data(st.session_state.tf)
    prediction = predict_next_price(df)
    latest_price = df['Close'].iloc[-1]
    diff = prediction - latest_price
    
    # 지표 출력
    m1, m2, m3 = st.columns(3)
    m1.metric("현재 가격 (USD)", f"${latest_price:,.1f}")
    m2.metric(f"{st.session_state.tf_name} 후 예측", f"${prediction:,.1f}", f"{diff:+.1f}$")
    m3.metric("RSI 지수", f"{df['RSI'].iloc[-1]:.1f}")
    
    # 브리핑 출력
    briefing = get_ai_briefing(df, prediction, st.session_state.tf_name)
    st.info(f"💬 **AI 참모의 실시간 브리핑**\n\n{briefing}")
    
    # 차트
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='BTC'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#ff00ff')), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="blue", opacity=0.1, row=2, col=1)
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)
