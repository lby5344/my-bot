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
st.set_page_config(page_title="AI 참모 v3.2 (멀티 타임프레임)", page_icon="🧠", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    html, body, [data-testid="stSidebar"], p, h1, h2, h3, h4, div { font-family: 'Nanum Gothic', sans-serif !important; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; color: #00ffbb; }
</style>
""", unsafe_allow_html=True)

# 2. 데이터 가져오기 함수 (업비트 기준)
@st.cache_data(ttl=300)
def get_analysis_data(tf):
    ex = ccxt.upbit()
    ohlcv = ex.fetch_ohlcv('BTC/KRW', timeframe=tf, limit=200)
    df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9) # KST 보정
    
    # RSI 계산
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

# 3. XGBoost 학습 및 예측 함수 (대소문자 수정 완료!)
def predict_next_price(df):
    df_train = df.copy()
    # [수정] close -> Close (대문자로 변경)
    df_train['target'] = df_train['Close'].shift(-1)
    df_train = df_train.dropna()
    
    X = np.array(range(len(df_train))).reshape(-1, 1)
    y = df_train['Close'] # [수정] 여기도 대문자 Close
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X, y)
    
    next_idx = np.array([[len(df)]])
    pred = model.predict(next_idx)[0]
    return pred

# 4. 제미나이 브리핑 함수
def get_ai_briefing(df, pred, tf_name):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        latest = df.iloc[-1]
        diff = pred - latest['Close']
        status = "상승" if diff > 0 else "하락"
        
        prompt = f"""
        당신은 암호화폐 전문 'AI 참모'입니다.
        - 분석 시간대: {tf_name}
        - 현재 비트코인 가격: {latest['Close']:,.0f}원
        - AI 예측 가격: {pred:,.0f}원 ({status} 예상)
        - 현재 RSI: {latest['RSI']:.1f}
        위 데이터를 바탕으로 마누라님께 딱 3줄로 핵심 투자 전략을 브리핑하세요. 말투는 정중하면서도 친절하게.
        """
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}).json()
        return response['candidates'][0]['content']['parts'][0]['text']
    except:
        return "🤖 (제미나이 연결에 문제가 있네요. 차트 데이터를 위주로 참고해 주세요!)"

# 5. 메인 UI 구성
st.title("🧠 AI 비트코인 참모 v3.2")
st.write("---")

# 상단 시간대 선택 버튼
col_btn1, col_btn2, col_btn3 = st.columns(3)
selected_tf = "1h"
tf_display = "1시간"

# 버튼 클릭 시 세션 상태에 저장 (버튼 누르면 화면이 새로고침되므로 상태 저장이 필요합니다)
if 'tf' not in st.session_state:
    st.session_state.tf = "1h"
    st.session_state.tf_name = "1시간"

if col_btn1.button("🔄 1시간 분석", use_container_width=True):
    st.session_state.tf = "1h"
    st.session_state.tf_name = "1시간"
if col_btn2.button("🔄 4시간 분석", use_container_width=True):
    st.session_state.tf = "4h"
    st.session_state.tf_name = "4시간"
if col_btn3.button("🔄 1일 분석", use_container_width=True):
    st.session_state.tf = "1d"
    st.session_state.tf_name = "1일"

# 최종 선택된 값 사용
selected_tf = st.session_state.tf
tf_display = st.session_state.tf_name

# 데이터 처리 및 출력
with st.spinner(f'{tf_display} 흐름을 분석하는 중...'):
    df = get_analysis_data(selected_tf)
    prediction = predict_next_price(df)
    latest_price = df['Close'].iloc[-1]
    diff = prediction - latest_price
    
    st.write(f"### 📊 {tf_display} 기준 분석 결과")
    
    # 지표 카드
    m1, m2, m3 = st.columns(3)
    m1.metric("현재 가격", f"{latest_price:,.0f}원")
    m2.metric(f"{tf_display} 후 예측", f"{prediction:,.0f}원", f"{diff:+.0f}원")
    m3.metric("현재 RSI (심리)", f"{df['RSI'].iloc[-1]:.1f}")
    
    # AI 브리핑
    briefing = get_ai_briefing(df, prediction, tf_display)
    st.info(f"💬 **AI 참모의 실시간 브리핑**\n\n{briefing}")
    
    # 차트 그리기
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='BTC'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#00ffbb')), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="blue", opacity=0.1, row=2, col=1)
    
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"🕒 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
