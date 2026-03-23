import os
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. 페이지 설정
st.set_page_config(page_title="AI 참모 v4.5 (LSTM 엔진)", page_icon="🤖", layout="wide")

# 2. 데이터 가져오기 (Kraken)
@st.cache_data(ttl=900)
def get_ai_briefing(df_json, prediction, model_name):
    # 이지패널 Environment 탭에 넣은 이름과 똑같아야 합니다!
    api_key = os.getenv("GROQ_API_KEY") 
    
    if not api_key:
        return "❌ API 키가 설정되지 않았습니다. 이지패널 설정을 확인하세요."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        prompt = f"""
        당신은 전문 비트코인 트레이딩 참모입니다. (모델: {model_name})
        최근 시장 데이터: {df_json}
        LSTM 예측 결과: {prediction}
        위 정보를 바탕으로 현재 상황을 3문장으로 날카롭게 브리핑해줘.
        """
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ 브리핑 생성 오류: {str(e)}"

# --- 여기 아래에 원래 있던 18행(def get_analysis_data)이 오면 됩니다 ---
def get_analysis_data(tf):
    try:
        ex = ccxt.kraken({'enableRateLimit': True})
        ohlcv = ex.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=200)
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
        
        # 보조지표 RSI 계산
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df.dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 로딩 실패: {e}")
        return None

# 3. 진짜 LSTM 예측 엔진
def predict_next_price(df):
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 최근 10개의 데이터를 보고 다음 1개를 예측하는 구조
    X, y = [], []
    for i in range(10, len(scaled_data)):
        X.append(scaled_data[i-10:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 모델 구성
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0) # 실시간 학습

    # 다음 가격 예측
    last_10 = scaled_data[-10:].reshape(1, 10, 1)
    pred_scaled = model.predict(last_10, verbose=0)
    return scaler.inverse_transform(pred_scaled)[0][0]

# 4. 제미나이 브리핑 (이지패널 API 키 적용 버전)
@st.cache_data(ttl=1800)
def get_groq_briefing(df_summary, prediction):
    # 이지패널 환경설정에 넣은 이름(GROQ_API_KEY)과 똑같아야 합니다!
    api_key = os.getenv("GROQ_API_KEY") 
    
    if not api_key:
        return "❌ API 키가 설정되지 않았습니다. 이지패널 Environment 탭을 확인하세요."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        prompt = f"""
        당신은 전문 비트코인 트레이딩 참모입니다.
        현재 데이터: {df_summary}
        LSTM 예측 결과: {prediction}
        위 데이터를 바탕으로 현재 상황을 3문장으로 날카롭게 브리핑해줘.
        """
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # FORCE UPDATE V2
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ 브리핑 생성 오류: {str(e)}"
# 5. UI 구성
st.title("🤖 AI 비트코인 참모 (LSTM v4.5)")

if 'tf' not in st.session_state: st.session_state.tf, st.session_state.tf_name = "1h", "1시간"
c1, c2, c3 = st.columns(3)
if c1.button("1시간 분석", use_container_width=True): st.session_state.tf, st.session_state.tf_name = "1h", "1시간"
if c2.button("4시간 분석", use_container_width=True): st.session_state.tf, st.session_state.tf_name = "4h", "4시간"
if c3.button("1일 분석", use_container_width=True): st.session_state.tf, st.session_state.tf_name = "1d", "1일"

df = get_analysis_data(st.session_state.tf)
if df is not None:
    with st.spinner('LSTM 엔진이 뇌를 풀가동 중입니다...'):
        pred = predict_next_price(df)
    
    cur = df['Close'].iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("현재가", f"${cur:,.1f}")
    m2.metric(f"{st.session_state.tf_name} 후 예측", f"${pred:,.1f}", f"{pred-cur:+.1f}$")
    m3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    
    # [중요] .to_json()을 붙여서 JSON 글자로 전달!
    st.info(f"💬 **AI 참모 LSTM 실시간 브리핑**\n\n{get_ai_briefing(df.to_json(), pred, st.session_state.tf_name)}")
    
    # 차트 그리기
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="BTC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#ff00ff')), row=2, col=1)
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)
