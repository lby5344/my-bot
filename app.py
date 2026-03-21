import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.graph_objects as go
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="AI 참모 v3.1 (XGBoost Lite)", page_icon="⚡")

st.title("🧠 AI 비트코인 참모 v3.1")
st.subheader("실전 압축형 XGBoost 엔진 탑재")

# 1. 데이터 가져오기 (업비트 - 한국에서 가장 안정적!)
@st.cache_data(ttl=600)
def get_data():
    # 바이낸스 대신 업비트로 교체!
    exchange = ccxt.upbit() 
    ohlcv = exchange.fetch_ohlcv('BTC/KRW', timeframe='1h', limit=200)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df
try:
    df = get_data()

    # 2. 아주 간단한 학습 데이터 만들기
    df['target'] = df['close'].shift(-1) # 다음 시간 가격 맞추기
    train_df = df.dropna()

    X = np.array(range(len(train_df))).reshape(-1, 1)
    y = train_df['close']

    # 3. XGBoost 모델 (가벼움의 극치!)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)

    # 4. 예측
    last_idx = np.array([[len(df)]])
    prediction = model.predict(last_idx)[0]
    current_price = df['close'].iloc[-1]
    diff = prediction - current_price

    # 5. 화면 출력
    col1, col2 = st.columns(2)
    col1.metric("현재 가격", f"${current_price:,.2f}")
    col2.metric("AI 예측 (다음 시간)", f"${prediction:,.2f}", f"{diff:+.2f}")

    # 차트 그리기
    fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close'])])
    fig.update_layout(title='최근 비트코인 흐름', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    st.success("✅ 가벼운 XGBoost 엔진으로 성공적으로 분석을 마쳤습니다!")

except Exception as e:
    st.error(f"에러 발생: {e}")

st.info("💡 이 버전은 메모리 제한이 있는 무료 서버를 위해 최적화된 Lite 버전입니다.")
