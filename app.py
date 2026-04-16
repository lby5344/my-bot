import os
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pytz
from datetime import datetime

# ==========================================
# [초기 설정] UI 및 시간 설정
# ==========================================
kst = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S')

st.set_page_config(page_title="AI 참모 v5.1 (다변량 LSTM 엔진)", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Nanum Gothic', sans-serif; }
    .briefing-box {
        background-color: #1E1E1E; color: #00FFA3; padding: 20px;
        border-radius: 10px; border-left: 5px solid #00FFA3;
        font-size: 1.1rem; line-height: 1.6; margin-bottom: 20px;
    }
    .time-display { text-align: right; color: #AAAAAA; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# [AI 브리핑] Groq LLM 연동
# ==========================================
@st.cache_data(ttl=900)
def get_ai_briefing(df_json, prediction, model_name):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY") 
    
    if not api_key:
        return "❌ API 키가 설정되지 않았습니다."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = f"""당신은 최고 수준의 비트코인 트레이딩 참모입니다. (기준 타임프레임: {model_name})\n최근 데이터: {df_json}\n예측가: {prediction}\n상황을 3문장 브리핑하고 포지션을 추천해."""
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ 브리핑 생성 오류: {str(e)}"

# ==========================================
# [데이터 수집 및 예측 엔진]
# ==========================================
def get_analysis_data(tf):
    try:
        ex = ccxt.kraken({'enableRateLimit': True})
        ohlcv = ex.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=300)
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['MA20'] = df['Close'].rolling(window=20).mean()
        return df.dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 로딩 실패: {e}"); return None

def build_and_train_model(X, y, input_shape):
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    return model

def predict_next_price(df, tf_name):
    features = ['Close', 'Volume', 'RSI']
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    seq_length = 10
    num_features = len(features)
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X_latest = scaled_data[-seq_length:].reshape(1, seq_length, num_features)
    model_path = f"ai_trader_lstm_{tf_name}.keras"
    if os.path.exists(model_path):
        try: model = load_model(model_path)
        except: 
            model = build_and_train_model(X, y, (seq_length, num_features))
            model.save(model_path)
    else:
        model = build_and_train_model(X, y, (seq_length, num_features))
        model.save(model_path)
    pred_scaled = model.predict(X_latest, verbose=0)
    dummy = np.zeros((1, num_features)); dummy[0, 0] = pred_scaled[0][0]
    return scaler.inverse_transform(dummy)[0][0]

# ==========================================
# [사이드바 설정] 
# ==========================================
st.sidebar.title("⚙️ 설정 및 관리")
if st.sidebar.button("♻️ AI 모델 재학습 (Retrain)", use_container_width=True):
    tf_current = st.session_state.get('tf', '1h')
    training_df = get_analysis_data(tf_current) 
    if training_df is not None:
        with st.spinner("최신 데이터로 재학습 중입니다..."):
            m_path = f"ai_trader_lstm_{st.session_state.get('tf_name', '1h')}.keras"
            if os.path.exists(m_path): os.remove(m_path)
        st.success("✅ 모델 초기화 완료! 메인 화면을 갱신하면 자동 재학습됩니다.")
        st.rerun()

# ==========================================
# [메인 UI] 대시보드 렌더링
# ==========================================
st.title("🤖 AI 비트코인 참모 (Multivariate LSTM v5.1)")

# 여기에 캡션 추가
st.caption("⚠️ 본 AI 예측은 과거 데이터를 기반으로 한 통계적 확률이며, 투자 결과에 대한 법적 책임을 지지 않습니다.")

st.markdown(f"<p class='time-display'>🕒 현재 분석 시간: {now_kst} (KST)</p>", unsafe_allow_html=True)
st.markdown("---")
if 'tf' not in st.session_state:
    st.session_state.tf, st.session_state.tf_name = "1h", "1h"

c1, c2, c3 = st.columns(3)
if c1.button("1시간 분석", use_container_width=True): st.session_state.tf, st.session_state.tf_name = "1h", "1h"
if c2.button("4시간 분석", use_container_width=True): st.session_state.tf, st.session_state.tf_name = "4h", "4h"
if c3.button("1일 분석", use_container_width=True): st.session_state.tf, st.session_state.tf_name = "1d", "1d"

df = get_analysis_data(st.session_state.tf)
if df is not None:
    with st.spinner(f'{st.session_state.tf_name} 기준 AI 분석 중...'):
        pred = predict_next_price(df, st.session_state.tf_name)
    
    cur = df['Close'].iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("현재가", f"${cur:,.1f}")
    m2.metric("다음 예측가", f"${pred:,.1f}", f"{pred-cur:+.1f}$")
    m3.metric("현재 RSI", f"{df['RSI'].iloc[-1]:.1f}")
    
    briefing = get_ai_briefing(df[['Close', 'Volume', 'RSI']].tail(5).to_json(), pred, st.session_state.tf_name)
    st.markdown(f'<div class="briefing-box"><strong>💬 AI 전략 브리핑</strong><br><br>{briefing}</div>', unsafe_allow_html=True)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="BTC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='MA20', line=dict(color='yellow', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#00FFA3')), row=2, col=1)
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=650, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
