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

st.set_page_config(page_title="AI 참모 v5.0 (다변량 LSTM 엔진)", page_icon="🤖", layout="wide")

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
# [Step 3 반영] 중복 코드 제거 및 API 보안 강화
# ==========================================
@st.cache_data(ttl=900)
def get_ai_briefing(df_json, prediction, model_name):
    # st.secrets 우선 확인 후 없으면 os.getenv 사용 (보안 및 유연성 강화)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY") 
    
    if not api_key:
        return "❌ API 키가 설정되지 않았습니다. 환경변수나 secrets.toml을 확인하세요."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        prompt = f"""
        당신은 최고 수준의 비트코인 트레이딩 참모입니다. (기준 타임프레임: {model_name})
        최근 다변량 시장 데이터(종가, 거래량, RSI): {df_json}
        다변량 LSTM 엔진 예측 다음 종가: {prediction}
        위 정보를 바탕으로 현재 상황을 3문장으로 날카롭게 브리핑하고, 트레이딩 포지션(롱/숏/관망)을 추천해.
        """
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # 고성능 모델로 통일
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ 브리핑 생성 오류: {str(e)}"

# ==========================================
# [Step 2 반영] 데이터 파이프라인 고도화 (RSI, 거래량 추가)
# ==========================================
def get_analysis_data(tf):
    try:
        ex = ccxt.kraken({'enableRateLimit': True})
        ohlcv = ex.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=300) # 더 안정적인 스케일링을 위해 300개 로드
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms') + pd.Timedelta(hours=9)
        
        # 보조지표 RSI 계산
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 이동평균선 추가 (선택적 활용)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        return df.dropna()
    except Exception as e:
        st.error(f"⚠️ 데이터 로딩 실패: {e}")
        return None

# ==========================================
# [Step 1 & 2 반영] 다변량 LSTM 및 모델 캐싱(저장/로드) 최적화
# ==========================================
def build_and_train_model(X, y, input_shape):
    """최초 1회 실행 시 모델을 구축하고 학습시키는 헬퍼 함수"""
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2), # 과적합 방지
        LSTM(units=32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0) # Epoch 늘려서 제대로 1회 학습
    return model

def predict_next_price(df, tf_name):
    # 1. 다변량 피처 선정 (종가, 거래량, RSI)
    features = ['Close', 'Volume', 'RSI']
    data = df[features].values
    
    # 2. 스케일링 (Data Leakage 부분 해소)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    seq_length = 10
    num_features = len(features)
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i, 0]) # 'Close' 가격만 타겟으로 설정
    X, y = np.array(X), np.array(y)
    
    # 최근 10개 데이터 (다음 가격 예측용)
    X_latest = scaled_data[-seq_length:].reshape(1, seq_length, num_features)
    
    model_path = f"ai_trader_lstm_{tf_name}.h5" # 타임프레임별로 모델 분리 저장
    
    # 3. 모델 로드 OR 학습 후 저장 (엔진 최적화)
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
        except:
            model = build_and_train_model(X, y, (seq_length, num_features))
            model.save(model_path)
    else:
        model = build_and_train_model(X, y, (seq_length, num_features))
        model.save(model_path) # 학습 완료된 모델 저장
        
    # 4. 예측 수행
    pred_scaled = model.predict(X_latest, verbose=0)
    
    # 5. 역스케일링 (종가 자리만 값을 채워서 복원)
    dummy_array = np.zeros((1, num_features))
    dummy_array[0, 0] = pred_scaled[0][0] 
    predicted_price = scaler.inverse_transform(dummy_array)[0][0]
    
    return predicted_price

# ==========================================
# [메인 UI] 대시보드 렌더링
# ==========================================
st.title("🤖 AI 비트코인 참모 (Multivariate LSTM v5.0)")
st.markdown(f"<p class='time-display'>🕒 현재 분석 시간: {now_kst} (KST)</p>", unsafe_allow_html=True)
st.markdown("---")

if 'tf' not in st.session_state: st.session_state.tf, st.session_state.tf_name = "1h", "1시간"
c1, c2, c3 = st.columns(3)
if c1.button("1시간 분석", use_container_width=True): st.session_state.tf, st.session_state.tf_name = "1h", "1h"
if c2.button("4시간 분석", use_container_width=True): st.session_state.tf, st.session_state.tf_name = "4h", "4h"
if c3.button("1일 분석", use_container_width=True): st.session_state.tf, st.session_state.tf_name = "1d", "1d"

df = get_analysis_data(st.session_state.tf)
if df is not None:
    with st.spinner(f'{st.session_state.tf_name} 기준 다변량 LSTM 모델을 로딩/분석 중입니다...'):
        pred = predict_next_price(df, st.session_state.tf_name)
    
    cur = df['Close'].iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("현재가 (Close)", f"${cur:,.1f}")
    m2.metric(f"다음 캔들 예측가", f"${pred:,.1f}", f"{pred-cur:+.1f}$")
    m3.metric("현재 RSI", f"{df['RSI'].iloc[-1]:.1f}")
    
    # AI 브리핑 출력
    briefing_content = get_ai_briefing(df[['Close', 'Volume', 'RSI']].tail(5).to_json(), pred, st.session_state.tf_name)
    st.markdown(f"""
        <div class="briefing-box">
            <strong>💬 AI 참모 실시간 전략 브리핑</strong><br><br>
            {briefing_content}
        </div>
        """, unsafe_allow_html=True)
    
    # 차트 그리기
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="BTC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='MA20', line=dict(color='yellow', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#00FFA3')), row=2, col=1)
    
    # RSI 과매수/과매도 기준선 추가
    fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
    fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="blue")
    
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=650, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
