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

# 4. 제미나이 브리핑 (캐시 기능을 넣어서 Limit 에러 방지!)
@st.cache_data(ttl=1800) # 한번 물어본 브리핑은 30분(1800초) 동안 재사용합니다!
def get_ai_briefing(df_json, pred, tf_name): # df 대신 json 형태로 넘겨야 캐싱이 잘 됩니다.
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        
        # 2.5-flash가 Limit이 심하면 1.5-flash로 수동 고정하는 것도 방법입니다.
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        df = pd.read_json(df_json) # 전달받은 데이터를 다시 데이터프레임으로
        latest = df.iloc[-1]
        prompt = f"비트코인 {tf_name} 기준 현재 {latest['Close']:.1f}$, AI예측 {pred:.1f}$. 투자 전략 3줄 요약해줘."
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "⚠️ 구글 AI가 지금 너무 바쁩니다(한도 초과). 10분만 쉬었다가 다시 물어봐 주세요!"
        return f"❌ 브리핑 생성 오류: {str(e)}"

# [주의] 아래 메인 UI 부분에서 호출할 때 이렇게 바꿔주세요!
# briefing = get_ai_briefing(df.to_json(), prediction, st.session_state.tf_name)

# 5. UI 구성
st.title("🧠 AI 비트코인 참모 v4.0")
from datetime import datetime
import pytz

kst = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(kst)

st.subheader(f"🕒 현재 시간(KST): {now_kst.strftime('%Y-%m-%d %H:%M:%S')}")
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
