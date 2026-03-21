import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai  # 구글 공식 엔진 도입!
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="AI 참모 v3.8 (공식엔진)", page_icon="🧠", layout="wide")
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

# 4. 제미나이 공식 엔진 브리핑
def get_ai_briefing(df, pred, tf_name):
    # Secrets 키 이름 확인 (대소문자/띄어쓰기 주의!)
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        
        # 모델 설정 (가장 안정적인 1.5-flash 사용)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        latest = df.iloc[-1]
        prompt = f"""
        당신은 전문 투자 참모입니다.
        - 타임프레임: {tf_name}
        - 현재가: ${latest['Close']:,.1f}
        - AI예측가: ${pred:,.1f}
        - RSI: {latest['RSI']:.1f}
        마누라님께 딱 3줄로 친절하게 투자 전략을 브리핑하세요.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except KeyError:
        return "❌ [에러] 스트림릿 Secrets에 'GEMINI_API_KEY'라는 이름이 없습니다."
    except Exception as e:
        return f"❌ [공식 엔진 에러] {str(e)}"

# 5. UI 구성
st.title("🧠 AI 비트코인 참모 v3.8")
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
    
    # 브리핑 출력
    st.info(f"💬 **AI 참모 실시간 브리핑**\n\n{get_ai_briefing(df, pred, st.session_state.tf_name)}")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="BTC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#ff00ff')), row=2, col=1)
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)
