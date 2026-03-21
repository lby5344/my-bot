import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="AI 참모 v3.5 (진단모드)", page_icon="🧠", layout="wide")

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
        st.error(f"⚠️ 거래소 연결 에러: {e}")
        return None

# 3. XGBoost 예측
def predict_next_price(df):
    df_train = df.copy()
    df_train['target'] = df_train['Close'].shift(-1)
    df_train = df_train.dropna()
    X = np.array(range(len(df_train))).reshape(-1, 1)
    y = df_train['Close']
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    return model.predict(np.array([[len(df)]]))[0]

# 4. 제미나이 브리핑 (진단 기능 강화!)
def get_ai_briefing(df, pred, tf_name):
    # 1단계: 키가 있는지 확인
    if "GEMINI_API_KEY" not in st.secrets:
        return "❌ [에러] 스트림릿 Secrets에 'GEMINI_API_KEY'라는 이름이 아예 없습니다. 이름을 확인해 주세요!"
    
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        latest = df.iloc[-1]
        prompt = f"비트코인 {tf_name} 기준, 현재 {latest['Close']:.1f}$, AI예측 {pred:.1f}$. 짧게 3줄 전략 짜줘."
        
        # 최신 모델로 시도
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
        
        if res.status_code == 200:
            return res.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            # 여기서 왜 에러가 나는지 범인을 잡아냅니다!
            return f"❌ [에러 코드: {res.status_code}] 구글 서버가 거절했습니다. 이유: {res.text[:100]}"
            
    except Exception as e:
        return f"❌ [기타 에러] 연결 중 오류 발생: {str(e)}"

# 5. UI 구성
st.title("🧠 AI 비트코인 참모 v3.5 (진단중)")
st.subheader(f"🕒 현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if 'tf' not in st.session_state: st.session_state.tf, st.session_state.tf_name = "1h", "1시간"
c1, c2, c3 = st.columns(3)
if c1.button("1시간"): st.session_state.tf, st.session_state.tf_name = "1h", "1시간"
if c2.button("4시간"): st.session_state.tf, st.session_state.tf_name = "4h", "4시간"
if c3.button("1일"): st.session_state.tf, st.session_state.tf_name = "1d", "1일"

df = get_analysis_data(st.session_state.tf)
if df is not None:
    pred = predict_next_price(df)
    cur = df['Close'].iloc[-1]
    
    st.metric("현재가 vs AI예측", f"${cur:,.1f}", f"{pred-cur:+.1f}$")
    
    # 💬 여기가 핵심입니다!
    st.info(f"💬 **AI 참모 진단 보고서**\n\n{get_ai_briefing(df, pred, st.session_state.tf_name)}")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
