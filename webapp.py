import streamlit as st
import yfinance as yf
import pandas as pd
import numpy
import time
import joblib
import os
from textblob import TextBlob
from newsapi import NewsApiClient
from dotenv import load_dotenv
import mplfinance as mpf
import matplotlib.pyplot as plt
import warnings

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="🚀",
    layout="wide"
)

# --- Suppress Warnings & Patches ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
numpy_version = tuple(map(int, numpy.__version__.split('.')))
if numpy_version >= (2, 0, 0):
    numpy.NaN = numpy.nan
import pandas_ta as ta
load_dotenv()

# --- Custom CSS for a cooler UI ---
st.markdown("""
<style>
    .stMetric {
        border-radius: 10px;
        padding: 15px;
        background-color: #262730;
    }
    .stMetric-value {
        font-size: 24px;
    }
    .metric-red .stMetric-value { color: #f63366; }
    .metric-green .stMetric-value { color: #00b093; }
</style>
""", unsafe_allow_html=True)

# --- Global Definitions ---
SECTORS = {
    "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"]
}
SECTOR_MAP = {ticker: sector for sector, tickers in SECTORS.items() for ticker in tickers}
analysis_log = []

# --- Cached Functions for Performance ---
@st.cache_data
def get_historical_data(ticker, period="1y"):
    hist = yf.Ticker(ticker).history(period=period)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    hist.columns = [col.capitalize() for col in hist.columns]
    return hist

@st.cache_data
def get_sector_peers_health(sector, main_ticker):
    log("🔬 Analyzing Sector Health...")
    peers = [t for t in SECTORS.get(sector, []) if t != main_ticker.upper()]
    if not peers: return "Unknown", "N/A"
    bullish_peers = 0
    with st.spinner(f'Analyzing {sector} sector peers...'):
        for peer_ticker in peers:
            try:
                hist = yf.Ticker(peer_ticker).history(period="3mo", progress=False)
                if not hist.empty:
                    sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                    if hist['Close'].iloc[-1] > sma_50: bullish_peers += 1
            except Exception: continue
    health_score = bullish_peers / len(peers)
    if health_score > 0.7: health = "Strongly Bullish"
    elif health_score > 0.4: health = "Bullish"
    else: health = "Bearish"
    details = f"{bullish_peers} of {len(peers)} peers are in an uptrend."
    log(f"   - {details}")
    log(f"   - Sector Health: {health}")
    return health, details

@st.cache_data
def get_news_sentiment(company_name):
    log("\n🔍 Analyzing News Sentiment Trend...")
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key: return "API Key Missing", 0.0
    try:
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy', page_size=20)['articles']
        if not articles: return "No News", 0.0
        polarity = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        trend = "Improving" if polarity > 0.05 else "Worsening" if polarity < -0.05 else "Stable"
        log(f"   - Avg. Polarity: {polarity:.2f} ({trend})")
        return trend, polarity
    except Exception: return "Error", 0.0

@st.cache_resource
def load_aggressive_model():
    try:
        model_payload = joblib.load("aggressive_ai_model.joblib")
        return model_payload['model'], model_payload['features']
    except FileNotFoundError:
        return None, None

def log(message):
    """Appends a message to the in-memory log."""
    analysis_log.append(message)

# --- Main App UI ---
st.title("🚀 Aggressive AI Predictive Dashboard")
model, model_features = load_aggressive_model()
if not model:
    st.warning("Warning: `aggressive_ai_model.joblib` not found. Predictive features will be disabled.", icon="⚠️")

with st.sidebar:
    st.header("Analysis Configuration")
    company_name = st.text_input("Company Name", "Tata Motors")
    ticker_symbol = st.text_input("Ticker Symbol", "TATAMOTORS.NS")
    analyze_button = st.button("Analyze & Predict", use_container_width=True, type="primary")

if analyze_button:
    analysis_log = [] # Reset log
    if not company_name or not ticker_symbol:
        st.error("Please enter both a company name and a ticker symbol.")
    else:
        data = get_historical_data(ticker_symbol)
        if data.empty:
            st.error("Could not fetch data. Please check the ticker symbol.")
        else:
            with st.spinner("Running Advanced Analysis..."):
                log("📈 Analyzing Market Trends...")
                data.ta.macd(append=True); data.ta.bbands(append=True); data.ta.obv(append=True)
                data.ta.rsi(append=True); data.ta.sma(length=50, append=True); data.ta.sma(length=200, append=True)
                data['trend_50'] = (data['Close'] > data['SMA_50']).astype(int)
                data['trend_200'] = (data['Close'] > data['SMA_200']).astype(int)
                data.dropna(inplace=True)

                latest_price = data['Close'].iloc[-1]
                latest_sma50 = data['SMA_50'].iloc[-1]
                latest_rsi = data['RSI_14'].iloc[-1]

                price_trend = "Uptrend" if latest_price > latest_sma50 else "Downtrend"
                rsi_condition = f"Overbought ({latest_rsi:.2f})" if latest_rsi > 70 else f"Oversold ({latest_rsi:.2f})" if latest_rsi < 30 else f"Neutral ({latest_rsi:.2f})"
                log(f"   - Price Trend: {price_trend}")
                log(f"   - Momentum (RSI): {rsi_condition}")

                sector = SECTOR_MAP.get(ticker_symbol.upper())
                sector_health, sector_details = get_sector_peers_health(sector, ticker_symbol) if sector else ("N/A", "Not in defined sectors.")
                sentiment_trend, sentiment_score = get_news_sentiment(company_name)
                
                prediction_prob_str = "N/A"
                if model and model_features:
                    try:
                        latest_features = data.iloc[-1]
                        current_features_df = latest_features[model_features].to_frame().T
                        prediction_prob = model.predict_proba(current_features_df)[0][1]
                        prediction_prob_str = f"{prediction_prob:.0%}"
                        log(f"\n🧠 AI Prediction calculated: {prediction_prob_str}")
                    except Exception as e:
                        log(f"\n🧠 Prediction failed: {e}")
                        prediction_prob_str = "Error"

            st.header(f"Analysis for {company_name} ({ticker_symbol})")
            
            # --- Key Metrics with Color ---
            pt_color = "green" if price_trend == "Uptrend" else "red"
            sh_color = "green" if "Bullish" in sector_health else "red" if "Bearish" in sector_health else "normal"
            
            st.markdown(f'<div class="metric-{pt_color}">', unsafe_allow_html=True)
            st.metric("📈 Price Trend", price_trend)
            st.markdown('</div>', unsafe_allow_html=True)

            st.metric("🎛️ Momentum (RSI)", rsi_condition)
            
            st.markdown(f'<div class="metric-{sh_color}">', unsafe_allow_html=True)
            st.metric("🏢 Sector Health", sector_health, help=sector_details)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.metric("📰 News Sentiment", sentiment_trend, help=f"Avg. Polarity: {sentiment_score:.2f}")

            # --- Charting ---
            chart_data = data.tail(126) # Chart last 6 months
            fig, _ = mpf.plot(chart_data, type='candle', style='yahoo', title='', ylabel='Price', volume=True,
                              mav=(50, 200), addplot=[mpf.make_addplot(chart_data['RSI_14'], panel=2, ylabel='RSI')],
                              returnfig=True, figsize=(12, 6))
            st.pyplot(fig)
            
            # --- Final Verdict ---
            st.subheader("🤖 AI Analyst Final Verdict")
            
            verdict = "Neutral. The key indicators are not aligned."
            if "Uptrend" in price_trend and "Bullish" in sector_health and "Overbought" not in rsi_condition:
                verdict = "High Conviction Buy. Strong uptrend in a bullish sector with healthy momentum."
            elif "Downtrend" in price_trend and "Bearish" in sector_health:
                verdict = "High Conviction Avoid. The stock and its sector are both in a downtrend."
            elif "Uptrend" in price_trend and "Overbought" in rsi_condition:
                verdict = "Bullish, but with caution. Overbought RSI suggests a short-term pullback is likely."

            st.info(f"**💡 Verdict:** {verdict}")
            st.success(f"**🧠 Prediction:** {prediction_prob_str} chance of >3% gain in the next 5 days.")

            # --- Analysis Log in an Expander ---
            with st.expander("Show Full Analysis Log"):
                st.code('\n'.join(analysis_log))
else:
    st.info("👋 Welcome! Enter a stock in the sidebar and click 'Analyze & Predict' to begin.")