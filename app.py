import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import pickle
import hashlib
import requests
from typing import Dict, List, Tuple, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Advanced All-Asset Scanner v3 Ultimate", 
    page_icon="🌍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-strong {
        background-color: #90EE90;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
    }
    .alert-buy {
        background-color: #ADD8E6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #007bff;
    }
    .options-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .backtest-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .portfolio-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ===== SESSION STATE INITIALIZATION =====
if 'user' not in st.session_state:
    st.session_state.user = None
if 'scan_data' not in st.session_state:
    st.session_state.scan_data = {}
if 'scan_times' not in st.session_state:
    st.session_state.scan_times = {}
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'watchlists' not in st.session_state:
    st.session_state.watchlists = {}
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}
if 'email_alerts' not in st.session_state:
    st.session_state.email_alerts = {}
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = []
if 'preferences' not in st.session_state:
    st.session_state.preferences = {
        'theme': 'dark',
        'data_retention_days': 30,
        'alert_frequency': 'daily',
        'email_enabled': False,
        'sms_enabled': False,
        'desktop_alerts': True
    }

# ===== ENHANCED ASSET CLASS DEFINITIONS =====
ASSET_CLASSES = {
    "💎 Penny Stocks": {
        "default": """VFF
IOVA
BITF
RLMD
PBYI
ONDS
SOUN
PLUG
HIVE
RIOT""",
        "max_price": 5.0,
        "price_range": (0.01, 10.0),
        "step": 0.50,
        "description": "Stocks under $5 trading on major exchanges"
    },
    
    "👑 Dividend Aristocrats": {
        "default": """PG
JNJ
KO
PEP
MCD
MMM
ABT
CL
ED
SYK
ABBV
CAH
CMS
SPGI
JCI""",
        "max_price": 500.0,
        "price_range": (1.0, 500.0),
        "step": 50.0,
        "description": "Stocks with 25+ years of dividend increases - NEW PHASE 2A"
    },
    
    "🚀 Small Cap Growth": {
        "default": """IWM
SCHA
IUOT
VBR
VIOV
EEM
ARKW
TILT
GGCR
SMLL
SCSS
SLYV
DGRO
RSPB
VIOV""",
        "max_price": 200.0,
        "price_range": (1.0, 500.0),
        "step": 25.0,
        "description": "Small-cap growth stocks and ETFs - NEW PHASE 2A"
    },

    "₿ Cryptocurrency": {
        "default": """BTC-USD
ETH-USD
BNB-USD
SOL-USD
XRP-USD
ADA-USD
DOGE-USD
MATIC-USD
DOT-USD
AVAX-USD
PEPE-USD
BONK-USD
WIF-USD
FLOKI-USD
MEME-USD""",
        "max_price": 5000.0,
        "price_range": (0.0001, 100000.0),
        "step": 100.0,
        "description": "Major cryptocurrencies vs USD"
    },

    "🌊 Crypto DeFi Tokens": {
        "default": """AAVE-USD
UNI-USD
SUSHI-USD
1INCH-USD
CURVE-USD
LIDO-USD
MKR-USD
SNX-USD
YEARN-USD
COMP-USD
GRV-USD
LDO-USD
BALANCER-USD""",
        "max_price": 10000.0,
        "price_range": (0.01, 100000.0),
        "step": 100.0,
        "description": "DeFi tokens - Aave, Uniswap, Curve, Lido - NEW PHASE 2A"
    },

    "🥇 Precious Metals": {
        "default": """GLD
SLV
IAU
PSLV
SIVR
ABX
NEM
WPM
GOLD
HL
FSM
PAAS
GPL
CDE
USAS""",
        "max_price": 500.0,
        "price_range": (1.0, 500.0),
        "step": 10.0,
        "description": "Gold/Silver stocks and ETFs - NEW PHASE 2A"
    },

    "💱 Forex": {
        "default": """EURUSD=X
GBPUSD=X
USDJPY=X
AUDUSD=X
USDCAD=X
USDCHF=X
NZDUSD=X
EURGBP=X
EURJPY=X
GBPJPY=X
AUDJPY=X
CHFJPY=X""",
        "max_price": 200.0,
        "price_range": (0.01, 200.0),
        "step": 10.0,
        "description": "Currency pairs (15-min delay)"
    },
    
    "🥇 Commodities": {
        "default": """GC=F
SI=F
CL=F
NG=F
HG=F
PL=F
PA=F
ZC=F
ZS=F
ZW=F
CT=F
KC=F
SB=F
CC=F""",
        "max_price": 3000.0,
        "price_range": (1.0, 5000.0),
        "step": 100.0,
        "description": "Futures: Gold, Silver, Oil, Grains, etc."
    },
    
    "📊 ETFs": {
        "default": """SPY
QQQ
IWM
DIA
VTI
VOO
GLD
SLV
TLT
XLE
XLF
XLK
XLV
ARKK
SQQQ
TQQQ
VEA
VWO
AGG
BND""",
        "max_price": 500.0,
        "price_range": (1.0, 1000.0),
        "step": 50.0,
        "description": "Index funds and sector ETFs"
    },
}

# ===== ENHANCED TECHNICAL INDICATORS =====
def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands - NEW PHASE 2A"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels - NEW PHASE 2A"""
    diff = high - low
    fib_levels = {
        '0%': high,
        '23.6%': high - (diff * 0.236),
        '38.2%': high - (diff * 0.382),
        '50%': high - (diff * 0.5),
        '61.8%': high - (diff * 0.618),
        '100%': low,
    }
    return fib_levels

def calculate_moving_average_crossover(prices: pd.Series, fast: int = 10, slow: int = 20) -> Tuple[pd.Series, pd.Series]:
    """Calculate Moving Average Crossover - NEW PHASE 2A"""
    fast_ma = prices.rolling(window=fast).mean()
    slow_ma = prices.rolling(window=slow).mean()
    return fast_ma, slow_ma

def detect_volume_spikes(volumes: pd.Series, threshold_multiplier: float = 1.5) -> pd.Series:
    """Detect volume spikes - NEW PHASE 2A"""
    avg_volume = volumes.rolling(window=20).mean()
    spike_indicator = volumes > (avg_volume * threshold_multiplier)
    return spike_indicator

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD"""
    ema_12 = prices.ewm(span=12).mean()
    ema_26 = prices.ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    return macd, signal, histogram

# ===== NEWS & SENTIMENT ANALYSIS =====
def fetch_stock_news(symbol: str, limit: int = 5) -> List[Dict]:
    """Fetch latest news for a symbol - NEW PHASE 2A"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if news:
            return news[:limit]
    except Exception as e:
        st.warning(f"Could not fetch news for {symbol}: {str(e)}")
    return []

def analyze_sentiment(text: str) -> Dict[str, float]:
    """Simple sentiment analysis - NEW PHASE 2A"""
    positive_words = ['buy', 'bullish', 'surge', 'soar', 'gains', 'profit', 'strong', 'excellent', 'beat']
    negative_words = ['sell', 'bearish', 'crash', 'plunge', 'loss', 'weak', 'poor', 'miss', 'concern']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total = positive_count + negative_count
    if total == 0:
        return {'sentiment': 'NEUTRAL', 'score': 0.5}
    
    score = positive_count / total
    if score > 0.65:
        sentiment = 'BULLISH'
    elif score < 0.35:
        sentiment = 'BEARISH'
    else:
        sentiment = 'NEUTRAL'
    
    return {'sentiment': sentiment, 'score': score}

# ===== PRICE ALERTS =====
def send_email_alert(recipient: str, symbol: str, price: float, target: float, alert_type: str) -> bool:
    """Send email alert - NEW PHASE 2A - SendGrid Structure Ready"""
    try:
        # Structure ready for SendGrid API integration
        alert_data = {
            'to': recipient,
            'from': 'alerts@scanner.app',
            'subject': f'🚨 Price Alert: {symbol}',
            'html': f"""
            <html>
                <body style="font-family: Arial;">
                    <h2>Price Alert: {symbol}</h2>
                    <p><strong>Current Price:</strong> ${price:.2f}</p>
                    <p><strong>Target Price:</strong> ${target:.2f}</p>
                    <p><strong>Alert Type:</strong> {alert_type}</p>
                    <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </body>
            </html>
            """
        }
        st.success(f"✅ Email structure ready (SendGrid API key needed)")
        return True
    except Exception as e:
        st.error(f"Email error: {str(e)}")
        return False

def send_sms_alert(phone: str, symbol: str, price: float, target: float) -> bool:
    """Send SMS alert - NEW PHASE 2A - Twilio Structure Ready"""
    try:
        # Structure ready for Twilio API integration
        alert_message = f"Alert: {symbol} hit ${price:.2f} (Target: ${target:.2f})"
        st.info(f"📱 SMS ready (Twilio API key needed): {alert_message}")
        return True
    except Exception as e:
        st.error(f"SMS error: {str(e)}")
        return False

def create_desktop_alert(symbol: str, message: str):
    """Create desktop notification - NEW PHASE 2A"""
    st.toast(f"🔔 {symbol}: {message}", icon="📢")

# ===== WATCHLIST MANAGEMENT =====
def import_watchlist_csv(csv_content: str) -> Dict[str, List[str]]:
    """Import watchlist from CSV - NEW PHASE 2A"""
    try:
        df = pd.read_csv(StringIO(csv_content))
        watchlists = {}
        
        if 'Watchlist' in df.columns and 'Symbols' in df.columns:
            for _, row in df.iterrows():
                name = row['Watchlist']
                symbols = [s.strip() for s in str(row['Symbols']).split(',')]
                watchlists[name] = symbols
        else:
            st.error("CSV must have 'Watchlist' and 'Symbols' columns")
        
        return watchlists
    except Exception as e:
        st.error(f"CSV import error: {str(e)}")
        return {}

def export_watchlist_csv(watchlists: Dict) -> str:
    """Export watchlist to CSV - NEW PHASE 2A"""
    data = []
    for name, symbols in watchlists.items():
        data.append({'Watchlist': name, 'Symbols': ', '.join(symbols)})
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

# ===== PORTFOLIO TRACKING =====
def add_to_portfolio(symbol: str, quantity: float, entry_price: float, entry_date: str = None):
    """Add holding to portfolio - NEW PHASE 2A"""
    if entry_date is None:
        entry_date = datetime.now().strftime('%Y-%m-%d')
    
    holding = {
        'symbol': symbol,
        'quantity': quantity,
        'entry_price': entry_price,
        'entry_date': entry_date,
        'entry_value': quantity * entry_price
    }
    
    st.session_state.portfolio.append(holding)
    st.success(f"✅ Added {quantity} shares of {symbol} at ${entry_price}")

def calculate_portfolio_metrics(portfolio: List[Dict]) -> Dict:
    """Calculate portfolio P&L and metrics - NEW PHASE 2A"""
    if not portfolio:
        return {
            'total_invested': 0,
            'total_value': 0,
            'total_pl': 0,
            'total_pl_pct': 0,
            'holdings': 0
        }
    
    total_invested = sum(h['entry_value'] for h in portfolio)
    total_value = 0
    current_prices = {}
    
    for holding in portfolio:
        try:
            ticker = yf.Ticker(holding['symbol'])
            current_price = ticker.info.get('currentPrice', 0)
            current_prices[holding['symbol']] = current_price
            total_value += holding['quantity'] * current_price
        except:
            pass
    
    total_pl = total_value - total_invested
    total_pl_pct = (total_pl / total_invested * 100) if total_invested > 0 else 0
    
    return {
        'total_invested': total_invested,
        'total_value': total_value,
        'total_pl': total_pl,
        'total_pl_pct': total_pl_pct,
        'holdings': len(portfolio),
        'current_prices': current_prices
    }

def format_portfolio_dataframe(portfolio: List[Dict], metrics: Dict) -> pd.DataFrame:
    """Format portfolio data for display - NEW PHASE 2A"""
    data = []
    
    for holding in portfolio:
        symbol = holding['symbol']
        current_price = metrics['current_prices'].get(symbol, 0)
        current_value = holding['quantity'] * current_price
        pl = current_value - holding['entry_value']
        pl_pct = (pl / holding['entry_value'] * 100) if holding['entry_value'] > 0 else 0
        
        data.append({
            'Symbol': symbol,
            'Quantity': holding['quantity'],
            'Entry Price': f"${holding['entry_price']:.2f}",
            'Current Price': f"${current_price:.2f}",
            'Entry Value': f"${holding['entry_value']:.2f}",
            'Current Value': f"${current_value:.2f}",
            'P&L': f"${pl:.2f}",
            'P&L %': f"{pl_pct:.2f}%",
            'Entry Date': holding['entry_date']
        })
    
    return pd.DataFrame(data)

# ===== SCANNER ENGINE (Enhanced) =====
def scan_asset_class(symbols: List[str]) -> pd.DataFrame:
    """Scan symbols with enhanced indicators - NEW PHASE 2A"""
    results = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="200d")
            
            if hist.empty:
                continue
            
            # Get current data
            current_price = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = ((current_price - previous_close) / previous_close) * 100
            
            # Volume data
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].tail(20).mean()
            
            # Enhanced indicators - PHASE 2A
            rsi = calculate_rsi(hist['Close']).iloc[-1]
            macd, signal, histogram = calculate_macd(hist['Close'])
            macd_val = macd.iloc[-1]
            signal_val = signal.iloc[-1]
            histogram_val = histogram.iloc[-1]
            
            # Bollinger Bands - NEW
            upper_band, middle_band, lower_band = calculate_bollinger_bands(hist['Close'])
            bb_upper = upper_band.iloc[-1]
            bb_lower = lower_band.iloc[-1]
            bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100 if bb_upper != bb_lower else 50
            
            # Volume spikes - NEW
            volume_spike = detect_volume_spikes(hist['Volume']).iloc[-1]
            
            # Moving average crossover - NEW
            fast_ma, slow_ma = calculate_moving_average_crossover(hist['Close'])
            ma_fast = fast_ma.iloc[-1]
            ma_slow = slow_ma.iloc[-1]
            ma_signal = 'BULLISH' if ma_fast > ma_slow else 'BEARISH'
            
            # Generate alert
            alert = generate_alert_v3(rsi, macd_val, signal_val, bb_position, ma_signal, volume_spike)
            
            results.append({
                'Symbol': symbol,
                'Price': current_price,
                'Change %': change,
                'RSI': rsi,
                'MACD': macd_val,
                'Signal': signal_val,
                'Histogram': histogram_val,
                'BB Upper': bb_upper,
                'BB Lower': bb_lower,
                'BB Position': bb_position,
                'MA Signal': ma_signal,
                'Volume Spike': '🔥' if volume_spike else '',
                'Alert': alert,
                'Volume': current_volume,
                'Avg Volume': avg_volume,
            })
            
        except Exception as e:
            continue
    
    return pd.DataFrame(results)

def generate_alert_v3(rsi: float, macd: float, signal: float, bb_pos: float, ma_signal: str, volume_spike: bool) -> str:
    """Generate enhanced alert with new indicators - PHASE 2A"""
    score = 0
    
    # RSI scoring
    if rsi < 30:
        score += 2
    elif rsi > 70:
        score -= 1
    
    # MACD scoring
    if macd > signal:
        score += 1
    
    # Bollinger Bands scoring
    if bb_pos < 20:
        score += 1
    elif bb_pos > 80:
        score -= 1
    
    # MA signal
    if ma_signal == 'BULLISH':
        score += 1
    
    # Volume spike boost
    if volume_spike:
        score += 0.5
    
    if score >= 3:
        return "🟢 STRONG BUY"
    elif score >= 1.5:
        return "🔵 BUY"
    elif score >= 0:
        return "🟡 WATCH"
    else:
        return "🔴 SELL"

# ===== MAIN APP =====
def main():
    st.markdown('<h1 class="main-header">🌍 Advanced All-Asset Scanner v3 Ultimate</h1>', unsafe_allow_html=True)
    st.markdown("**Phase 2A Features: New Assets • Advanced Indicators • News Integration • Portfolio Tracking**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Scanner Controls")
        st.session_state.user = "Demo User"
        
        selected_asset = st.selectbox("Select Asset Class", list(ASSET_CLASSES.keys()))
        asset_info = ASSET_CLASSES[selected_asset]
        
        st.info(f"📌 {asset_info['description']}")
        
        symbols_input = st.text_area(
            "Symbols to scan",
            value=asset_info['default'],
            height=100
        )
        
        # NEW PHASE 2A: CSV Import
        st.markdown("---")
        st.markdown("### 📥 CSV Watchlist Import")
        csv_file = st.file_uploader("Upload CSV watchlist", type=['csv'])
        if csv_file:
            csv_content = csv_file.getvalue().decode('utf-8')
            imported = import_watchlist_csv(csv_content)
            for name, symbols in imported.items():
                st.session_state.watchlists[name] = symbols
            st.success(f"✅ Imported {len(imported)} watchlist(s)")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔍 Scanner",
        "📊 Backtest",
        "🎯 Options",
        "📈 Signals",
        "💼 Portfolio",
        "❤️ Watchlists",
        "⚙️ Settings"
    ])
    
    # ===== TAB 1: SCANNER (Enhanced) =====
    with tab1:
        st.markdown("### 🔍 Enhanced Asset Scanner")
        st.markdown("*Phase 2A: 4 New Indicators • Volume Spikes • MA Crossovers • Fibonacci Levels*")
        
        if st.button("▶️ RUN ENHANCED SCAN", type="primary", use_container_width=True, key="scan_btn"):
            with st.spinner("🔄 Scanning with Phase 2A indicators..."):
                symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
                
                scan_df = scan_asset_class(symbols)
                st.session_state.scan_data[selected_asset] = scan_df
                st.session_state.scan_times[selected_asset] = datetime.now()
                
                if not scan_df.empty:
                    st.success(f"✅ Scanned {len(symbols)} assets")
                    st.markdown("---")
                    
                    # Display results with all Phase 2A indicators
                    display_cols = ['Symbol', 'Price', 'Change %', 'RSI', 'BB Position', 'MA Signal', 'Volume Spike', 'Alert']
                    st.dataframe(
                        scan_df[display_cols].style.highlight_max(axis=0),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Enhanced alerts
                    st.markdown("---")
                    st.markdown("### 🚨 Alert Summary")
                    
                    strong_buys = scan_df[scan_df['Alert'] == '🟢 STRONG BUY']
                    buys = scan_df[scan_df['Alert'] == '🔵 BUY']
                    
                    col_alerts1, col_alerts2, col_alerts3 = st.columns(3)
                    
                    with col_alerts1:
                        st.markdown(f"<div class='metric-box'><h3>{len(strong_buys)}</h3><p>Strong Buys</p></div>", unsafe_allow_html=True)
                    
                    with col_alerts2:
                        st.markdown(f"<div class='metric-box'><h3>{len(buys)}</h3><p>Buys</p></div>", unsafe_allow_html=True)
                    
                    with col_alerts3:
                        st.markdown(f"<div class='metric-box'><h3>{datetime.now().strftime('%H:%M:%S')}</h3><p>Scan Time</p></div>", unsafe_allow_html=True)
                    
                    # NEW PHASE 2A: News integration
                    st.markdown("---")
                    st.markdown("### 📰 News & Sentiment (Phase 2A)")
                    
                    if not strong_buys.empty:
                        selected_symbol = st.selectbox("Select symbol for news", strong_buys['Symbol'].values)
                        news = fetch_stock_news(selected_symbol, limit=3)
                        
                        if news:
                            for item in news:
                                col_news1, col_news2 = st.columns([3, 1])
                                with col_news1:
                                    st.write(f"**{item.get('title', 'No title')}**")
                                    sentiment = analyze_sentiment(item.get('title', ''))
                                with col_news2:
                                    st.metric("Sentiment", sentiment['sentiment'], delta=f"{sentiment['score']:.2f}")
    
    # ===== TAB 2: BACKTEST =====
    with tab2:
        st.markdown("### 📊 Strategy Backtest")
        
        col_bt1, col_bt2 = st.columns(2)
        
        with col_bt1:
            bt_symbol = st.text_input("Symbol to backtest", value="SPY")
        
        with col_bt2:
            bt_days = st.number_input("Days of history", value=365, min_value=30)
        
        if st.button("▶️ Run Backtest", type="primary"):
            st.info("📌 Full backtesting engine (Phase 2B)")
    
    # ===== TAB 3: OPTIONS =====
    with tab3:
        st.markdown("### 🎯 Options Strategy Builder")
        st.info("📌 Options pricing with Greeks (Phase 2B)")
    
    # ===== TAB 4: SIGNALS =====
    with tab4:
        st.markdown("### 📈 Signal Performance History")
        st.info("📌 Signal tracking (Phase 2B)")
    
    # ===== TAB 5: PORTFOLIO TRACKING (NEW PHASE 2A) =====
    with tab5:
        st.markdown("### 💼 Portfolio Tracking - NEW PHASE 2A")
        st.markdown("*Track holdings, calculate P&L, compare vs benchmarks*")
        
        col_port1, col_port2 = st.columns(2)
        
        with col_port1:
            st.markdown("#### ➕ Add Holding")
            port_symbol = st.text_input("Symbol", placeholder="AAPL")
            port_qty = st.number_input("Quantity", value=1.0, min_value=0.01)
            port_price = st.number_input("Entry Price", value=100.0, min_value=0.01)
            
            if st.button("➕ Add to Portfolio", type="primary", use_container_width=True):
                add_to_portfolio(port_symbol, port_qty, port_price)
                st.rerun()
        
        with col_port2:
            st.markdown("#### 📊 Portfolio Summary")
            if st.session_state.portfolio:
                metrics = calculate_portfolio_metrics(st.session_state.portfolio)
                
                st.markdown(f"""
                <div class='portfolio-box'>
                    <p><b>Total Invested:</b> ${metrics['total_invested']:.2f}</p>
                    <p><b>Current Value:</b> ${metrics['total_value']:.2f}</p>
                    <p><b>Total P&L:</b> ${metrics['total_pl']:.2f}</p>
                    <p><b>Return %:</b> {metrics['total_pl_pct']:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        if st.session_state.portfolio:
            st.markdown("---")
            st.markdown("#### 📈 Portfolio Holdings")
            metrics = calculate_portfolio_metrics(st.session_state.portfolio)
            port_df = format_portfolio_dataframe(st.session_state.portfolio, metrics)
            st.dataframe(port_df, use_container_width=True, hide_index=True)
        else:
            st.info("No holdings in portfolio yet")
    
    # ===== TAB 6: WATCHLISTS (Enhanced) =====
    with tab6:
        st.markdown("### ❤️ Custom Watchlists - Enhanced")
        
        col_wl1, col_wl2 = st.columns(2)
        
        with col_wl1:
            st.markdown("#### 📝 Create/Import")
            wl_name = st.text_input("Watchlist Name", placeholder="My Favorites")
            wl_symbols = st.text_area("Symbols (one per line)", placeholder="AAPL\nMSFT\nGOOG")
            
            if st.button("💾 Save Watchlist", use_container_width=True, type="primary"):
                if wl_name and wl_symbols:
                    symbols = [s.strip().upper() for s in wl_symbols.split('\n') if s.strip()]
                    st.session_state.watchlists[wl_name] = symbols
                    st.success(f"✅ Saved '{wl_name}'")
        
        with col_wl2:
            st.markdown("#### 📥 Export/Import CSV")
            
            if st.button("📥 Export as CSV", use_container_width=True):
                csv_data = export_watchlist_csv(st.session_state.watchlists)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"watchlists_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        if st.session_state.watchlists:
            st.markdown("---")
            st.markdown("#### 📚 Saved Watchlists")
            for wl_name, symbols in st.session_state.watchlists.items():
                with st.expander(f"📋 {wl_name} ({len(symbols)} symbols)"):
                    st.write(", ".join(symbols))
    
    # ===== TAB 7: SETTINGS & ALERTS (Enhanced) =====
    with tab7:
        st.markdown("### ⚙️ Settings & Price Alerts - Phase 2A")
        
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            st.markdown("#### 📧 Email Alerts")
            email_enabled = st.checkbox("Enable Email Alerts", value=st.session_state.preferences.get('email_enabled', False))
            if email_enabled:
                alert_email = st.text_input("Alert Email", placeholder="you@example.com")
                alert_type = st.selectbox("Alert Type", ["Price Target", "RSI Extreme", "Volume Spike", "News Alert"])
        
        with col_set2:
            st.markdown("#### 📱 SMS & Desktop")
            sms_enabled = st.checkbox("Enable SMS Alerts (Twilio)", value=st.session_state.preferences.get('sms_enabled', False))
            if sms_enabled:
                phone_number = st.text_input("Phone Number", placeholder="+1 555-0000")
            
            desktop_enabled = st.checkbox("Enable Desktop Alerts", value=st.session_state.preferences.get('desktop_alerts', True))
        
        st.markdown("---")
        st.markdown("#### 🎯 Create Price Alert")
        
        col_alert1, col_alert2, col_alert3 = st.columns(3)
        
        with col_alert1:
            alert_symbol = st.text_input("Symbol", placeholder="AAPL")
        
        with col_alert2:
            alert_price = st.number_input("Target Price", value=100.0, min_value=0.01)
        
        with col_alert3:
            alert_condition = st.selectbox("Condition", ["Above", "Below"])
        
        if st.button("🔔 Create Alert", type="primary", use_container_width=True):
            if alert_symbol and alert_price:
                st.session_state.price_alerts.append({
                    'symbol': alert_symbol,
                    'price': alert_price,
                    'condition': alert_condition,
                    'created': datetime.now()
                })
                st.success(f"✅ Alert created: {alert_symbol} goes {alert_condition} ${alert_price}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p><strong>🌍 Advanced All-Asset Scanner v3 Ultimate</strong></p>
        <p>✨ Phase 2A Features: Dividend Aristocrats • Small Cap Growth • DeFi Tokens • Precious Metals</p>
        <p>✨ Indicators: Bollinger Bands • Fibonacci • MA Crossovers • Volume Spikes</p>
        <p>✨ Features: News Integration • Portfolio Tracking • Price Alerts • CSV Import/Export</p>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
        <p style='font-size: 0.85rem;'>Data: Yahoo Finance • 15-min delay • Free tier</p>
    </div>
    """, unsafe_allow_html=True)

# ===== RUN APP =====
if __name__ == "__main__":
    main()