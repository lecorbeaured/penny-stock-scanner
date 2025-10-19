import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
import hashlib

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Advanced All-Asset Scanner", 
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
if 'preferences' not in st.session_state:
    st.session_state.preferences = {
        'theme': 'dark',
        'data_retention_days': 30,
        'alert_frequency': 'daily'
    }

# ===== ASSET CLASS DEFINITIONS =====
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
    
    "🌏 International": {
        "default": """0700.HK
BABA
TSM
RELIANCE.NS
INFY.NS
TCS.NS
SAP
NVO
ASML
005930.KS
6758.T
7203.T
VALE
PBR
ITUB""",
        "max_price": 500.0,
        "price_range": (0.01, 1000.0),
        "step": 50.0,
        "description": "Global stocks (HK, India, Japan, Korea, Brazil)"
    },
    
    "🎮 Gaming Stocks": {
        "default": """TTWO
EA
ATVI
RBLX
U
PLTK
GMBL
SKLZ
DKNG
PENN""",
        "max_price": 200.0,
        "price_range": (1.0, 500.0),
        "step": 10.0,
        "description": "Gaming, esports, and gambling stocks"
    },
    
    "💰 Dividend Kings": {
        "default": """JNJ
PG
KO
PEP
WMT
MCD
CVX
XOM
MMM
CAT
IBM
T
VZ
ABT
CL""",
        "max_price": 500.0,
        "price_range": (10.0, 1000.0),
        "step": 50.0,
        "description": "High-quality dividend aristocrats"
    }
}

# ===== USER AUTHENTICATION =====
def hash_password(password):
    """Hash password for secure storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    """PLACEHOLDER: Authenticate user against database
    TODO: Implement with streamlit-authenticator or Firebase
    """
    # Placeholder: Check against hardcoded users for MVP
    hardcoded_users = {
        "demo": hash_password("demo123"),
        "user": hash_password("password123")
    }
    
    if username in hardcoded_users:
        if hash_password(password) == hardcoded_users[username]:
            return True
    return False

def login_page():
    """User authentication page"""
    st.markdown('<h1 class="main-header">🌍 All-Asset Scanner Pro</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        username = st.text_input("Username", placeholder="demo")
        password = st.text_input("Password", type="password", placeholder="demo123")
        
        col_login1, col_login2 = st.columns(2)
        
        with col_login1:
            if st.button("🔓 Login", use_container_width=True, type="primary"):
                if authenticate_user(username, password):
                    st.session_state.user = username
                    st.success(f"✅ Welcome, {username}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        with col_login2:
            if st.button("📝 Demo Mode", use_container_width=True):
                st.session_state.user = "guest"
                st.info("Using demo account (read-only)")
                time.sleep(1)
                st.rerun()
        
        st.markdown("---")
        st.info("""
        **Demo Credentials:**
        - Username: `demo`
        - Password: `demo123`
        
        Or click **Demo Mode** to try without login.
        """)

# ===== SIDEBAR =====
def render_sidebar():
    """Render sidebar with controls"""
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
        <h1 style='color: white; margin: 0;'>🌍</h1>
        <h3 style='color: white; margin: 0;'>Advanced Scanner</h3>
        <p style='color: white; margin: 0; font-size: 0.8rem;'>v2.0 - With AI & Backtesting</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User info
    if st.session_state.user:
        st.sidebar.markdown(f"**👤 User:** {st.session_state.user}")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.rerun()
        st.sidebar.markdown("---")
    
    # Asset class selector
    selected_asset = st.sidebar.selectbox(
        "📂 Select Asset Class",
        list(ASSET_CLASSES.keys()),
        help="Choose which market to scan"
    )
    
    asset_config = ASSET_CLASSES[selected_asset]
    st.sidebar.info(f"ℹ️ {asset_config['description']}")
    st.sidebar.markdown("---")
    
    return selected_asset, asset_config

# ===== OPTIONS STRATEGY BUILDER =====
def calculate_options_greeks(stock_price, strike_price, time_to_expiry_days, volatility, risk_free_rate=0.05):
    """PLACEHOLDER: Calculate options Greeks
    TODO: Implement with scipy.stats for Black-Scholes
    
    For now, returns placeholder values based on moneyness
    """
    moneyness = stock_price / strike_price
    intrinsic_value = max(stock_price - strike_price, 0)
    time_value = (moneyness * 0.02) * (time_to_expiry_days / 365) * volatility
    
    # Placeholder Greeks
    delta = min(max(moneyness - 0.95, 0), 1)
    gamma = (0.4 * moneyness) / strike_price
    theta = -time_value / (time_to_expiry_days / 365) if time_to_expiry_days > 0 else 0
    vega = 0.1 * np.sqrt(time_to_expiry_days / 365)
    
    return {
        'option_price': intrinsic_value + time_value,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'intrinsic_value': intrinsic_value,
        'time_value': time_value
    }

def get_options_strategy(symbol, current_price, alert_level="BUY"):
    """Get options trading strategies
    
    PLACEHOLDER: Real implementation would fetch from options chains
    TODO: Integrate with yfinance options data
    """
    
    # Determine strategy based on alert level
    if alert_level == "STRONG BUY":
        strategies = [
            {
                'name': 'Long Call (Bullish)',
                'description': 'Buy calls to profit from upside',
                'risk': 'Limited to premium paid',
                'profit': 'Unlimited',
                'breakeven': current_price
            },
            {
                'name': 'Call Spread (Conservative)',
                'description': 'Buy lower strike, sell higher strike',
                'risk': 'Limited spread width',
                'profit': 'Limited to spread',
                'breakeven': current_price
            },
            {
                'name': 'Straddle (High Volatility)',
                'description': 'Buy both calls and puts at same strike',
                'risk': 'Double premium if flat',
                'profit': 'On large moves',
                'breakeven': current_price
            }
        ]
    else:
        strategies = [
            {
                'name': 'Covered Call (Income)',
                'description': 'Own stock, sell calls for premium',
                'risk': 'Capped upside',
                'profit': 'Premium + dividends',
                'breakeven': current_price
            }
        ]
    
    results = []
    for i, strat in enumerate(strategies):
        strike_price = current_price * (1.0 + (i * 0.05))
        greeks = calculate_options_greeks(
            current_price, 
            strike_price, 
            30,  # 30 days to expiry
            0.3,  # 30% volatility
        )
        
        results.append({
            'strategy': strat['name'],
            'description': strat['description'],
            'strike': strike_price,
            'premium': greeks['option_price'],
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'theta': greeks['theta'],
            'vega': greeks['vega'],
            'risk': strat['risk'],
            'profit': strat['profit'],
            'leverage': 5 + (i * 2)
        })
    
    return results

# ===== BACKTESTING ENGINE =====
def backtest_strategy(symbol, start_date, end_date, entry_threshold=15, exit_threshold=50):
    """PLACEHOLDER: Backtest strategy on historical data
    TODO: Implement full backtesting with transaction costs, slippage
    """
    
    try:
        # Fetch historical data
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return None
        
        # Calculate metrics
        high_52w = data['Close'].tail(252).max()
        low_52w = data['Close'].tail(252).min()
        
        results = {
            'symbol': symbol,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'period_days': (end_date - start_date).days,
            'trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'avg_profit_per_trade': 0,
            'best_trade': 0,
            'worst_trade': 0
        }
        
        # Placeholder: Calculate some basic stats
        daily_returns = data['Close'].pct_change()
        results['total_return'] = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
        results['max_drawdown'] = ((data['Close'].rolling(252).min() - data['Close']) / data['Close']).min() * 100
        results['sharpe_ratio'] = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        return results
        
    except Exception as e:
        st.warning(f"Backtest error: {str(e)}")
        return None

# ===== SIGNAL HISTORY & DATABASE =====
def save_signal_history(symbol, alert_level, price, date):
    """Save signal to history for tracking"""
    signal = {
        'symbol': symbol,
        'alert': alert_level,
        'price': price,
        'date': date,
        'timestamp': datetime.now().isoformat()
    }
    st.session_state.signal_history.append(signal)

def get_signal_performance(symbol, days=30):
    """PLACEHOLDER: Calculate signal performance over time
    TODO: Compare signal entry prices to actual outcomes
    """
    signals = [s for s in st.session_state.signal_history if s['symbol'] == symbol]
    
    if not signals:
        return None
    
    recent_signals = [s for s in signals if (datetime.now() - datetime.fromisoformat(s['timestamp'])).days <= days]
    
    return {
        'total_signals': len(recent_signals),
        'strong_buy_count': len([s for s in recent_signals if 'STRONG' in s['alert']]),
        'avg_signal_price': np.mean([s['price'] for s in recent_signals]),
        'signal_accuracy': 0.65  # Placeholder: 65% win rate
    }

# ===== EMAIL ALERTS =====
def send_email_alert(email, symbol, alert_level, price):
    """PLACEHOLDER: Send email alert via SendGrid
    TODO: Implement with SendGrid API
    
    Email body template:
    Subject: 🚨 {alert_level} - {symbol} @ ${price}
    Body: Asset {symbol} triggered a {alert_level} signal...
    """
    
    # This would call SendGrid API
    # For now, just log it
    if email not in st.session_state.email_alerts:
        st.session_state.email_alerts[email] = []
    
    st.session_state.email_alerts[email].append({
        'symbol': symbol,
        'alert': alert_level,
        'price': price,
        'sent_at': datetime.now().isoformat()
    })
    
    return True

# ===== USER PREFERENCES & PERSISTENCE =====
def save_user_preferences(username):
    """PLACEHOLDER: Save user preferences to database
    TODO: Implement with Firebase or MongoDB
    """
    prefs_file = f"user_prefs_{username}.json"
    with open(prefs_file, 'w') as f:
        json.dump(st.session_state.preferences, f)

def load_user_preferences(username):
    """PLACEHOLDER: Load user preferences from database"""
    prefs_file = f"user_prefs_{username}.json"
    if os.path.exists(prefs_file):
        with open(prefs_file, 'r') as f:
            return json.load(f)
    return st.session_state.preferences

# ===== CUSTOM WATCHLISTS =====
def save_watchlist(watchlist_name, symbols):
    """Save custom watchlist"""
    st.session_state.watchlists[watchlist_name] = symbols
    st.success(f"✅ Watchlist '{watchlist_name}' saved!")

def load_watchlist(watchlist_name):
    """Load saved watchlist"""
    if watchlist_name in st.session_state.watchlists:
        return st.session_state.watchlists[watchlist_name]
    return None

def delete_watchlist(watchlist_name):
    """Delete watchlist"""
    if watchlist_name in st.session_state.watchlists:
        del st.session_state.watchlists[watchlist_name]
        st.success(f"✅ Watchlist '{watchlist_name}' deleted!")

# ===== FORMATTING FUNCTION =====
def format_dataframe(df):
    """Format DataFrame for nice display"""
    display = df.copy()
    
    # Format currency
    display['Price'] = display['Price'].apply(lambda x: f"${x:.2f}" if x < 1000 else f"${x:,.2f}")
    display['52W Low'] = display['52W Low'].apply(lambda x: f"${x:.2f}" if x < 1000 else f"${x:,.2f}")
    display['52W High'] = display['52W High'].apply(lambda x: f"${x:.2f}" if x < 1000 else f"${x:,.2f}")
    display['Entry Low'] = display['Entry Low'].apply(lambda x: f"${x:.2f}" if x < 1000 else f"${x:,.2f}")
    display['Entry High'] = display['Entry High'].apply(lambda x: f"${x:.2f}" if x < 1000 else f"${x:,.2f}")
    display['Stop Loss'] = display['Stop Loss'].apply(lambda x: f"${x:.2f}" if x < 1000 else f"${x:,.2f}")
    display['Target Exit'] = display['Target Exit'].apply(lambda x: f"${x:.2f}" if x < 1000 else f"${x:,.2f}")
    display['Alt Target'] = display['Alt Target'].apply(lambda x: f"${x:.2f}" if x < 1000 else f"${x:,.2f}")
    display['Risk/Reward'] = display['Risk/Reward'].apply(lambda x: f"{x:.2f}x")

    # Format percentages
    display['% From Low'] = display['% From Low'].apply(lambda x: f"{x:.1f}%")
    display['% From High'] = display['% From High'].apply(lambda x: f"{x:.1f}%")
    display['Vol Change %'] = display['Vol Change %'].apply(lambda x: f"+{x:.0f}%" if x > 0 else f"{x:.0f}%")
    
    # Format large numbers
    display['Market Cap'] = display['Market Cap'].apply(lambda x: f"${x:.0f}M" if x > 0 else "N/A")
    display['Volume'] = display['Volume'].apply(lambda x: f"{x:,.0f}")
    display['Avg Volume'] = display['Avg Volume'].apply(lambda x: f"{x:,.0f}")
    
    # Format RSI
    display['RSI'] = display['RSI'].apply(lambda x: f"{x:.1f}")
    
    # Reorder columns
    column_order = [
        'Symbol', 'Price', 'Alert', 
        'Entry Low', 'Entry High', 'Stop Loss', 'Target Exit', 'Alt Target', 'Risk/Reward',
        'RSI Signal', 'MACD',
        '52W Low', '52W High', '% From Low', '% From High',
        'Volume', 'Avg Volume', 'Vol Change %',
        'RSI', 'Market Cap', 'Alert Color'
    ]
    
    return display[column_order]

# ===== SCANNING FUNCTION =====
def scan_assets(symbols, max_price, max_from_low, min_rsi, max_rsi):
    """Main scanning engine"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(symbols)
    errors = []
    
    for i, symbol in enumerate(symbols):
        status_text.text(f"🔍 Scanning {symbol}... ({i+1}/{total})")
        progress_bar.progress((i + 1) / total)
        
        try:
            # Fetch data
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            
            if hist.empty:
                errors.append(f"{symbol}: No data available")
                continue
            
            # Basic data
            current_price = hist['Close'].iloc[-1]
            
            # Price filter
            if current_price > max_price:
                continue
            
            # 52-week range
            high_52 = hist['High'].max()
            low_52 = hist['Low'].min()
            
            # Calculate percentages
            pct_from_low = ((current_price - low_52) / low_52) * 100
            pct_from_high = ((current_price - high_52) / high_52) * 100
            
            # Filter by % from low
            if pct_from_low > max_from_low:
                continue
            
            # Volume data
            current_volume = hist['Volume'].iloc[-1]
            avg_volume_3m = hist['Volume'].tail(60).mean()
            volume_change = ((current_volume - avg_volume_3m) / avg_volume_3m) * 100
            
            # RSI calculation
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # RSI filter
            if current_rsi < min_rsi or current_rsi > max_rsi:
                continue
            
            # MACD calculation
            exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_histogram = macd - signal
            current_macd_hist = macd_histogram.iloc[-1]
            
            # Get market cap (if available)
            try:
                info = stock.info
                market_cap = info.get('marketCap', 0) / 1_000_000
            except:
                market_cap = 0
            
            # Determine alert level
            if pct_from_low < 15:
                alert = "🚨 STRONG BUY"
                alert_color = "strong"
            elif pct_from_low < 25:
                alert = "✅ BUY"
                alert_color = "buy"
            elif pct_from_low < 40:
                alert = "👀 WATCH"
                alert_color = "watch"
            else:
                alert = "⚪ NEUTRAL"
                alert_color = "neutral"
            
            # RSI signal
            if current_rsi < 30:
                rsi_signal = "🟢 Oversold"
            elif current_rsi > 70:
                rsi_signal = "🔴 Overbought"
            else:
                rsi_signal = "⚪ Neutral"
            
            # MACD signal
            if current_macd_hist > 0:
                macd_signal = "🟢 Bullish"
            else:
                macd_signal = "🔴 Bearish"
            
            # Calculate Entry/Exit/Stop Loss
            entry_low = current_price * 0.95
            entry_high = current_price * 0.97
            
            stop_loss_52w = low_52 * 0.98
            stop_loss_percent = entry_high * 0.90
            stop_loss = min(stop_loss_52w, stop_loss_percent)
            
            if pct_from_low < 15:
                target_exit = current_price * 1.50
            elif pct_from_low < 25:
                target_exit = current_price * 1.35
            elif pct_from_low < 40:
                target_exit = current_price * 1.20
            else:
                target_exit = current_price * 1.15
            
            target_52w_high = high_52 * 0.95
            
            risk = entry_high - stop_loss
            reward = target_exit - entry_high
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Store results
            results.append({
                'Symbol': symbol,
                'Price': current_price,
                'Entry Low': entry_low,
                'Entry High': entry_high,
                'Stop Loss': stop_loss,
                'Target Exit': target_exit,
                'Alt Target': target_52w_high,
                'Risk/Reward': risk_reward_ratio,
                '52W Low': low_52,
                '52W High': high_52,
                '% From Low': pct_from_low,
                '% From High': pct_from_high,
                'Market Cap': market_cap,
                'Volume': current_volume,
                'Avg Volume': avg_volume_3m,
                'Vol Change %': volume_change,
                'RSI': current_rsi,
                'RSI Signal': rsi_signal,
                'MACD': macd_signal,
                'Alert': alert,
                'Alert Color': alert_color
            })
            
            # Save to signal history
            save_signal_history(symbol, alert, current_price, datetime.now())
            
        except Exception as e:
            errors.append(f"{symbol}: {str(e)[:40]}")
            continue
        
        time.sleep(1.0)
    
    progress_bar.empty()
    status_text.empty()
    
    if errors and len(errors) <= 5:
        for error in errors:
            st.sidebar.warning(f"⚠️ {error}")
    elif errors:
        st.sidebar.warning(f"⚠️ {len(errors)} symbols had errors")
    
    return pd.DataFrame(results)

# ===== MAIN APP =====
def main():
    """Main application flow"""
    
    # Check authentication
    if not st.session_state.user:
        login_page()
        return
    
    # Get sidebar controls
    selected_asset, asset_config = render_sidebar()
    
    # Main header
    st.markdown('<h1 class="main-header">🌍 Advanced All-Asset Scanner</h1>', unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: gray; font-size: 1.1rem;'>v2.0 - With Options, Backtesting & AI</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Scanner",
        "📊 Backtest",
        "🎯 Options",
        "📈 Signals",
        "⚙️ Settings",
        "❤️ Watchlists"
    ])
    
    # ===== TAB 1: SCANNER =====
    with tab1:
        st.markdown("### 🔍 Scan for Opportunities")
        
        # Filters in sidebar
        st.sidebar.subheader("🎯 Filters")
        
        max_price = st.sidebar.number_input(
            "💵 Max Price ($)", 
            min_value=asset_config["price_range"][0],
            max_value=asset_config["price_range"][1],
            value=asset_config["max_price"],
            step=asset_config["step"]
        )
        
        max_from_low = st.sidebar.slider(
            "📊 Max % From 52-Week Low", 
            min_value=0, max_value=100, value=50
        )
        
        min_rsi = st.sidebar.slider(
            "📈 Min RSI (Oversold)", 
            min_value=0, max_value=100, value=0
        )
        
        max_rsi = st.sidebar.slider(
            "📉 Max RSI (Overbought)", 
            min_value=0, max_value=100, value=100
        )
        
        # Watchlist
        st.sidebar.markdown("---")
        watchlist_input = st.sidebar.text_area(
            "📝 Watchlist (one per line)", 
            value=asset_config["default"],
            height=150
        )
        watchlist = [s.strip().upper() for s in watchlist_input.split('\n') if s.strip()]
        
        st.sidebar.markdown(f"**Tracking:** {len(watchlist)} assets")
        
        # Scan button
        st.sidebar.markdown("---")
        scan_button = st.sidebar.button("🔍 SCAN NOW", type="primary", use_container_width=True)
        
        if st.sidebar.button("🗑️ Clear Cache", use_container_width=True):
            st.session_state.scan_data = {}
            st.rerun()
        
        # Run scan
        if scan_button:
            if len(watchlist) == 0:
                st.error("⚠️ Please add at least one symbol")
            else:
                with st.spinner("🔄 Scanning..."):
                    df = scan_assets(watchlist, max_price, max_from_low, min_rsi, max_rsi)
                    st.session_state.scan_data[selected_asset] = df
                    st.session_state.scan_times[selected_asset] = datetime.now()
                    
                    if len(df) > 0:
                        st.success(f"✅ Found **{len(df)}** opportunities!")
                    else:
                        st.warning("⚠️ No results found. Try adjusting filters.")
        
        # Display results
        if selected_asset in st.session_state.scan_data:
            df = st.session_state.scan_data[selected_asset]
            
            if not df.empty:
                # Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <h2 style='margin: 0; color: white;'>{len(df)}</h2>
                        <p style='margin: 0; color: white;'>Found</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <h2 style='margin: 0; color: white;'>${df['Price'].mean():.2f}</h2>
                        <p style='margin: 0; color: white;'>Avg Price</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    best_symbol = df.loc[df['% From Low'].idxmin(), 'Symbol']
                    st.markdown(f"""
                    <div class='metric-box'>
                        <h2 style='margin: 0; color: white;'>{best_symbol}</h2>
                        <p style='margin: 0; color: white;'>Best Signal</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <h2 style='margin: 0; color: white;'>{df['% From Low'].mean():.1f}%</h2>
                        <p style='margin: 0; color: white;'>Avg % From Low</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    strong_buys = len(df[df['Alert'].str.contains('STRONG')])
                    st.markdown(f"""
                    <div class='metric-box'>
                        <h2 style='margin: 0; color: white;'>{strong_buys}</h2>
                        <p style='margin: 0; color: white;'>Strong Buys</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Results table
                st.markdown("### 📋 Results")
                
                display_df = format_dataframe(df)
                st.dataframe(
                    display_df.drop(['Alert Color'], axis=1),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Export buttons
                st.markdown("---")
                st.markdown("### 📥 Export")
                
                col_exp1, col_exp2, col_exp3 = st.columns(3)
                
                with col_exp1:
                    csv = display_df.drop(['Alert Color'], axis=1).to_csv(index=False)
                    st.download_button(
                        label="📥 CSV",
                        data=csv,
                        file_name=f"scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_exp2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        display_df.drop(['Alert Color'], axis=1).to_excel(writer, sheet_name='Scan', index=False)
                    
                    st.download_button(
                        label="📊 Excel",
                        data=output.getvalue(),
                        file_name=f"scan_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with col_exp3:
                    strong_buys = df[df['Alert'].str.contains('STRONG')]
                    if not strong_buys.empty:
                        strong_csv = format_dataframe(strong_buys).drop(['Alert Color'], axis=1).to_csv(index=False)
                        st.download_button(
                            label="🚨 Strong Buys",
                            data=strong_csv,
                            file_name=f"strong_buys_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
    
    # ===== TAB 2: BACKTEST =====
    with tab2:
        st.markdown("### 📊 Strategy Backtest")
        st.markdown("*PHASE 2: Test your strategy on historical data*")
        
        col_bt1, col_bt2 = st.columns(2)
        
        with col_bt1:
            bt_symbol = st.text_input("Symbol to backtest", value="SPY")
        
        with col_bt2:
            bt_days = st.number_input("Days of history", value=365, min_value=30)
        
        if st.button("▶️ Run Backtest", type="primary"):
            with st.spinner("🔄 Backtesting..."):
                end_date = datetime.now()
                start_date = end_date - timedelta(days=bt_days)
                
                results = backtest_strategy(bt_symbol, start_date, end_date)
                
                if results:
                    st.session_state.backtest_results[bt_symbol] = results
                    
                    st.markdown("---")
                    
                    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                    
                    with col_res1:
                        st.markdown(f"""
                        <div class='backtest-box'>
                            <h3 style='margin: 0;'>{results['total_return']:.1f}%</h3>
                            <p style='margin: 0; font-size: 0.9rem;'>Total Return</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res2:
                        st.markdown(f"""
                        <div class='backtest-box'>
                            <h3 style='margin: 0;'>{results['sharpe_ratio']:.2f}</h3>
                            <p style='margin: 0; font-size: 0.9rem;'>Sharpe Ratio</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res3:
                        st.markdown(f"""
                        <div class='backtest-box'>
                            <h3 style='margin: 0;'>{results['max_drawdown']:.1f}%</h3>
                            <p style='margin: 0; font-size: 0.9rem;'>Max Drawdown</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res4:
                        st.markdown(f"""
                        <div class='backtest-box'>
                            <h3 style='margin: 0;'>{results['win_rate']:.0f}%</h3>
                            <p style='margin: 0; font-size: 0.9rem;'>Win Rate</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.info("📌 **Placeholder Data**: Full backtesting engine coming in Phase 2")
    
    # ===== TAB 3: OPTIONS =====
    with tab3:
        st.markdown("### 🎯 Options Strategy Builder")
        st.markdown("*PHASE 2: Calculate Greeks and find best strategies*")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            opt_symbol = st.text_input("Symbol", value="SPY")
            opt_current = st.number_input("Current Price ($)", value=100.0, min_value=0.01)
        
        with col_opt2:
            opt_alert = st.selectbox("Alert Level", ["STRONG BUY", "BUY", "WATCH"])
        
        if st.button("🎯 Analyze Options", type="primary"):
            with st.spinner("Calculating Greeks..."):
                strategies = get_options_strategy(opt_symbol, opt_current, opt_alert)
                
                for strat in strategies:
                    with st.expander(f"📊 {strat['strategy']} - Delta: {strat['delta']:.2f}"):
                        col_s1, col_s2 = st.columns(2)
                        
                        with col_s1:
                            st.markdown(f"""
                            <div class='options-box'>
                                <p><b>Premium:</b> ${strat['premium']:.2f}</p>
                                <p><b>Strike:</b> ${strat['strike']:.2f}</p>
                                <p><b>Leverage:</b> {strat['leverage']}x</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_s2:
                            st.markdown(f"""
                            <div class='options-box'>
                                <p><b>Delta:</b> {strat['delta']:.3f}</p>
                                <p><b>Gamma:</b> {strat['gamma']:.3f}</p>
                                <p><b>Theta:</b> {strat['theta']:.3f}</p>
                                <p><b>Vega:</b> {strat['vega']:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.write(f"**Risk:** {strat['risk']}")
                        st.write(f"**Profit Potential:** {strat['profit']}")
        
        st.info("📌 **Placeholder Greeks**: Full options pricing (Black-Scholes) coming in Phase 2")
    
    # ===== TAB 4: SIGNALS =====
    with tab4:
        st.markdown("### 📈 Signal Performance History")
        st.markdown("*Track your signal accuracy over time*")
        
        if st.session_state.signal_history:
            signals_df = pd.DataFrame(st.session_state.signal_history)
            
            col_sig1, col_sig2 = st.columns(2)
            
            with col_sig1:
                st.metric("Total Signals", len(signals_df))
            
            with col_sig2:
                strong_count = len(signals_df[signals_df['alert'].str.contains('STRONG', na=False)])
                st.metric("Strong Buys", strong_count)
            
            st.markdown("---")
            st.markdown("### Recent Signals")
            
            signals_display = signals_df.sort_values('timestamp', ascending=False).head(10).copy()
            signals_display['date'] = pd.to_datetime(signals_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                signals_display[['symbol', 'alert', 'price', 'date']].rename(
                    columns={'symbol': 'Symbol', 'alert': 'Alert', 'price': 'Price', 'date': 'Time'}
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("📌 No signals recorded yet. Run a scan to start tracking!")
    
    # ===== TAB 5: SETTINGS =====
    with tab5:
        st.markdown("### ⚙️ User Settings")
        
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            st.markdown("#### 🎨 Display")
            theme = st.selectbox("Theme", ["Dark", "Light"], index=0)
            st.session_state.preferences['theme'] = theme
        
        with col_set2:
            st.markdown("#### 📧 Alerts")
            alert_email = st.text_input("Alert Email (PHASE 2)", placeholder="your@email.com")
            alert_freq = st.selectbox("Alert Frequency", ["Real-time", "Daily", "Weekly"])
            st.session_state.preferences['alert_frequency'] = alert_freq
        
        st.markdown("---")
        st.markdown("#### 📊 Data")
        
        col_dat1, col_dat2 = st.columns(2)
        
        with col_dat1:
            data_retention = st.slider("Data Retention (days)", 1, 365, 30)
            st.session_state.preferences['data_retention_days'] = data_retention
        
        with col_dat2:
            if st.button("💾 Save Preferences", use_container_width=True):
                save_user_preferences(st.session_state.user)
                st.success("✅ Preferences saved!")
        
        st.markdown("---")
        st.markdown("#### 📱 API Integrations (PHASE 2)")
        
        col_api1, col_api2 = st.columns(2)
        
        with col_api1:
            st.markdown("**Polygon.io** (Real-time data)")
            polygon_key = st.text_input("API Key", type="password", placeholder="pk_...")
            st.caption("$29/month for real-time data")
        
        with col_api2:
            st.markdown("**SendGrid** (Email alerts)")
            sendgrid_key = st.text_input("API Key", type="password", placeholder="SG...")
            st.caption("$20/month for emails")
    
    # ===== TAB 6: WATCHLISTS =====
    with tab6:
        st.markdown("### ❤️ Custom Watchlists")
        
        col_wl1, col_wl2 = st.columns(2)
        
        with col_wl1:
            st.markdown("#### 📝 Create Watchlist")
            wl_name = st.text_input("Watchlist Name", placeholder="My Favorites")
            wl_symbols = st.text_area("Symbols (one per line)", placeholder="AAPL\nMSFT\nGOOG")
            
            if st.button("💾 Save Watchlist", use_container_width=True, type="primary"):
                if wl_name and wl_symbols:
                    symbols = [s.strip().upper() for s in wl_symbols.split('\n') if s.strip()]
                    save_watchlist(wl_name, symbols)
        
        with col_wl2:
            st.markdown("#### 📚 Saved Watchlists")
            
            if st.session_state.watchlists:
                for wl_name, symbols in st.session_state.watchlists.items():
                    with st.expander(f"📋 {wl_name} ({len(symbols)} symbols)"):
                        st.write(", ".join(symbols))
                        if st.button(f"🗑️ Delete {wl_name}", key=f"del_{wl_name}", use_container_width=True):
                            delete_watchlist(wl_name)
                            st.rerun()
            else:
                st.info("No watchlists saved yet")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p><strong>🌍 Advanced All-Asset Scanner v2.0</strong></p>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
        <p style='font-size: 0.85rem;'>Data: Yahoo Finance • 15-min delay • Free tier</p>
        <p style='font-size: 0.8rem; color: #999;'>Features: Scanner • Options • Backtest • Signals • Auth • Watchlists</p>
    </div>
    """, unsafe_allow_html=True)

# ===== RUN APP =====
if __name__ == "__main__":
    main()