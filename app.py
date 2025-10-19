import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="All-Asset Scanner", 
    page_icon="🌍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
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
    }
,
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
}
,
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

# ===== SIDEBAR =====
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
    <h1 style='color: white; margin: 0;'>🌍</h1>
    <h3 style='color: white; margin: 0;'>All-Asset Scanner</h3>
    <p style='color: white; margin: 0; font-size: 0.8rem;'>Stocks • Crypto • Forex • Commodities</p>
</div>
""", unsafe_allow_html=True)

# Asset class selector
selected_asset = st.sidebar.selectbox(
    "📂 Select Asset Class",
    list(ASSET_CLASSES.keys()),
    help="Choose which market to scan"
)

asset_config = ASSET_CLASSES[selected_asset]

st.sidebar.info(f"ℹ️ {asset_config['description']}")

st.sidebar.markdown("---")

# Watchlist input
watchlist_input = st.sidebar.text_area(
    "📝 Watchlist (one per line)", 
    value=asset_config["default"],
    height=250,
    help="Enter ticker symbols. One per line."
)

# Parse watchlist
watchlist = [s.strip().upper() for s in watchlist_input.split('\n') if s.strip()]

st.sidebar.markdown(f"**Tracking:** {len(watchlist)} assets")

st.sidebar.markdown("---")

# ===== FILTERS =====
st.sidebar.subheader("🎯 Filters")

max_price = st.sidebar.number_input(
    "💵 Max Price ($)", 
    min_value=asset_config["price_range"][0],
    max_value=asset_config["price_range"][1],
    value=asset_config["max_price"],
    step=asset_config["step"],
    help="Only show assets under this price"
)

max_from_low = st.sidebar.slider(
    "📊 Max % From 52-Week Low", 
    min_value=0, 
    max_value=100, 
    value=50,
    help="Filter by distance from yearly low"
)

min_rsi = st.sidebar.slider(
    "📈 Min RSI (Oversold Filter)",
    min_value=0,
    max_value=100,
    value=0,
    help="0 = disabled, 30 = show oversold only"
)

max_rsi = st.sidebar.slider(
    "📉 Max RSI (Overbought Filter)",
    min_value=0,
    max_value=100,
    value=100,
    help="100 = disabled, 70 = exclude overbought"
)

st.sidebar.markdown("---")

# ===== SCAN BUTTON =====
scan_button = st.sidebar.button(
    "🔍 SCAN NOW", 
    type="primary", 
    use_container_width=True,
    help="Click to refresh data"
)

# Clear cache button
if st.sidebar.button("🗑️ Clear Cache & Reset", use_container_width=True):
    st.session_state.clear()
    st.rerun()

# Auto-refresh option
auto_refresh = st.sidebar.checkbox("♻️ Auto-refresh (every 5 min)")

st.sidebar.markdown("---")

# ===== PRICE ALERTS =====
st.sidebar.markdown("---")
st.sidebar.subheader("🔔 Price Alerts")

alert_symbol = st.sidebar.text_input("Symbol to track", placeholder="BTC-USD")
alert_price = st.sidebar.number_input("Alert when price reaches $", value=0.0)

if alert_symbol and alert_price > 0:
    st.sidebar.info(f"Alert set: {alert_symbol} @ ${alert_price}")

# ===== INFO BOX =====
st.sidebar.markdown(f"""
<div style='background-color: #f0f2f6; padding: 1rem; border-radius: 8px;'>
    <p style='margin: 0; font-size: 0.9rem;'><strong>Last Updated:</strong></p>
    <p style='margin: 0; color: #667eea;'>{datetime.now().strftime('%I:%M:%S %p')}</p>
</div>
""", unsafe_allow_html=True)

# ===== MAIN HEADER =====
st.markdown('<h1 class="main-header">🌍 All-Asset Scanner</h1>', unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: gray; font-size: 1.1rem;'>Real-time analysis of {selected_asset} near 52-week lows</p>", unsafe_allow_html=True)
st.markdown("---")

# ===== FORMATTING FUNCTION =====
def format_dataframe(df):
    """Format DataFrame for nice display"""
    display = df.copy()
    
    # Format currency
    display['Price'] = display['Price'].apply(lambda x: f"${x:.2f}" if x < 1000 else f"${x:,.2f}")
    display['52W Low'] = display['52W Low'].apply(lambda x: f"${x:.2f}" if x < 1000 else f"${x:,.2f}")
    display['52W High'] = display['52W High'].apply(lambda x: f"${x:.2f}" if x < 1000 else f"${x:,.2f}")
    # Format entry/exit prices
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

# ==== SESSION STATE ====
if 'scan_data' not in st.session_state:
    st.session_state.scan_data = {}
    st.session_state.scan_times = {}

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
            # Entry: 3-5% below current price (better entry on pullback)
            entry_low = current_price * 0.95   # 5% below current
            entry_high = current_price * 0.97  # 3% below current
            
            # Stop Loss: Below 52-week low or 10% below entry
            stop_loss_52w = low_52 * 0.98     # 2% below 52W low
            stop_loss_percent = entry_high * 0.90  # 10% below entry
            stop_loss = min(stop_loss_52w, stop_loss_percent)
            
            # Target Exit: Based on alert level
            if pct_from_low < 15:  # Strong Buy
                target_exit = current_price * 1.50  # 50% gain target
            elif pct_from_low < 25:  # Buy
                target_exit = current_price * 1.35  # 35% gain target
            elif pct_from_low < 40:  # Watch
                target_exit = current_price * 1.20  # 20% gain target
            else:  # Neutral
                target_exit = current_price * 1.15  # 15% gain target
            
            # Alternative target: Resistance at 52W high
            target_52w_high = high_52 * 0.95  # 5% below 52W high (resistance)
            
            # Risk/Reward calculation
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
            
        except Exception as e:
            errors.append(f"{symbol}: {str(e)[:40]}")
            continue
        
        # Rate limiting
        time.sleep(1.0)
    
    # Clean up progress
    progress_bar.empty()
    status_text.empty()
    
    # Show errors if any
    if errors and len(errors) <= 5:
        for error in errors:
            st.sidebar.warning(f"⚠️ {error}")
    elif errors:
        st.sidebar.warning(f"⚠️ {len(errors)} symbols had errors")
    
    return pd.DataFrame(results)

# ===== RUN SCAN =====
if scan_button or selected_asset not in st.session_state.scan_data:
    
    if len(watchlist) == 0:
        st.error("⚠️ Please add at least one symbol to your watchlist")
    else:
        scan_start_time = time.time()
        
        with st.spinner(f'🔄 Scanning {len(watchlist)} {selected_asset}... This may take 30-90 seconds'):
            df = scan_assets(watchlist, max_price, max_from_low, min_rsi, max_rsi)
            
            # Store in session state
            st.session_state.scan_data[selected_asset] = df
            st.session_state.scan_times[selected_asset] = datetime.now()
            
            scan_duration = time.time() - scan_start_time
            
            if len(df) > 0:
                st.success(f"✅ Scan complete! Found **{len(df)}** opportunities in {scan_duration:.1f}s")
            else:
                st.warning("⚠️ No assets found matching your filters. Try adjusting criteria.")

# ===== DISPLAY RESULTS =====
if selected_asset in st.session_state.scan_data:
    df = st.session_state.scan_data[selected_asset]
    
    if not df.empty:
        
        # ===== TOP METRICS =====
        st.markdown("### 📊 Quick Stats")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown("""
            <div class='metric-box'>
                <h2 style='margin: 0; color: white;'>{}</h2>
                <p style='margin: 0; color: white;'>Assets Found</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-box'>
                <h2 style='margin: 0; color: white;'>${:.2f}</h2>
                <p style='margin: 0; color: white;'>Avg Price</p>
            </div>
            """.format(df['Price'].mean()), unsafe_allow_html=True)
        
        with col3:
            best_symbol = df.loc[df['% From Low'].idxmin(), 'Symbol']
            st.markdown("""
            <div class='metric-box'>
                <h2 style='margin: 0; color: white;'>{}</h2>
                <p style='margin: 0; color: white;'>Nearest Low</p>
            </div>
            """.format(best_symbol), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='metric-box'>
                <h2 style='margin: 0; color: white;'>{:.1f}%</h2>
                <p style='margin: 0; color: white;'>Avg % From Low</p>
            </div>
            """.format(df['% From Low'].mean()), unsafe_allow_html=True)
        
        with col5:
            strong_buys = len(df[df['Alert'].str.contains('STRONG')])
            st.markdown("""
            <div class='metric-box'>
                <h2 style='margin: 0; color: white;'>{}</h2>
                <p style='margin: 0; color: white;'>Strong Buys</p>
            </div>
            """.format(strong_buys), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ===== HIGHLIGHT OPPORTUNITIES =====
        strong_alerts = df[df['Alert'].str.contains('STRONG')]
        buy_alerts = df[df['Alert'].str.contains('BUY') & ~df['Alert'].str.contains('STRONG')]
        
        if not strong_alerts.empty:
            st.markdown('<div class="alert-strong">', unsafe_allow_html=True)
            st.markdown(f"### 🚨 STRONG BUY SIGNALS ({len(strong_alerts)} found)")
            st.markdown("**These assets are within 15% of their 52-week lows**")
            
            # Display strong buys
            display_strong = strong_alerts.copy()
            display_strong = format_dataframe(display_strong)
            st.dataframe(
                display_strong.drop(['Alert Color'], axis=1), 
                use_container_width=True, 
                hide_index=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("")
        
        if not buy_alerts.empty:
            st.markdown('<div class="alert-buy">', unsafe_allow_html=True)
            st.markdown(f"### ✅ BUY SIGNALS ({len(buy_alerts)} found)")
            st.markdown("**These assets are within 25% of their 52-week lows**")
            
            display_buy = buy_alerts.copy()
            display_buy = format_dataframe(display_buy)
            st.dataframe(
                display_buy.drop(['Alert Color'], axis=1), 
                use_container_width=True, 
                hide_index=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("")
        
        # ===== ALL RESULTS TABLE =====
        st.markdown("### 📋 Complete Results")
        
        # Sort options
        sort_col1, sort_col2 = st.columns([1, 3])
        
        with sort_col1:
            sort_by = st.selectbox(
                "Sort by",
                ['% From Low', 'Price', 'RSI', 'Vol Change %', 'Market Cap']
            )
        
        with sort_col2:
            sort_order = st.radio("Order", ['Ascending', 'Descending'], horizontal=True)
        
        # Apply sorting
        df_sorted = df.sort_values(
            by=sort_by, 
            ascending=(sort_order == 'Ascending')
        )
        
        # Format for display
        display_df = format_dataframe(df_sorted)
        
        # Show table
        st.dataframe(
            display_df.drop(['Alert Color'], axis=1),
            use_container_width=True,
            hide_index=True
        )

# ===== CHART VIEWER =====
if st.checkbox("📈 Show Price Chart"):
    chart_symbol = st.selectbox("Select symbol to chart", df['Symbol'].tolist())
    
    if chart_symbol:
        stock = yf.Ticker(chart_symbol)
        hist = stock.history(period="1y")
        
        st.line_chart(hist['Close'])
        st.caption(f"{chart_symbol} - 1 Year Price History")
        
        # ===== DOWNLOAD SECTION =====
        st.markdown("---")
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            # CSV download
            csv = display_df.drop(['Alert Color'], axis=1).to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
# Excel download
            from io import BytesIO
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                display_df.drop(['Alert Color'], axis=1).to_excel(writer, sheet_name='Scanner Results', index=False)
            
            st.download_button(
                label="📊 Download Excel (.xlsx)",
                data=output.getvalue(),
                file_name=f"scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
                file_name=f"{selected_asset.lower().replace(' ', '_').replace('💎', '').replace('₿', '').replace('💱', '').replace('🥇', '').replace('📊', '').replace('🌏', '').strip()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_download2:
            # Strong buys only CSV
            if not strong_alerts.empty:
                strong_csv = format_dataframe(strong_alerts).drop(['Alert Color'], axis=1).to_csv(index=False)
                st.download_button(
                    label="🚨 Download Strong Buys Only (CSV)",
                    data=strong_csv,
                    file_name=f"strong_buys_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Scan time
        st.caption(f"Last scanned: {st.session_state.scan_times[selected_asset].strftime('%Y-%m-%d %I:%M:%S %p')}")
        
    else:
        st.warning("⚠️ No assets found matching your criteria. Try adjusting the filters.")
        st.info("""
        **Suggestions:**
        - Increase "Max Price" limit
        - Increase "Max % From 52-Week Low" slider
        - Disable RSI filters (set Min to 0, Max to 100)
        - Check if your symbols are correct
        """)

else:
    # ===== WELCOME SCREEN =====
    st.markdown("""
    <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
        <h2>👋 Welcome to the All-Asset Scanner!</h2>
        <p style='font-size: 1.2rem;'>Your one-stop solution for scanning multiple markets</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("""
        ### 🎯 How to Use
        1. Select an asset class
        2. Review/edit the watchlist
        3. Adjust filters as needed
        4. Click **SCAN NOW**
        5. Analyze results
        6. Download CSV
        """)
    
    with col_info2:
        st.markdown("""
        ### 📊 Alert Levels
        - 🚨 **Strong Buy**: < 15% from low
        - ✅ **Buy**: 15-25% from low
        - 👀 **Watch**: 25-40% from low
        - ⚪ **Neutral**: > 40% from low
        """)
    
    with col_info3:
        st.markdown("""
        ### 🔧 Technical Indicators
        - **RSI**: Relative Strength Index
        - **MACD**: Trend direction
        - **Volume**: Trading activity
        - **52W Range**: Yearly price range
        """)
    
    st.markdown("---")
    
    st.info("""
    💡 **Pro Tip:** Start with one asset class, scan it, then switch tabs to scan others. 
    Your results are saved in memory until you refresh the page.
    """)

# ===== FORMATTING FUNCTION =====

# ===== AUTO-REFRESH =====
if auto_refresh:
    time.sleep(300)  # 5 minutes
    st.rerun()
# ===== EMAIL ALERTS =====
if st.sidebar.checkbox("📧 Email Alerts (Beta)"):
    email = st.sidebar.text_input("Your Email")
    
    if email and selected_asset in st.session_state.scan_data:
        df = st.session_state.scan_data[selected_asset]
        strong_buys = df[df['Alert'].str.contains('STRONG')]
        
        if not strong_buys.empty:
            st.sidebar.success(f"🚨 {len(strong_buys)} strong buys found!")
            st.sidebar.info("Email feature coming soon. For now, download the CSV.")

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p><strong>⚠️ Disclaimer</strong></p>
    <p>This tool is for educational and informational purposes only. Not financial advice.</p>
    <p>Always conduct your own research before making investment decisions.</p>
    <p style='margin-top: 1rem; font-size: 0.9rem;'>Data provided by Yahoo Finance • 15-minute delay • Free tier</p>
    <p style='font-size: 0.8rem; color: #999;'>Built with Streamlit • Python • yfinance</p>
</div>
""", unsafe_allow_html=True)
