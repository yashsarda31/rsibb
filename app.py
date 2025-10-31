import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NIFTY 50 Scanner - BB & RSI Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Card styling */
    .signal-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .buy-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .sell-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    
    .watch-card {
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 14px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border-radius: 5px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: white;
        border-radius: 5px 5px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Stock symbols
NIFTY_50_SYMBOLS = [
    '^NSEI', '^NSEBANK', 'ACC.NS', 'ADANIENT.NS', 'ADANIPORTS.NS',
    'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AUROPHARMA.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
    'BAJFINANCE.NS', 'BANDHANBNK.NS', 'BERGEPAINT.NS', 'BHARTIARTL.NS', 'BIOCON.NS',
    'BOSCHLTD.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'COALINDIA.NS',
    'COLPAL.NS', 'DLF.NS', 'DABUR.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
    'GRASIM.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS',
    'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'PERSISTENT.NS',
    'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'SAIL.NS',
    'ETERNAL.NS', 'M&M.NS', 'MARICO.NS', 'MARUTI.NS', 'MUTHOOTFIN.NS', 'NTPC.NS',
    'NESTLEIND.NS', 'ONGC.NS', 'PETRONET.NS', 'PIDILITIND.NS', 'POWERGRID.NS',
    'PGHH.NS', 'PNB.NS', 'TRENT.NS', 'SBICARD.NS', 'SBILIFE.NS', 'SHREECEM.NS',
    'ABB.NS', 'SIEMENS.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS',
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TORNTPHARM.NS', 'UPL.NS',
    'ULTRACEMCO.NS', 'VEDL.NS', 'WIPRO.NS', 'YESBANK.NS', 'INDIGO.NS', 'NAUKRI.NS',
    'NYKAA.NS', 'PAYTM.NS', 'POLICYBZR.NS', 'BHEL.NS', 'VMM.NS', 'GC=F', 'SI=F'
]

class BollingerRSIScanner:
    def __init__(self, symbols, config):
        self.symbols = symbols
        self.config = config
        self.signal_history = self.load_signal_history()
        
    def load_signal_history(self):
        if os.path.exists('signal_history.json'):
            try:
                with open('signal_history.json', 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_signal_history(self, signals):
        timestamp = datetime.now().isoformat()
        entry = {
            'timestamp': timestamp,
            'signals': signals
        }
        self.signal_history.append(entry)
        
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]
        
        try:
            with open('signal_history.json', 'w') as f:
                json.dump(self.signal_history, f, indent=2)
        except Exception as e:
            st.error(f"Error saving signal history: {e}")
    
    def calculate_rsi(self, data, period=14):
        if len(data) < period:
            return pd.Series(index=data.index, dtype=float)
        
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(com=period-1, adjust=True, min_periods=period).mean()
        avg_loss = loss.ewm(com=period-1, adjust=True, min_periods=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, data, period=20, std_multiplier=2):
        if len(data) < period:
            return pd.Series(index=data.index), pd.Series(index=data.index), pd.Series(index=data.index)
        
        middle_band = data.rolling(window=period, min_periods=period).mean()
        std = data.rolling(window=period, min_periods=period).std()
        upper_band = middle_band + (std_multiplier * std)
        lower_band = middle_band - (std_multiplier * std)
        
        return upper_band, middle_band, lower_band
    
    def calculate_volume_ratio(self, volume_data):
        if len(volume_data) < 20:
            return 1.0
        
        avg_volume = volume_data.iloc[-20:-1].mean()
        current_volume = volume_data.iloc[-1]
        
        if avg_volume == 0:
            return 1.0
            
        return current_volume / avg_volume
    
    def fetch_stock_data(self, symbol, retries=3):
        for attempt in range(retries):
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(
                    period=self.config['data_period'], 
                    interval=self.config['data_interval']
                )
                
                if len(data) < max(self.config['bb_period'], self.config['rsi_period']) + 5:
                    data = stock.history(period='1mo', interval=self.config['data_interval'])
                
                if not data.empty:
                    return data
                    
            except Exception as e:
                if attempt == retries - 1:
                    st.error(f"Error fetching {symbol}: {str(e)}")
                time.sleep(0.5)
        
        return None
    
    def calculate_signal_strength(self, price, rsi, upper_bb, lower_bb):
        strength = 0
        
        if price <= lower_bb:
            below_pct = ((lower_bb - price) / lower_bb) * 100
            strength = min(100, below_pct * 20)
        elif price >= upper_bb:
            above_pct = ((price - upper_bb) / upper_bb) * 100
            strength = min(100, above_pct * 20)
        
        if rsi < self.config['rsi_oversold']:
            rsi_strength = (self.config['rsi_oversold'] - rsi) * 3.33
            strength = (strength + rsi_strength) / 2
        elif rsi > self.config['rsi_overbought']:
            rsi_strength = (rsi - self.config['rsi_overbought']) * 3.33
            strength = (strength + rsi_strength) / 2
        
        return min(100, max(0, strength))
    
    def get_bb_position(self, price, upper_bb, lower_bb):
        bb_width = upper_bb - lower_bb
        if bb_width == 0:
            return 50.0
        
        position = ((price - lower_bb) / bb_width) * 100
        return round(position, 2)
    
    def check_signals(self, symbol):
        data = self.fetch_stock_data(symbol)
        
        if data is None or len(data) < max(self.config['bb_period'], self.config['rsi_period']) + 1:
            return None
        
        try:
            close_prices = data['Close']
            volume = data['Volume']
            
            rsi = self.calculate_rsi(close_prices, self.config['rsi_period'])
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(
                close_prices, self.config['bb_period'], self.config['bb_std']
            )
            
            current_price = close_prices.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_upper_bb = upper_band.iloc[-1]
            current_lower_bb = lower_band.iloc[-1]
            current_middle_bb = middle_band.iloc[-1]
            
            if pd.isna(current_rsi) or pd.isna(current_upper_bb) or pd.isna(current_lower_bb):
                return None
            
            volume_ratio = self.calculate_volume_ratio(volume)
            
            signal = None
            signal_strength = 0
            
            if (current_price <= current_lower_bb) and (current_rsi < self.config['rsi_oversold']):
                if not self.config['enable_volume_confirmation'] or volume_ratio >= self.config['volume_threshold']:
                    signal = 'BUY'
                    signal_strength = self.calculate_signal_strength(
                        current_price, current_rsi, current_upper_bb, current_lower_bb
                    )
            
            elif (current_price >= current_upper_bb) and (current_rsi > self.config['rsi_overbought']):
                if not self.config['enable_volume_confirmation'] or volume_ratio >= self.config['volume_threshold']:
                    signal = 'SELL'
                    signal_strength = self.calculate_signal_strength(
                        current_price, current_rsi, current_upper_bb, current_lower_bb
                    )
            
            bb_position = self.get_bb_position(current_price, current_upper_bb, current_lower_bb)
            
            distance_from_lower = ((current_price - current_lower_bb) / current_price) * 100
            distance_from_upper = ((current_upper_bb - current_price) / current_price) * 100
            
            return {
                'symbol': symbol.replace('.NS', ''),
                'current_price': round(current_price, 2),
                'rsi': round(current_rsi, 2),
                'upper_bb': round(current_upper_bb, 2),
                'middle_bb': round(current_middle_bb, 2),
                'lower_bb': round(current_lower_bb, 2),
                'signal': signal,
                'signal_strength': round(signal_strength, 1),
                'bb_position': bb_position,
                'volume_ratio': round(volume_ratio, 2),
                'distance_from_lower': round(distance_from_lower, 2),
                'distance_from_upper': round(distance_from_upper, 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data': data  # Store for charting
            }
            
        except Exception as e:
            st.error(f"Error processing {symbol}: {str(e)}")
            return None
    
    def scan_all_stocks(self, progress_bar, status_text):
        results = []
        total = len(self.symbols)
        
        for i, symbol in enumerate(self.symbols):
            result = self.check_signals(symbol)
            if result:
                results.append(result)
            
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f'Scanning {symbol} ({i+1}/{total})')
        
        buy_signals = [r for r in results if r['signal'] == 'BUY']
        sell_signals = [r for r in results if r['signal'] == 'SELL']
        watch_list = []
        
        for r in results:
            if r['signal'] is None:
                if (25 < r['rsi'] < 35) or (65 < r['rsi'] < 75):
                    watch_list.append(r)
                elif r['distance_from_lower'] < 2 or r['distance_from_upper'] < 2:
                    watch_list.append(r)
        
        buy_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
        sell_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
        
        if buy_signals or sell_signals:
            self.save_signal_history(buy_signals + sell_signals)
        
        return buy_signals, sell_signals, watch_list

def create_technical_chart(stock_data, symbol, bb_data, rsi_data):
    """Create interactive technical analysis chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} - Price & Bollinger Bands', 'RSI', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=bb_data['upper'],
            name='Upper BB',
            line=dict(color='rgba(250, 128, 114, 0.5)', width=1),
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=bb_data['middle'],
            name='Middle BB',
            line=dict(color='rgba(169, 169, 169, 0.5)', width=1, dash='dash'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=bb_data['lower'],
            name='Lower BB',
            line=dict(color='rgba(144, 238, 144, 0.5)', width=1),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.1)',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=rsi_data,
            name='RSI',
            line=dict(color='purple', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # RSI thresholds
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, row=2, col=1)
    
    # Volume
    colors = ['red' if stock_data['Close'].iloc[i] < stock_data['Open'].iloc[i] 
              else 'green' for i in range(len(stock_data))]
    
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig

def display_signal_card(signal, signal_type):
    """Display a beautiful signal card"""
    card_class = "buy-card" if signal_type == "BUY" else "sell-card"
    icon = "🟢" if signal_type == "BUY" else "🔴"
    strength_bar = "🔥" * int(signal['signal_strength'] / 20)
    
    html = f"""
    <div class="signal-card {card_class}">
        <h3>{icon} {signal['symbol']}</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div>
                <small>Current Price</small>
                <h4>₹{signal['current_price']}</h4>
            </div>
            <div>
                <small>RSI</small>
                <h4>{signal['rsi']}</h4>
            </div>
            <div>
                <small>Signal Strength</small>
                <h4>{signal['signal_strength']}% {strength_bar}</h4>
            </div>
            <div>
                <small>Volume Ratio</small>
                <h4>{signal['volume_ratio']}x</h4>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("📈 NIFTY 50 Scanner")
        st.markdown("**Bollinger Bands + RSI Strategy**")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        config = {
            'data_period': st.selectbox("Data Period", ['1d', '5d', '1mo', '3mo'], index=1),
            'data_interval': st.selectbox("Interval", ['1m', '5m', '15m', '30m', '1h', '1d'], index=4),
            'bb_period': st.number_input("BB Period", min_value=10, max_value=50, value=20),
            'bb_std': st.number_input("BB Std Dev", min_value=1.0, max_value=3.0, value=2.0, step=0.5),
            'rsi_period': st.number_input("RSI Period", min_value=5, max_value=30, value=14),
            'rsi_oversold': st.number_input("RSI Oversold", min_value=10, max_value=40, value=30),
            'rsi_overbought': st.number_input("RSI Overbought", min_value=60, max_value=90, value=70),
            'volume_threshold': st.number_input("Volume Threshold", min_value=1.0, max_value=3.0, value=1.5, step=0.1),
            'enable_volume_confirmation': st.checkbox("Enable Volume Confirmation", value=True)
        }
        
        st.markdown("---")
        
        # Stock selection
        st.markdown("## 📊 Stock Selection")
        use_all = st.checkbox("Scan All NIFTY 50", value=True)
        
        if not use_all:
            selected_symbols = st.multiselect(
                "Select Stocks",
                options=NIFTY_50_SYMBOLS,
                default=NIFTY_50_SYMBOLS[:10]
            )
        else:
            selected_symbols = NIFTY_50_SYMBOLS
        
        st.markdown("---")
        
        # Scan button
        scan_button = st.button("🔍 Start Scan", use_container_width=True, type="primary")
        
        # Auto refresh
        auto_refresh = st.checkbox("Auto Refresh")
        if auto_refresh:
            refresh_interval = st.number_input("Refresh Interval (minutes)", min_value=5, max_value=60, value=30)
    
    # Main content area
    if scan_button or auto_refresh:
        # Initialize scanner
        scanner = BollingerRSIScanner(selected_symbols, config)
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Scanning stocks..."):
            buy_signals, sell_signals, watch_list = scanner.scan_all_stocks(progress_bar, status_text)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🟢 Buy Signals", "🔴 Sell Signals", "👀 Watchlist", "📈 Charts"])
        
        with tab1:
            # Dashboard metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Scanned", len(selected_symbols), delta=None)
            with col2:
                st.metric("Buy Signals", len(buy_signals), delta=f"+{len(buy_signals)}")
            with col3:
                st.metric("Sell Signals", len(sell_signals), delta=f"-{len(sell_signals)}")
            with col4:
                st.metric("Watchlist", len(watch_list), delta=None)
            
            st.markdown("---")
            
            # Signal overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🟢 Top Buy Signals")
                if buy_signals:
                    for signal in buy_signals[:3]:
                        display_signal_card(signal, "BUY")
                else:
                    st.info("No buy signals found")
            
            with col2:
                st.markdown("### 🔴 Top Sell Signals")
                if sell_signals:
                    for signal in sell_signals[:3]:
                        display_signal_card(signal, "SELL")
                else:
                    st.info("No sell signals found")
        
        with tab2:
            st.markdown("### 🟢 Buy Signals")
            if buy_signals:
                # Convert to DataFrame for display
                buy_df = pd.DataFrame(buy_signals)
                buy_df = buy_df[['symbol', 'current_price', 'rsi', 'lower_bb', 'bb_position', 
                                'distance_from_lower', 'volume_ratio', 'signal_strength', 'timestamp']]
                buy_df.columns = ['Symbol', 'Price', 'RSI', 'Lower BB', 'BB Position %', 
                                 'Dist from Lower %', 'Volume Ratio', 'Signal Strength %', 'Time']
                
                # Add strength indicator
                buy_df['Strength'] = buy_df['Signal Strength %'].apply(lambda x: '🔥' if x > 70 else '✓')
                
                st.dataframe(
                    buy_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Signal Strength %": st.column_config.ProgressColumn(
                            "Signal Strength %",
                            help="Signal strength based on price position and RSI",
                            format="%d",
                            min_value=0,
                            max_value=100,
                        ),
                    }
                )
                
                # Download button
                csv = buy_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Buy Signals",
                    data=csv,
                    file_name=f"buy_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No buy signals found. Signals appear when: Price ≤ Lower BB AND RSI < 30")
        
        with tab3:
            st.markdown("### 🔴 Sell Signals")
            if sell_signals:
                # Convert to DataFrame for display
                sell_df = pd.DataFrame(sell_signals)
                sell_df = sell_df[['symbol', 'current_price', 'rsi', 'upper_bb', 'bb_position', 
                                  'distance_from_upper', 'volume_ratio', 'signal_strength', 'timestamp']]
                sell_df.columns = ['Symbol', 'Price', 'RSI', 'Upper BB', 'BB Position %', 
                                  'Dist from Upper %', 'Volume Ratio', 'Signal Strength %', 'Time']
                
                # Add strength indicator
                sell_df['Strength'] = sell_df['Signal Strength %'].apply(lambda x: '🔥' if x > 70 else '✓')
                
                st.dataframe(
                    sell_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Signal Strength %": st.column_config.ProgressColumn(
                            "Signal Strength %",
                            help="Signal strength based on price position and RSI",
                            format="%d",
                            min_value=0,
                            max_value=100,
                        ),
                    }
                )
                
                # Download button
                csv = sell_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sell Signals",
                    data=csv,
                    file_name=f"sell_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No sell signals found. Signals appear when: Price ≥ Upper BB AND RSI > 70")
        
        with tab4:
            st.markdown("### 👀 Watchlist (Near Signal)")
            if watch_list:
                # Convert to DataFrame for display
                watch_df = pd.DataFrame(watch_list)
                watch_df = watch_df[['symbol', 'current_price', 'rsi', 'bb_position', 
                                    'distance_from_lower', 'distance_from_upper', 'volume_ratio']]
                watch_df.columns = ['Symbol', 'Price', 'RSI', 'BB Position %', 
                                   'Dist from Lower %', 'Dist from Upper %', 'Volume Ratio']
                
                # Add zone indicator
                watch_df['Zone'] = watch_df['RSI'].apply(lambda x: '🟢 Buy Zone' if x < 50 else '🔴 Sell Zone')
                
                st.dataframe(
                    watch_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No stocks in watchlist")
        
        with tab5:
            st.markdown("### 📈 Technical Analysis Charts")
            
            # Select stock for charting
            all_signals = buy_signals + sell_signals + watch_list
            if all_signals:
                chart_symbols = [s['symbol'] for s in all_signals]
                selected_chart = st.selectbox("Select Stock for Chart", chart_symbols)
                
                # Find the selected stock data
                selected_data = next((s for s in all_signals if s['symbol'] == selected_chart), None)
                
                if selected_data and 'data' in selected_data:
                    stock_data = selected_data['data']
                    
                    # Calculate indicators for the full dataset
                    rsi_full = scanner.calculate_rsi(stock_data['Close'], config['rsi_period'])
                    upper_bb, middle_bb, lower_bb = scanner.calculate_bollinger_bands(
                        stock_data['Close'], config['bb_period'], config['bb_std']
                    )
                    
                    bb_data = {
                        'upper': upper_bb,
                        'middle': middle_bb,
                        'lower': lower_bb
                    }
                    
                    # Create and display chart
                    fig = create_technical_chart(stock_data, selected_chart, bb_data, rsi_full)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display current metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"₹{selected_data['current_price']}")
                    with col2:
                        st.metric("RSI", f"{selected_data['rsi']}")
                    with col3:
                        st.metric("BB Position", f"{selected_data['bb_position']}%")
                    with col4:
                        signal_label = selected_data['signal'] if selected_data['signal'] else "No Signal"
                        st.metric("Signal", signal_label)
            else:
                st.info("No data available for charting")
        
        # Auto refresh logic
        if auto_refresh:
            time.sleep(refresh_interval * 60)
            st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to NIFTY 50 Scanner! 👋
        
        This scanner helps you identify trading opportunities using:
        - **Bollinger Bands**: Identify overbought/oversold conditions
        - **RSI (Relative Strength Index)**: Confirm momentum
        - **Volume Analysis**: Validate signals with volume confirmation
        
        ### How to use:
        1. Configure your parameters in the sidebar
        2. Select stocks to scan (or use all NIFTY 50)
        3. Click **Start Scan** to begin analysis
        4. Review signals in different tabs
        5. Use charts for detailed technical analysis
        
        ### Signal Criteria:
        - 🟢 **Buy Signal**: Price ≤ Lower BB AND RSI < 30
        - 🔴 **Sell Signal**: Price ≥ Upper BB AND RSI > 70
        - 👀 **Watchlist**: Stocks approaching signal zones
        
        Ready to start? Configure your settings and click **Start Scan**!
        """)
        
        # Show sample chart
        st.markdown("### 📊 Sample Technical Analysis")
        sample_data = yf.Ticker('INFY.NS').history(period='1mo', interval='1h')
        if not sample_data.empty:
            scanner = BollingerRSIScanner(['INFY.NS'], {
                'bb_period': 20, 'bb_std': 2, 'rsi_period': 14,
                'rsi_oversold': 30, 'rsi_overbought': 70
            })
            
            rsi = scanner.calculate_rsi(sample_data['Close'], 14)
            upper_bb, middle_bb, lower_bb = scanner.calculate_bollinger_bands(sample_data['Close'], 20, 2)
            
            bb_data = {'upper': upper_bb, 'middle': middle_bb, 'lower': lower_bb}
            fig = create_technical_chart(sample_data, 'INFY', bb_data, rsi)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
