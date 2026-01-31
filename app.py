import streamlit as st
import yfinance as yf
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import datetime

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="MacroNexus Pro",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Institutional" Dark Mode Look
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .metric-card {
        background-color: #1e2127;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4b5563;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-label { font-size: 12px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .metric-value { font-size: 24px; font-weight: bold; color: #f3f4f6; }
    .metric-delta { font-size: 14px; font-weight: bold; }
    
    /* Risk Levels */
    .risk-critical { border-left-color: #ef4444 !important; } /* Red */
    .risk-safe { border-left-color: #22c55e !important; } /* Green */
    .risk-warning { border-left-color: #f59e0b !important; } /* Orange */
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LAYER (Live Data Fetching) ---
# NOTE: Switched to ETFs (GLD, USO, UUP, VIXY) for robustness on Cloud Hosting.
# ^TNX (Yields) is crucial, so we keep it but have fallback logic.
TICKERS = {
    'US10Y': '^TNX',       # 10 Year Treasury Yield (CBOE)
    'DXY': 'UUP',          # PROXY: Invesco DB US Dollar Index Bullish Fund
    'VIX': 'VIXY',         # PROXY: ProShares VIX Short-Term Futures ETF
    'HYG': 'HYG',          # High Yield Corp Bonds (Credit)
    'SPY': 'SPY',          # S&P 500
    'QQQ': 'QQQ',          # Nasdaq
    'IWM': 'IWM',          # Russell 2000
    'BTC': 'BTC-USD',      # Bitcoin
    'GOLD': 'GLD',         # PROXY: SPDR Gold Shares
    'OIL': 'USO',          # PROXY: United States Oil Fund
    'COPPER': 'CPER',      # PROXY: United States Copper Index Fund
    'TLT': 'TLT'           # 20Y Treasury Bond
}

@st.cache_data(ttl=300) # Cache data for 5 minutes
def fetch_live_data():
    """Fetches real-time percent change for macro assets individually to prevent bulk failures."""
    data_map = {}
    
    for key, symbol in TICKERS.items():
        try:
            # Fetch individual ticker history
            ticker = yf.Ticker(symbol)
            # Get 5 days to ensure we have at least 2 valid trading days (ignoring weekends/holidays)
            hist = ticker.history(period="5d")
            
            if not hist.empty and len(hist) >= 2:
                # Ensure we use Close prices
                closes = hist['Close']
                current = closes.iloc[-1]
                prev = closes.iloc[-2]
                
                # Calculate change
                change_pct = ((current - prev) / prev) * 100
                
                data_map[key] = {
                    'price': current,
                    'change': change_pct,
                    'symbol': symbol
                }
            else:
                # Fallback if data is empty (e.g. market closed or API issue)
                data_map[key] = {'price': 0.0, 'change': 0.0, 'symbol': symbol}
                
        except Exception as e:
            # Silent fail for individual ticker to keep dashboard alive
            print(f"Error fetching {symbol}: {e}")
            data_map[key] = {'price': 0.0, 'change': 0.0, 'symbol': symbol}
            
    return data_map

# --- 2. LOGIC LAYER (Regime Engine) ---
def analyze_market(data):
    """Determines the current Macro Regime based on asset performance."""
    if not data: return None

    # Safe extraction with defaults
    def get_chg(k): return data.get(k, {}).get('change', 0)

    hyg_chg = get_chg('HYG')
    vix_chg = get_chg('VIX')
    oil_chg = get_chg('OIL')
    copper_chg = get_chg('COPPER')
    us10y_chg = get_chg('US10Y')
    dxy_chg = get_chg('DXY')
    btc_chg = get_chg('BTC')

    # Logic Tree
    regime = "NEUTRAL"
    color = "gray"
    longs = []
    shorts = []
    alerts = []

    # 1. THE VETO (Risk Off)
    # If Credit (HYG) is down significantly OR Volatility (VIXY) is up
    if hyg_chg < -0.3 or vix_chg > 2.0:
        regime = "RISK OFF (DEFENSIVE)"
        color = "red"
        longs = ["DXY (Cash)", "VIX", "US10Y (Bonds)"]
        shorts = ["Tech", "Crypto", "Small Caps", "EM"]
        alerts.append("⛔ CRITICAL: Credit spreads widening or Volatility spiking. VETO all long risk trades.")
    
    # 2. REFLATION (Growth + Yields Up)
    # Commodities moving up + Yields rising = Inflation trade
    elif (oil_chg > 0.5 or copper_chg > 0.5) and us10y_chg > 1.0:
        regime = "REFLATION (INFLATIONARY GROWTH)"
        color = "orange"
        longs = ["Energy", "Banks", "Industrials", "Commodities"]
        shorts = ["Tech (Rate Sensitive)", "Bonds (TLT)"]
        alerts.append("🔥 INFLATION PULSE: Commodities leading. Rotate from Tech to Energy/Industrials.")

    # 3. LIQUIDITY PUMP (Dollar Down, Crypto Up)
    elif dxy_chg < -0.1 and btc_chg > 1.0:
        regime = "LIQUIDITY PUMP (RISK ON)"
        color = "purple"
        longs = ["Bitcoin", "Nasdaq", "Gold", "Spec Tech"]
        shorts = ["DXY", "Cash"]
        alerts.append("🌊 LIQUIDITY INJECTION: Dollar weak. Green light for High Beta Assets.")

    # 4. GOLDILOCKS (Vol down, Yields stable)
    elif vix_chg < 0 and abs(us10y_chg) < 2.0:
        regime = "GOLDILOCKS (SLOW GRIND UP)"
        color = "green"
        longs = ["S&P 500", "Tech", "Quality Growth"]
        shorts = ["VIX"]
        alerts.append("✅ STABLE: Volatility is dropping. 'Buy the dip' environment.")

    return {
        'regime': regime,
        'color': color,
        'longs': longs,
        'shorts': shorts,
        'alerts': alerts
    }

# --- 3. VISUALIZATION LAYER (Plotly Network) ---
def create_nexus_graph(market_data, regime_data):
    """Draws the interactive node graph using Plotly."""
    
    # Define Nodes (Positioning is manual for better layout)
    nodes = {
        # Center (Gravity)
        'US10Y': {'pos': (0, 0), 'group': 'Rates', 'label': 'US 10Y'},
        'DXY':   {'pos': (1, 1), 'group': 'Forex', 'label': 'USD (UUP)'},
        
        # Inner Ring (Primary Assets)
        'SPY':   {'pos': (-1, 1), 'group': 'Equity', 'label': 'S&P 500'},
        'QQQ':   {'pos': (-1.5, 0.5), 'group': 'Equity', 'label': 'Nasdaq'},
        'GOLD':  {'pos': (1, -1), 'group': 'Commodity', 'label': 'Gold (GLD)'},
        'HYG':   {'pos': (-0.5, -1), 'group': 'Credit', 'label': 'Credit (HYG)'},
        
        # Outer Ring (High Beta / Macro Proxies)
        'BTC':   {'pos': (-2, 2), 'group': 'Crypto', 'label': 'Bitcoin'},
        'OIL':   {'pos': (2, -0.5), 'group': 'Commodity', 'label': 'Oil (USO)'},
        'COPPER':{'pos': (1.5, -1.5), 'group': 'Commodity', 'label': 'Copper'},
        'IWM':   {'pos': (-1.5, -1.5), 'group': 'Equity', 'label': 'Russell 2000'},
    }

    # Define Edges (Connections)
    edges = [
        ('US10Y', 'QQQ', 'red'),   # Rates hurt Tech
        ('US10Y', 'GOLD', 'red'),  # Rates hurt Gold
        ('DXY', 'GOLD', 'red'),    # Dollar hurts Gold
        ('DXY', 'OIL', 'red'),     # Dollar hurts Oil
        ('HYG', 'SPY', 'green'),   # Credit helps Stocks
        ('HYG', 'IWM', 'green'),   # Credit helps Small Caps
        ('QQQ', 'BTC', 'green'),   # Tech correlates with Crypto
        ('COPPER', 'US10Y', 'green') # Growth drives Rates
    ]

    edge_x = []
    edge_y = []
    edge_colors = []

    for edge in edges:
        x0, y0 = nodes[edge[0]]['pos']
        x1, y1 = nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        # Color line based on simple logic or static definition
        col = 'rgba(239, 68, 68, 0.6)' if edge[2] == 'red' else 'rgba(34, 197, 94, 0.6)'
        edge_colors.append(col)

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for key, info in nodes.items():
        x, y = info['pos']
        node_x.append(x)
        node_y.append(y)
        
        # Get live change
        chg = market_data.get(key, {}).get('change', 0)
        formatted_chg = f"{chg:+.2f}%"
        
        # Node Color Logic (Green if UP, Red if DOWN)
        # Exception: US10Y and DXY Up is usually "Red" for risk assets, but we color the node by its own price action here
        # If DXY or US10Y are UP, let's color them RED to signify "Risk Off pressure" for easier visual reading?
        # Actually, let's keep it simple: Green = Price Up, Red = Price Down.
        col = '#22c55e' if chg > 0 else '#ef4444'
        if chg == 0: col = '#6b7280'
        
        node_color.append(col)
        node_size.append(45 if key in ['US10Y', 'DXY', 'HYG'] else 35)
        node_text.append(f"<b>{info['label']}</b><br>{formatted_chg}")

    # Draw Lines
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#4b5563'),
        hoverinfo='none',
        mode='lines'
    ))

    # Draw Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        textfont=dict(size=14, color="white"),
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=2, color='white')
        )
    ))

    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    return fig

# --- 4. MAIN APP LAYOUT ---

def main():
    # Load Data
    with st.spinner('Connecting to Global Markets...'):
        market_data = fetch_live_data()
        analysis = analyze_market(market_data)

    # --- SIDEBAR: KNOWLEDGE BASE ---
    with st.sidebar:
        st.title("📚 Knowledge Base")
        
        with st.expander("📖 The Daily Playbook", expanded=True):
            st.markdown("""
            **1. The Veto Check (Credit)**
            Check `Credit (HYG)`. If Red (> -0.3%), **STOP**. Do not buy the dip.
            
            **2. The Regime Check**
            Look at the Dashboard. 
            * **Green:** Buy Stocks/Crypto.
            * **Red:** Cash is King.
            * **Orange:** Buy Energy/Banks.
            
            **3. The Confirmation**
            Before buying Gold, check `10Y Yields`. They must be falling.
            """)
            
        with st.expander("🧠 Deep Dives"):
            st.info("**Why HYG Matters:** High Yield bonds are the 'canary in the coal mine'. If HYG falls, bankruptcy risk is rising.")
            st.info("**UUP vs DXY:** We use `UUP` as a proxy for the Dollar Index because it trades as an ETF and provides reliable data. If UUP is Green, the Dollar is getting stronger.")

        st.divider()
        st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')} (ETF Proxies Used)")
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # --- MAIN CONTENT ---
    st.title("MacroNexus Pro: Command Center")
    
    # 1. MORNING BRIEFING (DASHBOARD)
    # Determine CSS class for header
    header_class = "risk-safe"
    if analysis and "RISK OFF" in analysis['regime']: header_class = "risk-critical"
    elif analysis and "REFLATION" in analysis['regime']: header_class = "risk-warning"

    if analysis:
        st.markdown(f"""
        <div class="metric-card {header_class}">
            <div class="metric-label">Current Market Regime</div>
            <div class="metric-value">{analysis['regime']}</div>
            <div style="margin-top: 10px; font-size: 14px;">
                {' '.join([f"⚠️ {a}" for a in analysis['alerts']])}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. KEY METRICS ROW
        c1, c2, c3, c4 = st.columns(4)
        
        def metric_html(label, ticker):
            data = market_data.get(ticker, {})
            val = data.get('price', 0)
            chg = data.get('change', 0)
            
            color = "#ef4444" if chg < 0 else "#22c55e"
            # Inverse logic for VIX/US10Y/DXY: Green is BAD for risk assets usually, but let's keep visual consistency:
            # Green text = Price Went Up.
            # However, for the BORDER (Risk indicator), we can be smarter.
            border_color = color
            if ticker in ['VIX', 'US10Y', 'DXY']:
                # If these go UP (Green), it's a Warning (Orange/Red)
                if chg > 0: border_color = "#ef4444" 
                else: border_color = "#22c55e"

            return f"""
            <div class="metric-card" style="border-left-color: {border_color};">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val:.2f}</div>
                <div class="metric-delta" style="color: {color};">{chg:+.2f}%</div>
            </div>
            """

        c1.markdown(metric_html("Credit (Veto)", "HYG"), unsafe_allow_html=True)
        c2.markdown(metric_html("Volatility (VIXY)", "VIX"), unsafe_allow_html=True)
        c3.markdown(metric_html("10Y Yields", "US10Y"), unsafe_allow_html=True)
        c4.markdown(metric_html("Liquidity (BTC)", "BTC"), unsafe_allow_html=True)

        # 3. ACTION PLAN & GRAPH
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("🎯 Action Plan")
            st.markdown("Based on live data:")
            
            st.success(f"**FOCUS (LONG):**\n\n{', '.join(analysis['longs'])}")
            st.error(f"**AVOID (SHORT):**\n\n{', '.join(analysis['shorts'])}")
            
            st.markdown("---")
            st.markdown("**Live Correlations:**")
            
            # Safe getters
            def get_chg(k): return market_data.get(k, {}).get('change', 0)
            
            if get_chg('US10Y') > 0.5:
                st.write("📉 Yields Spiking -> Tech/Gold Headwind")
            if get_chg('DXY') > 0.2:
                st.write("📉 Dollar Strong -> Emerging Mkts Headwind")
            if get_chg('HYG') > 0.1:
                st.write("📈 Credit Healthy -> Buying Dip Supported")

        with col_right:
            st.subheader("🌐 Live Correlation Nexus")
            fig = create_nexus_graph(market_data, analysis)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Unable to load market data. Please refresh.")

if __name__ == "__main__":
    main()
