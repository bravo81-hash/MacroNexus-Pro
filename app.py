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
    }
    .metric-label { font-size: 12px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #f3f4f6; }
    .metric-delta { font-size: 14px; font-weight: bold; }
    
    /* Risk Levels */
    .risk-critical { border-left-color: #ef4444 !important; } /* Red */
    .risk-safe { border-left-color: #22c55e !important; } /* Green */
    .risk-warning { border-left-color: #f59e0b !important; } /* Orange */
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LAYER (Live Data Fetching) ---
# Mapping friendly names to Yahoo Finance Tickers
TICKERS = {
    'US10Y': '^TNX',       # 10 Year Treasury Yield
    'DXY': 'DX-Y.NYB',     # US Dollar Index
    'VIX': '^VIX',         # Volatility
    'HYG': 'HYG',          # High Yield Corp Bonds (Credit)
    'SPY': 'SPY',          # S&P 500
    'QQQ': 'QQQ',          # Nasdaq
    'IWM': 'IWM',          # Russell 2000
    'BTC': 'BTC-USD',      # Bitcoin
    'GOLD': 'GC=F',        # Gold Futures
    'OIL': 'CL=F',         # Crude Oil
    'COPPER': 'HG=F',      # Copper
    'TLT': 'TLT'           # 20Y Treasury Bond
}

@st.cache_data(ttl=300) # Cache data for 5 minutes
def fetch_live_data():
    """Fetches real-time percent change for macro assets."""
    data_map = {}
    
    try:
        # Fetch all at once for speed
        tickers_list = " ".join(TICKERS.values())
        data = yf.download(tickers_list, period="5d", progress=False)['Close']
        
        for key, symbol in TICKERS.items():
            if symbol in data.columns:
                series = data[symbol]
                # Calculate daily percent change
                if len(series) >= 2:
                    current = series.iloc[-1]
                    prev = series.iloc[-2]
                    change_pct = ((current - prev) / prev) * 100
                    data_map[key] = {
                        'price': current,
                        'change': change_pct,
                        'symbol': symbol
                    }
                else:
                    data_map[key] = {'price': 0, 'change': 0, 'symbol': symbol}
    except Exception as e:
        st.error(f"Data Feed Error: {e}")
        # Fallback to 0s if API fails (prevents crash)
        for key in TICKERS:
            data_map[key] = {'price': 0, 'change': 0, 'symbol': TICKERS[key]}
            
    return data_map

# --- 2. LOGIC LAYER (Regime Engine) ---
def analyze_market(data):
    """Determines the current Macro Regime based on asset performance."""
    if not data: return None

    # Thresholds
    hyg_chg = data.get('HYG', {}).get('change', 0)
    vix_chg = data.get('VIX', {}).get('change', 0)
    oil_chg = data.get('OIL', {}).get('change', 0)
    copper_chg = data.get('COPPER', {}).get('change', 0)
    us10y_chg = data.get('US10Y', {}).get('change', 0)
    dxy_chg = data.get('DXY', {}).get('change', 0)
    btc_chg = data.get('BTC', {}).get('change', 0)

    # Logic Tree
    regime = "NEUTRAL"
    color = "gray"
    longs = []
    shorts = []
    alerts = []

    # 1. THE VETO (Risk Off)
    # If Credit is down significantly or VIX is spiking hard
    if hyg_chg < -0.3 or vix_chg > 5.0:
        regime = "RISK OFF (DEFENSIVE)"
        color = "red"
        longs = ["DXY (Cash)", "VIX", "US10Y (Bonds)"]
        shorts = ["Tech", "Crypto", "Small Caps", "EM"]
        alerts.append("⛔ CRITICAL: Credit spreads widening. VETO all long risk trades.")
    
    # 2. REFLATION (Growth + Yields Up)
    elif (oil_chg > 0.5 or copper_chg > 0.5) and us10y_chg > 1.0:
        regime = "REFLATION (INFLATIONARY GROWTH)"
        color = "orange"
        longs = ["Energy", "Banks", "Industrials", "Commodities"]
        shorts = ["Tech (Rate Sensitive)", "Bonds (TLT)"]
        alerts.append("🔥 INFLATION PULSE: Commodities leading. Rotate from Tech to Energy.")

    # 3. LIQUIDITY PUMP (Dollar Down, Crypto Up)
    elif dxy_chg < -0.2 and btc_chg > 1.0:
        regime = "LIQUIDITY PUMP (RISK ON)"
        color = "purple"
        longs = ["Bitcoin", "Nasdaq", "Gold", "Spec Tech"]
        shorts = ["DXY", "Cash"]
        alerts.append("🌊 LIQUIDITY INJECTION: Dollar weak. Green light for High Beta.")

    # 4. GOLDILOCKS (Vol down, Yields stable)
    elif vix_chg < 0 and abs(us10y_chg) < 2.0:
        regime = "GOLDILOCKS (SLOW GRIND UP)"
        color = "green"
        longs = ["S&P 500", "Tech", "Quality Growth"]
        shorts = ["VIX"]
        alerts.append("✅ STABLE: Volatility is dropping. Buy the dip environment.")

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
    # X, Y coordinates approximate the "Solar System" view
    nodes = {
        # Center (Gravity)
        'US10Y': {'pos': (0, 0), 'group': 'Rates', 'label': 'US 10Y'},
        'DXY':   {'pos': (1, 1), 'group': 'Forex', 'label': 'USD (DXY)'},
        
        # Inner Ring (Primary Assets)
        'SPY':   {'pos': (-1, 1), 'group': 'Equity', 'label': 'S&P 500'},
        'QQQ':   {'pos': (-1.5, 0.5), 'group': 'Equity', 'label': 'Nasdaq'},
        'GOLD':  {'pos': (1, -1), 'group': 'Commodity', 'label': 'Gold'},
        'HYG':   {'pos': (-0.5, -1), 'group': 'Credit', 'label': 'Credit (HYG)'},
        
        # Outer Ring (High Beta / Macro Proxies)
        'BTC':   {'pos': (-2, 2), 'group': 'Crypto', 'label': 'Bitcoin'},
        'OIL':   {'pos': (2, -0.5), 'group': 'Commodity', 'label': 'Oil'},
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
        edge_colors.append(col) # Plotly handles line colors differently in complex traces, simplifying for robustness:
    
    # We use a single trace for lines for simplicity, but colored lines require split traces or advanced mapping.
    # For this dashboard, we will use grey lines but color the NODES dynamically based on live data.

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
    with st.spinner('Fetching Live Market Data...'):
        market_data = fetch_live_data()
        analysis = analyze_market(market_data)

    # --- SIDEBAR: KNOWLEDGE BASE ---
    with st.sidebar:
        st.title("📚 Knowledge Base")
        
        with st.expander("📖 The Daily Playbook", expanded=True):
            st.markdown("""
            **1. The Veto Check (Credit)**
            Check `HYG`. If Red (> -0.3%), **STOP**. Do not buy dip.
            
            **2. The Regime Check**
            Look at the Dashboard. 
            * **Green:** Buy Stocks/Crypto.
            * **Red:** Cash is King.
            * **Orange:** Buy Energy/Banks.
            
            **3. The Confirmation**
            Before buying Gold, check `Real Yields`. They must be falling.
            """)
            
        with st.expander("🧠 Deep Dives"):
            st.info("**Why HYG Matters:** High Yield bonds are the 'canary in the coal mine'. Equity investors can be delusional, but bond investors care about getting paid back. If HYG falls, bankruptcy risk is rising.")
            st.info("**Real Yields vs Gold:** Gold pays no interest. If Bonds pay 5% + Inflation, nobody wants Gold. Gold needs Yields to fall to shine.")

        st.divider()
        st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # --- MAIN CONTENT ---
    st.title("MacroNexus Pro: Command Center")
    
    # 1. MORNING BRIEFING (DASHBOARD)
    # Determine CSS class for header
    header_class = "risk-safe"
    if "RISK OFF" in analysis['regime']: header_class = "risk-critical"
    elif "REFLATION" in analysis['regime']: header_class = "risk-warning"

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
        # Inverse logic for VIX/US10Y
        if ticker in ['VIX', 'US10Y', 'DXY']:
            color = "#ef4444" if chg > 0 else "#22c55e" # Red if Up (Bad)
            
        return f"""
        <div class="metric-card" style="border-left-color: {color};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val:.2f}</div>
            <div class="metric-delta" style="color: {color};">{chg:+.2f}%</div>
        </div>
        """

    c1.markdown(metric_html("Credit (Veto)", "HYG"), unsafe_allow_html=True)
    c2.markdown(metric_html("Volatility", "VIX"), unsafe_allow_html=True)
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
        if market_data['US10Y']['change'] > 0.5:
            st.write("📉 Yields Spiking -> Tech/Gold Headwind")
        if market_data['DXY']['change'] > 0.2:
            st.write("📉 Dollar Strong -> Emerging Mkts Headwind")
        if market_data['HYG']['change'] > 0.1:
            st.write("📈 Credit Healthy -> Buying Dip Supported")

    with col_right:
        st.subheader("🌐 Live Correlation Nexus")
        fig = create_nexus_graph(market_data, analysis)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
