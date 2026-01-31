import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime
import graphviz

# --- CONFIGURATION ---
st.set_page_config(
    page_title="MacroNexus Pro",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (FIXED LAYOUT) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Fixed Metric Card - Prevents cutoff */
    .metric-container {
        background-color: #1e2127;
        padding: 12px 15px;
        border-radius: 8px;
        border-left: 5px solid #4b5563;
        margin-bottom: 10px;
        min-height: 90px; /* Ensures height consistency */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label { 
        font-size: 11px; 
        color: #9ca3af; 
        text-transform: uppercase; 
        letter-spacing: 1px; 
        margin-bottom: 4px;
        font-weight: 600;
    }
    .metric-val { 
        font-size: 22px; 
        font-weight: bold; 
        color: #f3f4f6; 
        line-height: 1.2;
    }
    .metric-chg { 
        font-size: 14px; 
        font-weight: bold; 
        margin-left: 8px;
    }
    
    /* Regime Badge styling */
    .regime-box {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 15px;
        border: 1px solid;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA FETCHING ---
TICKERS = {
    'US10Y': '^TNX',       # 10Y Yield
    'DXY': 'UUP',          # Dollar Proxy
    'VIX': 'VIXY',         # Volatility Proxy
    'HYG': 'HYG',          # Credit High Yield
    'SPY': 'SPY',          # S&P 500
    'QQQ': 'QQQ',          # Nasdaq
    'IWM': 'IWM',          # Russell 2000
    'BTC': 'BTC-USD',      # Bitcoin
    'GOLD': 'GLD',         # Gold
    'OIL': 'USO',          # Oil
    'COPPER': 'CPER',      # Copper
    'TLT': 'TLT'           # Bonds (20Y)
}

@st.cache_data(ttl=300)
def fetch_live_data():
    """Fetches data individually to prevent bulk failure."""
    data_map = {}
    for key, symbol in TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if not hist.empty and len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change_pct = ((current - prev) / prev) * 100
                data_map[key] = {'price': current, 'change': change_pct, 'symbol': symbol}
            else:
                data_map[key] = {'price': 0.0, 'change': 0.0, 'symbol': symbol}
        except:
            data_map[key] = {'price': 0.0, 'change': 0.0, 'symbol': symbol}
    return data_map

# --- 2. LOGIC ENGINE (UPDATED FOR NEUTRAL) ---
def analyze_market(data):
    """Determines Regime and Signals."""
    if not data: return None
    
    def get_c(k): return data.get(k, {}).get('change', 0)
    
    hyg, vix, oil, cop, us10y, dxy, btc = get_c('HYG'), get_c('VIX'), get_c('OIL'), get_c('COPPER'), get_c('US10Y'), get_c('DXY'), get_c('BTC')

    # Prepare logic containers
    regime = "NEUTRAL"
    desc = "No dominant macro trend. Following momentum."
    color_code = "#6b7280" # Gray
    longs, shorts, alerts = [], [], []

    # 1. MACRO REGIME CHECKS
    if hyg < -0.3 or vix > 2.0:
        regime = "RISK OFF"
        desc = "Credit widening or Vol spiking."
        color_code = "#ef4444" # Red
        longs = ["Cash (DXY)", "VIX"]
        shorts = ["Tech", "Crypto", "Small Caps", "EM"]
        alerts.append("⛔ CREDIT STRESS: Veto Longs.")

    elif (oil > 0.5 or cop > 0.5) and us10y > 1.0:
        regime = "REFLATION"
        desc = "Growth + Inflation rising."
        color_code = "#f59e0b" # Orange
        longs = ["Energy", "Banks", "Industrials"]
        shorts = ["Bonds", "Tech"]
        alerts.append("🔥 INFLATION: Buy Real Assets.")

    elif dxy < -0.1 and btc > 1.0:
        regime = "LIQUIDITY PUMP"
        desc = "Dollar weak, Crypto leading."
        color_code = "#a855f7" # Purple
        longs = ["Bitcoin", "Nasdaq", "Gold"]
        shorts = ["Dollar"]
        alerts.append("🌊 LIQUIDITY: Risk On.")

    elif vix < 0 and abs(us10y) < 2.0:
        regime = "GOLDILOCKS"
        desc = "Low vol, stable rates."
        color_code = "#22c55e" # Green
        longs = ["S&P 500", "Tech"]
        shorts = ["VIX"]
        alerts.append("✅ STABLE: Buy Dips.")

    # 2. FALLBACK MOMENTUM LOGIC (For Neutral Markets)
    # If the macro logic didn't populate longs/shorts, pick top performers
    if not longs:
        # Sort assets by performance
        assets = {k: get_c(k) for k in ['SPY', 'QQQ', 'IWM', 'BTC', 'GOLD', 'OIL', 'COPPER']}
        sorted_assets = sorted(assets.items(), key=lambda x: x[1], reverse=True)
        
        # Pick top 2 and bottom 2
        longs = [f"{k} (Mom.)" for k, v in sorted_assets[:2] if v > 0]
        shorts = [f"{k} (Mom.)" for k, v in sorted_assets[-2:] if v < 0]
        
        if not longs: longs = ["Cash"] # If nothing is up
        if not shorts: shorts = ["None"] # If nothing is down

    return {
        'regime': regime, 'desc': desc, 'color': color_code,
        'longs': longs, 'shorts': shorts, 'alerts': alerts
    }

# --- 3. GRAPHICS ENGINE ---
def create_nexus_graph(market_data):
    # Solar System Layout
    nodes = {
        'US10Y': {'pos': (0, 0), 'label': 'US 10Y'},
        'DXY':   {'pos': (0.8, 0.8), 'label': 'USD'},
        'SPY':   {'pos': (-0.8, 0.8), 'label': 'S&P 500'},
        'QQQ':   {'pos': (-1.2, 0.4), 'label': 'Nasdaq'},
        'GOLD':  {'pos': (0.8, -0.8), 'label': 'Gold'},
        'HYG':   {'pos': (-0.4, -0.8), 'label': 'Credit'},
        'BTC':   {'pos': (-1.5, 1.5), 'label': 'Bitcoin'},
        'OIL':   {'pos': (1.5, -0.4), 'label': 'Oil'},
        'COPPER':{'pos': (1.2, -1.2), 'label': 'Copper'},
        'IWM':   {'pos': (-1.2, -1.0), 'label': 'Russell'}
    }
    
    edges = [
        ('US10Y', 'QQQ'), ('US10Y', 'GOLD'), ('DXY', 'GOLD'), ('DXY', 'OIL'),
        ('HYG', 'SPY'), ('HYG', 'IWM'), ('QQQ', 'BTC'), ('COPPER', 'US10Y')
    ]
    
    edge_x, edge_y = [], []
    for u, v in edges:
        x0, y0 = nodes[u]['pos']
        x1, y1 = nodes[v]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    
    for key, info in nodes.items():
        x, y = info['pos']
        node_x.append(x)
        node_y.append(y)
        chg = market_data.get(key, {}).get('change', 0)
        
        col = '#22c55e' if chg > 0 else '#ef4444'
        if chg == 0: col = '#6b7280'

        node_color.append(col)
        node_size.append(45 if key in ['US10Y', 'DXY', 'HYG'] else 35)
        node_text.append(f"<b>{info['label']}</b><br>{chg:+.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#4b5563'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="bottom center",
                             marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')),
                             textfont=dict(size=12, color='white')))
    
    fig.update_layout(
        showlegend=False, margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.0, 2.0]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450
    )
    return fig

# --- 4. APP LAYOUT ---
def main():
    with st.spinner("Connecting to Markets..."):
        market_data = fetch_live_data()
        analysis = analyze_market(market_data)

    # --- TOP METRICS ROW ---
    c1, c2, c3, c4, c5 = st.columns(5)
    
    def metric_html(label, key):
        d = market_data.get(key, {})
        val = d.get('price', 0)
        chg = d.get('change', 0)
        color = "#ef4444" if chg < 0 else "#22c55e"
        if key in ['VIX', 'US10Y', 'DXY']: color = "#ef4444" if chg > 0 else "#22c55e" # Inverse logic
        
        return f"""
        <div class="metric-container" style="border-left-color: {color};">
            <div class="metric-label">{label}</div>
            <div>
                <span class="metric-val">{val:.2f}</span>
                <span class="metric-chg" style="color: {color};">{chg:+.2f}%</span>
            </div>
        </div>
        """
    
    c1.markdown(metric_html("Credit (Veto)", "HYG"), unsafe_allow_html=True)
    c2.markdown(metric_html("Volatility", "VIX"), unsafe_allow_html=True)
    c3.markdown(metric_html("Yields", "US10Y"), unsafe_allow_html=True)
    c4.markdown(metric_html("Dollar", "DXY"), unsafe_allow_html=True)
    c5.markdown(metric_html("Liquidity", "BTC"), unsafe_allow_html=True)

    # --- TABS ---
    tab_dash, tab_matrix, tab_flow, tab_guide = st.tabs(["🚀 Dashboard", "📊 Impact Matrix", "🌊 Liquidity Flow", "📖 Playbook"])

    # === TAB 1: COMPACT DASHBOARD ===
    with tab_dash:
        col_graph, col_action = st.columns([3, 1])
        
        with col_graph:
            st.plotly_chart(create_nexus_graph(market_data), use_container_width=True)
            
        with col_action:
            # Regime Box
            bg_col = analysis['color']
            st.markdown(f"""
            <div class="regime-box" style="background-color: {bg_col}22; border-color: {bg_col};">
                <div style="color: {bg_col}; font-weight: bold; font-size: 16px; margin-bottom: 5px;">{analysis['regime']}</div>
                <div style="font-size: 11px; color: #ccc;">{analysis['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action Plan
            st.markdown("##### 🎯 Action Plan")
            
            with st.container():
                st.markdown(f"**BUY / LONG**")
                if analysis['longs']:
                    for item in analysis['longs']:
                        st.markdown(f"- <span style='color: #4ade80;'>{item}</span>", unsafe_allow_html=True)
                else:
                    st.caption("No signals")

                st.markdown(f"**AVOID / SHORT**")
                if analysis['shorts']:
                    for item in analysis['shorts']:
                        st.markdown(f"- <span style='color: #f87171;'>{item}</span>", unsafe_allow_html=True)
                else:
                    st.caption("No signals")
            
            if analysis['alerts']:
                st.markdown("---")
                st.error(analysis['alerts'][0], icon="🚨")

    # === TAB 2: IMPACT MATRIX ===
    with tab_matrix:
        st.markdown("### 🧠 Correlation Heatmap")
        st.caption("How specific Drivers (Rows) impact specific Assets (Columns).")
        
        matrix_data = {
            "Driver": ["Fed Liquidity", "Real Yields (Rates)", "US Dollar (DXY)", "Credit (HYG)", "Global Growth"],
            "Tech / Crypto": ["High Positive (Bullish)", "High Negative (Bearish)", "Low Negative", "High Positive", "Neutral"],
            "Gold": ["Medium Positive", "High Negative (Bearish)", "High Negative (Inverse)", "Neutral", "Medium Positive"],
            "Emerging Mkts": ["Medium Positive", "Negative", "High Negative (Toxic)", "High Positive", "High Positive"],
            "Banks / Energy": ["Neutral", "Medium Positive", "Negative", "High Positive", "High Positive (Proxy)"]
        }
        df = pd.DataFrame(matrix_data).set_index("Driver")
        st.dataframe(df, use_container_width=True)

    # === TAB 3: FLOW ===
    with tab_flow:
        st.markdown("### 🌊 The Transmission Mechanism")
        try:
            graph = graphviz.Digraph()
            graph.attr(rankdir='TB', bgcolor='transparent')
            graph.attr('node', shape='box', style='filled', fontname='Helvetica', fontcolor='white')
            graph.attr('edge', color='#6b7280')

            graph.node('FED', '🏦 FED & TREASURY\n(Liquidity Source)', fillcolor='#4f46e5')
            graph.node('YIELDS', 'US 10Y YIELDS\n(Cost of Money)', fillcolor='#ef4444')
            graph.node('DXY', 'US DOLLAR\n(Global Collateral)', fillcolor='#3b82f6')
            graph.node('CREDIT', 'CREDIT (HYG)\n(Risk Appetite)', fillcolor='#a855f7')
            graph.node('TECH', 'TECH / CRYPTO\n(Long Duration)', fillcolor='#1f2937')
            graph.node('GOLD', 'GOLD / COMMOD.\n(Real Assets)', fillcolor='#1f2937')
            
            graph.edge('FED', 'YIELDS'); graph.edge('FED', 'DXY'); graph.edge('FED', 'TECH')
            graph.edge('YIELDS', 'TECH'); graph.edge('YIELDS', 'GOLD')
            graph.edge('DXY', 'GOLD')
            graph.edge('CREDIT', 'TECH')

            st.graphviz_chart(graph, use_container_width=True)
        except:
            st.error("Graphviz not installed on server. Please install 'graphviz' to view this diagram.")

    # === TAB 4: PLAYBOOK ===
    with tab_guide:
        st.markdown("""
        ## 📖 Trader's Daily Workflow
        ### 1. The Morning Veto (5 Mins)
        * Look at **Credit (Veto)**. If **Red (> -0.3%)**, STOP. Do not buy dip.
        
        ### 2. Identify Regime
        * **Liquidity Pump:** Crypto/Tech Leading. -> **Buy Beta.**
        * **Reflation:** Oil/Yields Green. -> **Buy Real Assets.**
        * **Risk Off:** VIX Green. -> **Cash.**
        """)

if __name__ == "__main__":
    main()
