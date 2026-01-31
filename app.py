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
    initial_sidebar_state="collapsed" # Collapsed to give more screen space
)

# Custom CSS for Compact Layout
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    /* Compact Metric Card */
    .metric-container {
        background-color: #1e2127;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #4b5563;
        margin-bottom: 5px;
    }
    .metric-label { font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }
    .metric-val { font-size: 20px; font-weight: bold; color: #f3f4f6; }
    .metric-chg { font-size: 12px; font-weight: bold; }
    
    /* Remove extra padding */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    
    /* Regime Badge */
    .regime-badge {
        padding: 8px; border-radius: 6px; text-align: center; font-weight: bold; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA FETCHING (Robust Proxy Tickers) ---
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

# --- 2. LOGIC ENGINE ---
def analyze_market(data):
    """Determines Regime and Signals."""
    if not data: return None
    
    def get_c(k): return data.get(k, {}).get('change', 0)
    
    hyg, vix, oil, cop, us10y, dxy, btc = get_c('HYG'), get_c('VIX'), get_c('OIL'), get_c('COPPER'), get_c('US10Y'), get_c('DXY'), get_c('BTC')

    regime = "NEUTRAL"
    desc = "No clear trend. Trade setups only."
    color_code = "gray"
    longs, shorts, alerts = [], [], []

    # Logic Hierarchy
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

    return {
        'regime': regime, 'desc': desc, 'color': color_code,
        'longs': longs, 'shorts': shorts, 'alerts': alerts
    }

# --- 3. GRAPHICS ENGINE ---
def create_nexus_graph(market_data):
    # Solar System Layout: Center (Gravity), Inner (Assets), Outer (High Beta)
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
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450 # Reduced height for compact view
    )
    return fig

# --- 4. APP LAYOUT ---
def main():
    with st.spinner("Connecting to Markets..."):
        market_data = fetch_live_data()
        analysis = analyze_market(market_data)

    # Top Bar: Key Metrics (Compact)
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
            <div class="metric-val">{val:.2f} <span class="metric-chg" style="color: {color};">{chg:+.2f}%</span></div>
        </div>
        """
    
    c1.markdown(metric_html("Credit (HYG)", "HYG"), unsafe_allow_html=True)
    c2.markdown(metric_html("Vol (VIX)", "VIX"), unsafe_allow_html=True)
    c3.markdown(metric_html("Yields", "US10Y"), unsafe_allow_html=True)
    c4.markdown(metric_html("Dollar", "DXY"), unsafe_allow_html=True)
    c5.markdown(metric_html("Liquidity", "BTC"), unsafe_allow_html=True)

    # Main Tabs
    tab_dash, tab_matrix, tab_flow, tab_guide = st.tabs(["🚀 Dashboard", "📊 Impact Matrix", "🌊 Liquidity Flow", "📖 Playbook"])

    # === TAB 1: COMPACT DASHBOARD ===
    with tab_dash:
        col_graph, col_action = st.columns([3, 1])
        
        with col_graph:
            st.plotly_chart(create_nexus_graph(market_data), use_container_width=True)
            
        with col_action:
            # Regime Badge
            bg_color = analysis['color']
            st.markdown(f"""
            <div style="background-color: {bg_color}33; border: 1px solid {bg_color}; padding: 10px; border-radius: 5px; text-align: center;">
                <strong style="color: {bg_color}; font-size: 14px;">{analysis['regime']}</strong><br>
                <span style="font-size: 10px; color: #ccc;">{analysis['desc']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("**🎯 BUY (LONG)**")
            if analysis['longs']:
                for item in analysis['longs']:
                    st.markdown(f"<span style='color: #4ade80; font-weight: bold;'>• {item}</span>", unsafe_allow_html=True)
            else:
                st.caption("No clear longs")

            st.markdown("**⛔ AVOID (SHORT)**")
            if analysis['shorts']:
                for item in analysis['shorts']:
                    st.markdown(f"<span style='color: #f87171; font-weight: bold;'>• {item}</span>", unsafe_allow_html=True)
            else:
                st.caption("No clear shorts")
                
            if analysis['alerts']:
                st.markdown("---")
                st.error(analysis['alerts'][0])

    # === TAB 2: IMPACT MATRIX ===
    with tab_matrix:
        st.markdown("### 🧠 Correlation Heatmap")
        st.caption("How specific Drivers (Rows) impact specific Assets (Columns).")
        
        # Define Data
        matrix_data = {
            "Driver": ["Fed Liquidity", "Real Yields (Rates)", "US Dollar (DXY)", "Credit (HYG)", "Global Growth"],
            "Tech / Crypto": ["High Positive (Bullish)", "High Negative (Bearish)", "Low Negative", "High Positive", "Neutral"],
            "Gold": ["Medium Positive", "High Negative (Bearish)", "High Negative (Inverse)", "Neutral", "Medium Positive"],
            "Emerging Mkts": ["Medium Positive", "Negative", "High Negative (Toxic)", "High Positive", "High Positive"],
            "Banks / Energy": ["Neutral", "Medium Positive", "Negative", "High Positive", "High Positive (Proxy)"]
        }
        
        df = pd.DataFrame(matrix_data).set_index("Driver")
        st.dataframe(df, use_container_width=True)
        
        st.info("💡 **How to read:** If 'Real Yields' are rising (Red row), look at the 'Gold' column. It says 'High Negative', meaning Gold will likely fall.")

    # === TAB 3: LIQUIDITY FLOW ===
    with tab_flow:
        st.markdown("### 🌊 The Transmission Mechanism")
        st.caption("Visualizing how Fed decisions cascade down to your portfolio.")
        
        graph = graphviz.Digraph()
        graph.attr(rankdir='TB', bgcolor='transparent')
        graph.attr('node', shape='box', style='filled', fontname='Helvetica', fontcolor='white')
        graph.attr('edge', color='#6b7280')

        # Level 1
        graph.node('FED', '🏦 FED & TREASURY\n(Liquidity Source)', fillcolor='#4f46e5')
        
        # Level 2
        graph.node('YIELDS', 'US 10Y YIELDS\n(Cost of Money)', fillcolor='#ef4444')
        graph.node('DXY', 'US DOLLAR\n(Global Collateral)', fillcolor='#3b82f6')
        graph.node('CREDIT', 'CREDIT (HYG)\n(Risk Appetite)', fillcolor='#a855f7')
        
        # Level 3
        graph.node('TECH', 'TECH / CRYPTO\n(Long Duration)', fillcolor='#1f2937')
        graph.node('GOLD', 'GOLD / COMMOD.\n(Real Assets)', fillcolor='#1f2937')
        graph.node('EM', 'EMERGING MKTS\n(Dollar Sensitive)', fillcolor='#1f2937')
        graph.node('REAL', 'BANKS / ENERGY\n(Growth Sensitive)', fillcolor='#1f2937')

        # Connections
        graph.edge('FED', 'YIELDS', label='Rates')
        graph.edge('FED', 'DXY', label='Tightening')
        graph.edge('FED', 'TECH', label='QE / TGA')
        
        graph.edge('YIELDS', 'TECH', label='Discount Rate')
        graph.edge('YIELDS', 'GOLD', label='Opp. Cost')
        
        graph.edge('DXY', 'EM', label='Debt Squeeze')
        graph.edge('DXY', 'GOLD', label='Denominator')
        
        graph.edge('CREDIT', 'TECH', label='Correlated')
        graph.edge('CREDIT', 'REAL', label='Correlated')

        st.graphviz_chart(graph, use_container_width=True)

    # === TAB 4: PLAYBOOK ===
    with tab_guide:
        st.markdown("""
        ## 📖 Trader's Daily Workflow
        
        ### 1. The Morning Veto (5 Mins)
        * Look at **Credit (HYG)** in the dashboard (Top Left Metric).
        * **If Red (> -0.3%):** Stop. Do not buy the dip. The market foundation is cracking.
        * **If Green/Gray:** Proceed to Step 2.
        
        ### 2. Identify The Weather (Regime)
        Check the colored badge on the Dashboard tab.
        * **Liquidity Pump (Purple):** Dollar is weak, Crypto is strong. -> **Buy High Beta.**
        * **Reflation (Orange):** Oil & Yields are green. Tech is red. -> **Buy Energy/Banks.**
        * **Risk Off (Red):** VIX is green. Everything else is red. -> **Go to Cash.**
        * **Goldilocks (Green):** VIX is red. Yields stable. -> **Buy Quality Tech.**
        
        ### 3. The Trade Confirmation
        Before you click "Buy" on your broker:
        * **Buying Gold?** Check `Yields`. Are they crashing? If not, wait.
        * **Buying Tech?** Check `Liquidity`. Is BTC rising? If not, wait.
        """)

if __name__ == "__main__":
    main()
