import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import graphviz

# --- CONFIGURATION ---
st.set_page_config(
    page_title="MacroNexus Pro",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (DENSE & RICH) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Fixed Metric Card */
    .metric-container {
        background-color: #1e2127;
        padding: 10px 12px;
        border-radius: 6px;
        border-left: 4px solid #4b5563;
        margin-bottom: 8px;
        min-height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .metric-label { 
        font-size: 10px; 
        color: #9ca3af; 
        text-transform: uppercase; 
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-ticker {
        font-size: 9px;
        color: #6b7280;
        font-family: monospace;
        background: #262730;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .metric-val { 
        font-size: 18px; 
        font-weight: bold; 
        color: #f3f4f6; 
    }
    .metric-chg { 
        font-size: 12px; 
        font-weight: bold; 
        margin-left: 6px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0e1117;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. EXPANDED DATA UNIVERSE ---
# Using liquid ETFs/Proxies for reliability on Streamlit Cloud
TICKERS = {
    # DRIVERS
    'US10Y': '^TNX',       # 10Y Yield
    'DXY': 'UUP',          # Dollar Proxy
    'VIX': 'VIXY',         # Volatility Proxy
    'HYG': 'HYG',          # Credit High Yield
    
    # INDICES
    'SPY': 'SPY',          # S&P 500
    'QQQ': 'QQQ',          # Nasdaq
    'IWM': 'IWM',          # Russell 2000
    'EEM': 'EEM',          # Emerging Markets
    'FXI': 'FXI',          # China Large Cap
    'EWJ': 'EWJ',          # Japan
    
    # COMMODITIES
    'GOLD': 'GLD',         # Gold
    'SILVER': 'SLV',       # Silver
    'OIL': 'USO',          # Oil
    'NATGAS': 'UNG',       # Natural Gas
    'COPPER': 'CPER',      # Copper
    'AG': 'DBA',           # Agriculture
    
    # SECTORS & THEMES
    'TECH': 'XLK',         # Tech Sector
    'SEMIS': 'SMH',        # Semiconductors
    'BANKS': 'XLF',        # Financials
    'ENERGY': 'XLE',       # Energy Stocks
    'HOME': 'XHB',         # Homebuilders
    'TLT': 'TLT',          # 20Y Bonds
    
    # FOREX & CRYPTO
    'EURO': 'FXE',         # Euro
    'YEN': 'FXY',          # Yen
    'BTC': 'BTC-USD'       # Bitcoin
}

@st.cache_data(ttl=300)
def fetch_live_data():
    """Fetches data individually."""
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
    if not data: return None
    def get_c(k): return data.get(k, {}).get('change', 0)
    
    hyg, vix, oil, cop, us10y, dxy, btc = get_c('HYG'), get_c('VIX'), get_c('OIL'), get_c('COPPER'), get_c('US10Y'), get_c('DXY'), get_c('BTC')

    regime = "NEUTRAL"
    desc = "No clear macro dominance. Following momentum."
    color_code = "#6b7280"
    longs, shorts, alerts = [], [], []

    # 1. MACRO REGIME CHECKS
    if hyg < -0.3 or vix > 2.0:
        regime = "RISK OFF"
        desc = "Credit widening or Vol spiking. Cash is King."
        color_code = "#ef4444"
        longs = ["Cash (UUP)", "Vol (VIXY)", "Bonds (TLT)"]
        shorts = ["Tech", "Crypto", "Small Caps", "EM", "High Yield"]
        alerts.append("⛔ CREDIT STRESS: Veto Longs.")

    elif (oil > 0.5 or cop > 0.5) and us10y > 1.0:
        regime = "REFLATION"
        desc = "Growth + Inflation rising."
        color_code = "#f59e0b"
        longs = ["Energy (XLE)", "Banks (XLF)", "Industrials", "Commodities"]
        shorts = ["Bonds (TLT)", "Tech (Rate Sensitive)", "Homebuilders (XHB)"]
        alerts.append("🔥 INFLATION: Rotate to Real Assets.")

    elif dxy < -0.1 and btc > 1.0:
        regime = "LIQUIDITY PUMP"
        desc = "Dollar weak, Crypto leading."
        color_code = "#a855f7"
        longs = ["Bitcoin", "Nasdaq (QQQ)", "Semis (SMH)", "Gold"]
        shorts = ["Dollar (UUP)"]
        alerts.append("🌊 LIQUIDITY: Risk On High Beta.")

    elif vix < 0 and abs(us10y) < 2.0:
        regime = "GOLDILOCKS"
        desc = "Low vol, stable rates."
        color_code = "#22c55e"
        longs = ["S&P 500", "Tech (XLK)", "Semis (SMH)"]
        shorts = ["Volatility"]
        alerts.append("✅ STABLE: Buy Dips.")

    # 2. FALLBACK MOMENTUM
    if not longs:
        # Sort full universe
        all_keys = list(TICKERS.keys())
        # Exclude inverse logic assets for mom sort
        valid_keys = [k for k in all_keys if k not in ['VIX', 'DXY', 'US10Y']]
        assets = {k: get_c(k) for k in valid_keys}
        sorted_assets = sorted(assets.items(), key=lambda x: x[1], reverse=True)
        
        longs = [f"{k} (Mom.)" for k, v in sorted_assets[:3] if v > 0]
        shorts = [f"{k} (Mom.)" for k, v in sorted_assets[-3:] if v < 0]
        
        if not longs: longs = ["Cash"]
        if not shorts: shorts = ["None"]

    return {
        'regime': regime, 'desc': desc, 'color': color_code,
        'longs': longs, 'shorts': shorts, 'alerts': alerts
    }

# --- 3. GRAPHICS ENGINES ---

def create_nexus_graph(market_data):
    # Expanded Solar System Layout
    nodes = {
        'US10Y': {'pos': (0, 0), 'label': 'Rates'},
        'DXY':   {'pos': (0.8, 0.8), 'label': 'Dollar'},
        'SPY':   {'pos': (-0.8, 0.8), 'label': 'S&P 500'},
        'QQQ':   {'pos': (-1.2, 0.4), 'label': 'Nasdaq'},
        'GOLD':  {'pos': (0.8, -0.8), 'label': 'Gold'},
        'HYG':   {'pos': (-0.4, -0.8), 'label': 'Credit'},
        'BTC':   {'pos': (-1.5, 1.5), 'label': 'Bitcoin'},
        'OIL':   {'pos': (1.5, -0.4), 'label': 'Oil'},
        'COPPER':{'pos': (1.2, -1.2), 'label': 'Copper'},
        'IWM':   {'pos': (-1.2, -1.0), 'label': 'Russell'},
        'SMH':   {'pos': (-1.8, 0.8), 'label': 'Semis'},
        'XLE':   {'pos': (1.8, -0.8), 'label': 'Energy'},
        'EEM':   {'pos': (-0.5, -1.5), 'label': 'EM'},
        'XHB':   {'pos': (-0.8, -0.4), 'label': 'Housing'}
    }
    
    edges = [
        ('US10Y', 'QQQ'), ('US10Y', 'GOLD'), ('US10Y', 'XHB'),
        ('DXY', 'GOLD'), ('DXY', 'OIL'), ('DXY', 'EEM'),
        ('HYG', 'SPY'), ('HYG', 'IWM'), 
        ('QQQ', 'BTC'), ('QQQ', 'SMH'),
        ('COPPER', 'US10Y'), ('OIL', 'XLE')
    ]
    
    edge_x, edge_y = [], []
    for u, v in edges:
        if u in nodes and v in nodes:
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
        # Using Ticker in label for clarity
        ticker = TICKERS.get(key, key)
        node_text.append(f"<b>{info['label']}</b><br>({ticker})<br>{chg:+.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#4b5563'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="bottom center",
                             marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')),
                             textfont=dict(size=11, color='white')))
    
    fig.update_layout(
        showlegend=False, margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.0, 2.0]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500
    )
    return fig

def create_heatmap_matrix():
    # Define detailed correlations (-1 to 1)
    data = [
        # Tech, Crypto, Gold, EM, Energy, Banks, Housing
        [ 0.9,  0.9,  0.4, 0.6,  0.1,  0.2,  0.7], # Fed Liquidity
        [-0.8, -0.6, -0.9, -0.3, 0.4,  0.6, -0.9], # Real Yields
        [-0.4, -0.5, -0.9, -0.9, -0.6, -0.1, -0.2], # Dollar
        [ 0.8,  0.7,  0.1, 0.8,  0.6,  0.9,  0.7], # Credit (HYG)
        [ 0.2,  0.3,  0.5, 0.9,  0.9,  0.8,  0.4]  # Growth
    ]
    
    x_labels = ['Tech (QQQ)', 'Crypto (BTC)', 'Gold (GLD)', 'Emerging (EEM)', 'Energy (XLE)', 'Banks (XLF)', 'Home (XHB)']
    y_labels = ['Fed Liquidity', 'Real Yields', 'Dollar (DXY)', 'Credit (HYG)', 'Global Growth']

    fig = px.imshow(
        data, 
        x=x_labels, 
        y=y_labels,
        color_continuous_scale=['red', '#1e1e24', 'green'], # Red to Dark to Green
        range_color=[-1, 1],
        text_auto=True,
        aspect="auto"
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    return fig

# --- 4. APP LAYOUT ---
def main():
    with st.spinner("Initializing MacroNexus Pro..."):
        market_data = fetch_live_data()
        analysis = analyze_market(market_data)

    # --- TOP METRICS GRID (EXPANDED) ---
    st.markdown("### 📡 Market Drivers")
    cols = st.columns(6) # 6 columns for dense data
    
    # Helper for rendering tiles
    def tile(c, label, key):
        d = market_data.get(key, {})
        val = d.get('price', 0)
        chg = d.get('change', 0)
        sym = d.get('symbol', key)
        color = "#ef4444" if chg < 0 else "#22c55e"
        if key in ['VIX', 'US10Y', 'DXY']: color = "#ef4444" if chg > 0 else "#22c55e"
        
        c.markdown(f"""
        <div class="metric-container" style="border-left-color: {color};">
            <div class="metric-header">
                <span class="metric-label">{label}</span>
                <span class="metric-ticker">{sym}</span>
            </div>
            <div>
                <span class="metric-val">{val:.2f}</span>
                <span class="metric-chg" style="color: {color};">{chg:+.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Row 1: The Plumbing
    tile(cols[0], "Credit", "HYG")
    tile(cols[1], "Volatility", "VIX")
    tile(cols[2], "10Y Yield", "US10Y")
    tile(cols[3], "Dollar", "DXY")
    tile(cols[4], "Oil", "OIL")
    tile(cols[5], "Copper", "COPPER")

    # --- TABS ---
    tab_dash, tab_matrix, tab_flow, tab_guide = st.tabs(["🚀 Dashboard", "📊 Visual Heatmap", "🌊 Liquidity Flow", "📖 Full Playbook"])

    # === TAB 1: DASHBOARD ===
    with tab_dash:
        col_graph, col_action = st.columns([2.5, 1])
        
        with col_graph:
            st.plotly_chart(create_nexus_graph(market_data), use_container_width=True)
            
        with col_action:
            # Regime Box
            bg_col = analysis['color']
            st.markdown(f"""
            <div style="background-color: {bg_col}22; border: 1px solid {bg_col}; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px;">
                <div style="color: #9ca3af; font-size: 10px; text-transform: uppercase; margin-bottom: 5px;">Current Regime</div>
                <div style="color: {bg_col}; font-weight: bold; font-size: 20px; margin-bottom: 5px;">{analysis['regime']}</div>
                <div style="font-size: 11px; color: #ccc;">{analysis['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### 🎯 Strategy")
            
            st.markdown(f"**🟢 LONG FOCUS**")
            if analysis['longs']:
                for item in analysis['longs']:
                    st.markdown(f"- {item}")
            else:
                st.caption("No clear longs")

            st.markdown(f"**🔴 AVOID / SHORT**")
            if analysis['shorts']:
                for item in analysis['shorts']:
                    st.markdown(f"- {item}")
            else:
                st.caption("No clear shorts")
                
            if analysis['alerts']:
                st.markdown("---")
                st.error(analysis['alerts'][0], icon="🚨")

    # === TAB 2: HEATMAP ===
    with tab_matrix:
        st.markdown("### 🧠 Cross-Asset Correlation Matrix")
        st.caption("Visualizing the impact of macro drivers (Rows) on asset classes (Columns). Green = Positive, Red = Inverse.")
        st.plotly_chart(create_heatmap_matrix(), use_container_width=True)
        
        st.info("""
        **How to read this Heatmap:**
        * **Real Yields vs Gold (-0.9):** Deep Red. If Real Yields go UP, Gold goes DOWN hard.
        * **Liquidity vs Crypto (+0.9):** Deep Green. If Fed adds Liquidity, Crypto flies.
        * **Dollar vs Emerging Markets (-0.9):** Deep Red. Strong Dollar crushes EM.
        """)

    # === TAB 3: FLOW ===
    with tab_flow:
        st.markdown("### 🌊 The Macro Transmission Mechanism")
        try:
            graph = graphviz.Digraph()
            graph.attr(rankdir='TB', bgcolor='transparent')
            graph.attr('node', shape='box', style='filled, rounded', fontname='Helvetica', fontcolor='white', penwidth='0')
            graph.attr('edge', color='#6b7280', arrowsize='0.8')

            # Level 1: The Source
            graph.node('FED', '🏦 FED & TREASURY\n(Liquidity Source)', fillcolor='#4f46e5')
            
            # Level 2: Transmission
            graph.node('YIELDS', 'US 10Y YIELDS\n(Cost of Money)', fillcolor='#b91c1c')
            graph.node('DXY', 'US DOLLAR\n(Global Collateral)', fillcolor='#1e3a8a')
            graph.node('CREDIT', 'CREDIT (HYG)\n(Risk Appetite)', fillcolor='#7e22ce')
            
            # Level 3: Assets
            graph.node('TECH', 'TECH / CRYPTO\n(Long Duration)', fillcolor='#1f2937')
            graph.node('GOLD', 'GOLD / COMMOD.\n(Real Assets)', fillcolor='#1f2937')
            graph.node('EM', 'EMERGING MKTS\n(Dollar Sensitive)', fillcolor='#1f2937')
            graph.node('CYCL', 'BANKS / ENERGY\n(Growth Sensitive)', fillcolor='#1f2937')
            
            # Level 4: Sectors (New Detail)
            graph.node('SEMI', 'Semis (SMH)', fillcolor='#111827', fontsize='10')
            graph.node('HOME', 'Housing (XHB)', fillcolor='#111827', fontsize='10')
            graph.node('IND', 'Industrials (XLI)', fillcolor='#111827', fontsize='10')

            # Connections
            graph.edge('FED', 'YIELDS', label='Rates')
            graph.edge('FED', 'DXY', label='Tightening')
            graph.edge('FED', 'TECH', label='QE / TGA')
            
            graph.edge('YIELDS', 'TECH', label='Discount Rate')
            graph.edge('YIELDS', 'GOLD', label='Opp. Cost')
            graph.edge('YIELDS', 'HOME', label='Mortgage Rates')
            
            graph.edge('DXY', 'EM', label='Debt Squeeze')
            graph.edge('DXY', 'GOLD', label='Denominator')
            
            graph.edge('CREDIT', 'TECH', label='Correlated')
            graph.edge('CREDIT', 'CYCL', label='Correlated')
            
            graph.edge('TECH', 'SEMI', style='dashed')
            graph.edge('CYCL', 'IND', style='dashed')

            st.graphviz_chart(graph, use_container_width=True)
        except:
            st.error("Graphviz not installed on server. Please install 'graphviz' to view this diagram.")

    # === TAB 4: FULL PLAYBOOK ===
    with tab_guide:
        st.markdown("""
        # 📖 MacroNexus Pro: Daily Trader's Playbook

        This guide explains how to use the interactive map as a decision-support engine.

        ## ⏰ The 5-Minute Morning Routine

        Before you look at a single stock chart, open the MacroNexus and perform this "Health Check."

        ### 1. Diagnose the "Plumbing" (The Veto Check)

        **Goal:** Determine if it is safe to take risk today.

        * **Check `Credit (HYG)`** (First Tile)
          * *Question:* Is HYG Green (Stable) or Red (Falling)?
          * *Logic:* HYG measures corporate stress.
          * *Decision:* If HYG is Red (> -0.3%), **DO NOT** buy the dip in Stocks (`SPY`, `IWM`). The rally is likely a trap.

        * **Check `10Y Yields`**
          * *Question:* Are Yields spiking (> +1.0%)?
          * *Logic:* High yields kill "duration" assets (Gold, Tech, Crypto).
          * *Decision:* If Yields are surging, **DO NOT** go long Gold or Nasdaq today.

        ### 2. Identify the Regime (The Tailwind Check)

        **Goal:** Align your trades with the current wind direction. Check the **Regime Box** on the Dashboard.

        | **If Market Regime Is...** | **Actionable Strategy** | 
        | :--- | :--- |
        | **LIQUIDITY PUMP** | **Focus:** Crypto (`BTC`), Tech (`QQQ`). **Ignore:** Value stocks. | 
        | **RISK-OFF** | **Focus:** Cash (`UUP`), Volatility (`VIX`). **Avoid:** Small Caps (`IWM`), Emerging Markets. | 
        | **GOLDILOCKS** | **Focus:** Everything works, but `Semis` and `Tech` lead. **Buy the Dip.** | 
        | **REFLATION** | **Focus:** Energy (`XLE`), Banks (`XLF`). **Avoid:** Tech (`QQQ`) - it hates inflation. | 

        ## 🚦 "What To Do" vs. "What NOT To Do"

        The tool is best used to filter your ideas. Here are specific examples:

        ### Scenario A: You want to buy NVIDIA or Tech (QQQ)
        1. **Check `10Y Yields`:**
           * *Tool View:* Look at the `US10Y` tile.
           * *Verdict:* If Yields are GREEN (Up), the target (Nasdaq) usually goes DOWN.
           * *Action:* **WAIT.** Don't fight the Fed.

        ### Scenario B: You want to buy the dip in Crypto (BTC)
        1. **Check `Dollar (DXY)`:**
           * *Tool View:* Look at the `DXY` tile.
           * *Verdict:* If Dollar is Green (Strong), Crypto struggles.
           * *Action:* Only buy Crypto if Dollar is Red.

        ### Scenario C: You want to trade a "China Reopening" (FXI/Copper)
        1. **Check `Dollar (DXY)`:**
           * *Verdict:* A strong dollar crushes emerging markets (because of dollar-denominated debt).
           * *Action:* Only buy China if the DXY is weakening.

        ## ⚡ How to Spot Opportunity (Divergences)

        * **The "Coil" Setup:**
          * If **Copper** rips higher (Growth signal)...
          * But **Oil** and **Rates** haven't moved yet...
          * *Trade:* The market is lagging. Look for **Energy (`XLE`)** or **Industrials** to play catch-up.

        * **The "Fakeout" Setup:**
          * If **S&P 500** makes a new high...
          * But **Credit (HYG)** makes a lower high or is Red...
          * *Trade:* This is a bearish divergence. Credit isn't confirming the move. **Short the S&P 500.**

        ## 🛠 TradingView Integration Workflow

        1. **Open this Tool.**
        2. **Check the Regime.**
           * Is it "Risk Off"?
           * **YES:** Sit on hands or short.
           * **NO:** Proceed.
        3. **Cross-Reference TradingView:**
           * Open your chart. Look at your custom watchlist.
           * Does `AMEX:HYG` confirm what you see here?
           * If yes, execute the "Focus Long" list from the Dashboard.
        """)

if __name__ == "__main__":
    main()
