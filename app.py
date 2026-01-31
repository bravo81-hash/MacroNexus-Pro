import streamlit as st
import yfinance as yf
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="MacroNexus Pro",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    # Extract changes safely
    def get_c(k): return data.get(k, {}).get('change', 0)
    
    hyg, vix, oil, cop, us10y, dxy, btc = get_c('HYG'), get_c('VIX'), get_c('OIL'), get_c('COPPER'), get_c('US10Y'), get_c('DXY'), get_c('BTC')

    # Default State
    regime = "NEUTRAL / CHOP"
    desc = "No clear macro dominance. Correlations may be weak. Trade setups only."
    color_code = "gray"
    longs, shorts, alerts = [], [], []

    # Logic Hierarchy (Veto first)
    if hyg < -0.3 or vix > 2.0:
        regime = "RISK OFF (DEFENSIVE)"
        desc = "Credit spreads widening or Volatility spiking. Cash is King."
        color_code = "red"
        longs = ["Cash (UUP)", "Volatility (VIX)"]
        shorts = ["Tech", "Crypto", "Small Caps", "EM"]
        alerts.append("⛔ CRITICAL: Credit Stress Detected. VETO all long risk trades.")

    elif (oil > 0.5 or cop > 0.5) and us10y > 1.0:
        regime = "REFLATION (GROWTH + INFLATION)"
        desc = "Commodities and Rates rising together. Growth is leading."
        color_code = "orange"
        longs = ["Energy", "Industrials", "Banks"]
        shorts = ["Bonds (TLT)", "Tech (Rate Sensitive)"]
        alerts.append("🔥 INFLATION PULSE: Rotate to Real Assets.")

    elif dxy < -0.1 and btc > 1.0:
        regime = "LIQUIDITY PUMP (RISK ON)"
        desc = "Dollar weakness fueling high-beta assets."
        color_code = "purple"
        longs = ["Bitcoin", "Nasdaq", "Gold"]
        shorts = ["Dollar (UUP)"]
        alerts.append("🌊 LIQUIDITY ON: Green light for High Beta.")

    elif vix < 0 and abs(us10y) < 2.0:
        regime = "GOLDILOCKS (STABLE)"
        desc = "Low volatility, stable rates. Favorable for equities."
        color_code = "green"
        longs = ["S&P 500", "Tech", "Quality Growth"]
        shorts = ["Volatility"]
        alerts.append("✅ STABLE: Buy the dip environment.")

    return {
        'regime': regime, 'desc': desc, 'color': color_code,
        'longs': longs, 'shorts': shorts, 'alerts': alerts
    }

# --- 3. GRAPHICS ENGINE ---
def create_nexus_graph(market_data):
    nodes = {
        'US10Y': {'pos': (0, 0), 'label': 'US 10Y'},
        'DXY':   {'pos': (1.2, 1.2), 'label': 'USD (UUP)'},
        'SPY':   {'pos': (-1.2, 1.2), 'label': 'S&P 500'},
        'QQQ':   {'pos': (-1.8, 0.5), 'label': 'Nasdaq'},
        'GOLD':  {'pos': (1.2, -1.2), 'label': 'Gold'},
        'HYG':   {'pos': (-0.5, -1.2), 'label': 'Credit'},
        'BTC':   {'pos': (-2.2, 2.2), 'label': 'Bitcoin'},
        'OIL':   {'pos': (2.2, -0.5), 'label': 'Oil'},
        'COPPER':{'pos': (1.8, -1.8), 'label': 'Copper'},
        'IWM':   {'pos': (-1.8, -1.5), 'label': 'Russell'}
    }
    
    # Create Edges (Lines)
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

    # Create Nodes (Bubbles)
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    
    for key, info in nodes.items():
        x, y = info['pos']
        node_x.append(x)
        node_y.append(y)
        chg = market_data.get(key, {}).get('change', 0)
        
        # Color Logic
        col = '#22c55e' if chg > 0 else '#ef4444' # Green/Red
        if key in ['US10Y', 'DXY', 'VIX']: 
             # Inverse visual for "Bad" up moves? Let's stick to price action for consistency
             pass 
        if chg == 0: col = '#6b7280' # Gray

        node_color.append(col)
        node_size.append(50 if key in ['US10Y', 'DXY', 'HYG'] else 40)
        node_text.append(f"<b>{info['label']}</b><br>{chg:+.2f}%")

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#4b5563'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="bottom center",
                             marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')),
                             textfont=dict(size=14, color='white')))
    
    # Fix Clipping with Range
    fig.update_layout(
        showlegend=False, margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]), # FIXED RANGE
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]), # FIXED RANGE
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=550
    )
    return fig

# --- 4. APP LAYOUT ---
def main():
    # --- HEADER ---
    st.title("🌐 MacroNexus Pro")
    st.markdown("Global Macro Command Center • _Live Data_")
    
    with st.spinner("Connecting to Global Markets..."):
        market_data = fetch_live_data()
        analysis = analyze_market(market_data)

    # --- TABS INTERFACE ---
    tab1, tab2, tab3 = st.tabs(["🚀 Dashboard", "🧠 Deep Dive & Matrix", "📖 Playbook"])

    # === TAB 1: DASHBOARD ===
    with tab1:
        # Regime Banner using Native Components (No Custom HTML)
        regime_container = st.container(border=True)
        with regime_container:
            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                st.subheader(f"Regime: {analysis['regime']}")
                st.caption(analysis['desc'])
            with col_r2:
                # Dynamic Alert Box
                if "RISK OFF" in analysis['regime']:
                    st.error("⚠️ RISK OFF ALERT")
                elif "GOLDILOCKS" in analysis['regime']:
                    st.success("✅ BUY DIP MODE")
                elif "NEUTRAL" in analysis['regime']:
                    st.info("ℹ️ CHOPPY / NEUTRAL")
                else:
                    st.warning("⚡ HIGH VOLATILITY")

        # Key Metrics
        c1, c2, c3, c4 = st.columns(4)
        def kpi(label, key):
            d = market_data.get(key, {})
            c1.metric(label, f"{d.get('price',0):.2f}", f"{d.get('change',0):+.2f}%")
        
        with c1: kpi("Credit (Veto)", "HYG")
        with c2: kpi("Volatility", "VIX")
        with c3: kpi("10Y Yield", "US10Y")
        with c4: kpi("Liquidity (BTC)", "BTC")

        # Graph & Action
        c_left, c_right = st.columns([1, 2])
        
        with c_left:
            st.write("#### 🎯 Action Plan")
            if analysis['longs']:
                st.success(f"**FOCUS (LONG):**\n\n{', '.join(analysis['longs'])}")
            else:
                st.info("No clear Longs")
                
            if analysis['shorts']:
                st.error(f"**AVOID (SHORT):**\n\n{', '.join(analysis['shorts'])}")
            else:
                st.info("No clear Shorts")
                
            st.divider()
            st.write("**Active Correlations:**")
            us10y_c = market_data['US10Y']['change']
            dxy_c = market_data['DXY']['change']
            
            if us10y_c > 0.5: st.warning("📉 Yields Spiking: Tech Headwind")
            elif us10y_c < -0.5: st.success("📈 Yields Falling: Tech Tailwind")
            
            if dxy_c > 0.2: st.warning("📉 Dollar Strong: Gold/EM Headwind")
            elif dxy_c < -0.2: st.success("📈 Dollar Weak: Gold/EM Tailwind")

        with c_right:
            st.plotly_chart(create_nexus_graph(market_data), use_container_width=True)

    # === TAB 2: DEEP DIVE & MATRIX ===
    with tab2:
        st.subheader("🔍 Asset Deep Dive")
        selected_asset = st.selectbox("Select Asset to Analyze:", list(TICKERS.keys()))
        
        # Logic for deep dive
        if selected_asset:
            d = market_data[selected_asset]
            st.metric(f"{selected_asset} Performance", f"{d['price']:.2f}", f"{d['change']:+.2f}%")
            
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                st.markdown("### 🔗 Primary Drivers")
                if selected_asset == 'GOLD':
                    st.write("- **Real Yields (Inverse):** If Yields UP, Gold DOWN.")
                    st.write("- **Dollar (Inverse):** If DXY UP, Gold DOWN.")
                elif selected_asset == 'QQQ':
                    st.write("- **Yields (Inverse):** Rates hit valuations.")
                    st.write("- **Liquidity (Direct):** Loves Fed Balance Sheet.")
                elif selected_asset == 'BTC':
                    st.write("- **Liquidity (Direct):** Purest liquidity proxy.")
                    st.write("- **Nasdaq (Direct):** Often moves with Tech.")
                else:
                    st.write("Select Gold, QQQ, or BTC for detailed driver breakdown.")
            
            with c_d2:
                st.markdown("### ⚠️ Veto Check")
                hyg_c = market_data['HYG']['change']
                if hyg_c < -0.2:
                    st.error(f"Credit Markets (HYG) are DOWN ({hyg_c:.2f}%). **WARNING:** The floor is weak.")
                else:
                    st.success(f"Credit Markets (HYG) are Stable ({hyg_c:.2f}%). **SAFE:** The floor is holding.")

        st.divider()
        st.subheader("📊 Correlation Matrix (The Cheat Sheet)")
        
        # Static Matrix Data for Display
        matrix_data = {
            "Asset": ["Tech (QQQ)", "Crypto (BTC)", "Gold", "Banks", "Emerging Mkts"],
            "vs Yields": ["High Inverse", "Medium Inverse", "High Inverse", "Positive", "Neutral"],
            "vs Dollar": ["Low Inverse", "Low Inverse", "High Inverse", "Neutral", "High Inverse"],
            "vs Liquidity": ["High Direct", "High Direct", "Medium Direct", "Neutral", "Medium Direct"]
        }
        df_matrix = pd.DataFrame(matrix_data)
        st.dataframe(df_matrix, use_container_width=True)

    # === TAB 3: PLAYBOOK ===
    with tab3:
        st.markdown("""
        ## 📖 Trader's Daily Workflow
        
        ### 1. The Morning Veto (5 Mins)
        * Look at **Credit (HYG)** in the dashboard.
        * **If Red:** Stop. Do not buy the dip. The market foundation is cracking.
        * **If Green/Gray:** Proceed.
        
        ### 2. Identify The Weather (Regime)
        * **Liquidity Pump:** Bitcoin & Tech are leading. Dollar is red. -> **Buy Beta.**
        * **Reflation:** Oil & Yields are green. Tech is red. -> **Buy Energy/Banks.**
        * **Risk Off:** VIX is green. Everything else is red. -> **Go to Cash.**
        
        ### 3. The Trade Confirmation
        Before you click "Buy" on your broker:
        * **Buying Gold?** Check `US10Y`. Is it crashing? If not, wait.
        * **Buying Tech?** Check `Liquidity`. Is it rising? If not, wait.
        """)

if __name__ == "__main__":
    main()
