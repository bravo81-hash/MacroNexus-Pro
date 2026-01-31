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

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .metric-container {
        background-color: #1e2127;
        padding: 10px 12px;
        border-radius: 6px;
        border-left: 4px solid #4b5563;
        margin-bottom: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-header {
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;
    }
    .metric-label { 
        font-size: 10px; color: #9ca3af; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;
    }
    .metric-ticker {
        font-size: 9px; color: #6b7280; font-family: monospace; background: #262730; padding: 1px 4px; border-radius: 3px;
    }
    .metric-val { font-size: 18px; font-weight: bold; color: #f3f4f6; }
    .metric-chg { font-size: 12px; font-weight: bold; margin-left: 6px; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 20px; border-bottom: 1px solid #2e3039; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: 600; font-size: 14px; }
    .regime-badge { padding: 15px; border-radius: 8px; text-align: center; border: 1px solid; margin-bottom: 20px; background: #1e2127; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA UNIVERSE ---
# Primary tickers (Indices) with ETF fallbacks
TICKERS = {
    # DRIVERS
    'US10Y': '^TNX',       # 10Y Yield (CBOE) - Logic: Basis Points
    'DXY': 'DX-Y.NYB',     # Primary: Index. Fallback logic below.
    'VIX': '^VIX',         # Primary: Spot VIX. Fallback logic below.
    'HYG': 'HYG',          # Credit High Yield
    'TLT': 'TLT',          # 20Y Bonds
    'TIP': 'TIP',          # TIPS (for Real Yields check)
    
    # ASSETS
    'SPY': 'SPY', 'QQQ': 'QQQ', 'IWM': 'IWM',
    'EEM': 'EEM', 'FXI': 'FXI', 'EWJ': 'EWJ',
    'GOLD': 'GLD', 'SILVER': 'SLV', 'OIL': 'USO',
    'COPPER': 'CPER', 'NATGAS': 'UNG', 'AG': 'DBA',
    
    # SECTORS
    'TECH': 'XLK', 'SEMIS': 'SMH', 'BANKS': 'XLF',
    'ENERGY': 'XLE', 'HOME': 'XHB', 'UTIL': 'XLU',
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD'
}

# Fallbacks if Indices fail on cloud server
FALLBACKS = {
    'DXY': 'UUP', 
    'VIX': 'VIXY'
}

@st.cache_data(ttl=300)
def fetch_live_data():
    """Fetches data with robust holiday handling and fallback logic."""
    data_map = {}
    failed_tickers = []
    
    for key, symbol in TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            # Fetch 10 days to bridge holidays/weekends safely
            hist = ticker.history(period="10d")
            
            # Switch to fallback if empty and fallback exists
            if hist.empty and key in FALLBACKS:
                symbol = FALLBACKS[key]
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="10d")
            
            # Clean data: Drop NaNs to handle holidays
            hist = hist['Close'].dropna()

            if not hist.empty and len(hist) >= 2:
                current = hist.iloc[-1]
                prev = hist.iloc[-2]
                
                # Metric Calculation
                if key == 'US10Y':
                    # ^TNX is 10x Yield. Ex: 42.50 -> 42.60. Diff 0.10.
                    # We want Basis Points (1 bp).
                    # 4.25% -> 4.26% is 1 bp.
                    # TNX Diff 0.10 * 10 = 1.0 (Basis Points)
                    change = (current - prev) * 10 
                else:
                    # Standard Percent Change
                    change = ((current - prev) / prev) * 100
                
                data_map[key] = {'price': current, 'change': change, 'symbol': symbol}
            else:
                data_map[key] = {'price': 0.0, 'change': 0.0, 'symbol': symbol}
                failed_tickers.append(key)
        except Exception as e:
            data_map[key] = {'price': 0.0, 'change': 0.0, 'symbol': symbol}
            failed_tickers.append(key)
            
    return data_map

# --- 2. INSTITUTIONAL REGIME ENGINE ---
def analyze_market(data):
    if not data: return None
    
    # Safe getters
    def get_c(k): return data.get(k, {}).get('change', 0)
    
    # Core Data
    hyg_chg = get_c('HYG')
    vix_val = data.get('VIX', {}).get('price', 0)
    vix_chg = get_c('VIX')
    
    oil_chg = get_c('OIL')
    cop_chg = get_c('COPPER')
    banks_chg = get_c('BANKS')
    
    us10y_chg = get_c('US10Y') # In Basis Points now
    dxy_chg = get_c('DXY')
    btc_chg = get_c('BTC')
    
    # Defaults
    regime = "NEUTRAL"
    desc = "No dominant macro trend. Follow momentum."
    color_code = "#6b7280"
    longs, shorts, alerts = [], [], []

    # --- LOGIC HIERARCHY ---
    
    # 1. RISK OFF (The Veto)
    # Thresholds: HYG < -0.5% (2-Sigma) OR VIX > 5%
    if hyg_chg < -0.5 or vix_chg > 5.0:
        regime = "RISK OFF"
        desc = "Credit Stress or Volatility Spike. Cash is King."
        color_code = "#ef4444" # Red
        longs = ["Cash (UUP)", "Vol (VIX)"]
        shorts = ["Tech", "Crypto", "Small Caps", "High Yield"]
        alerts.append("⛔ CREDIT VETO: HYG is breaking down. Stop all long risk.")

    # 2. REFLATION (Growth + Yields)
    # Requires: Commodities UP + Yields UP + Banks Participating
    elif (oil_chg > 2.0 or cop_chg > 2.0) and us10y_chg > 5.0 and banks_chg > 0:
        regime = "REFLATION"
        desc = "Inflationary Growth. Real Assets outperform."
        color_code = "#f59e0b" # Orange
        longs = ["Energy (XLE)", "Banks (XLF)", "Industrials"]
        shorts = ["Bonds (TLT)", "Tech (Rate Sensitive)"]
        alerts.append("🔥 INFLATION PULSE: Rotate to Cyclicals.")

    # 3. LIQUIDITY PUMP (Risk On)
    # Requires: Dollar Down + BTC Up
    elif dxy_chg < -0.2 and btc_chg > 2.0:
        regime = "LIQUIDITY PUMP"
        desc = "Dollar weakness fueling high-beta assets."
        color_code = "#a855f7" # Purple
        longs = ["Bitcoin", "Nasdaq (QQQ)", "Semis (SMH)"]
        shorts = ["Dollar (DXY)", "Defensives"]
        alerts.append("🌊 LIQUIDITY ON: Green light for Beta.")

    # 4. GOLDILOCKS (Stability)
    # Requires: VIX Down + Yields Stable (< 5bps move) + Credit Stable
    elif vix_chg < 0 and abs(us10y_chg) < 5.0 and hyg_chg > -0.1:
        regime = "GOLDILOCKS"
        desc = "Low vol, stable rates. Favorable for equities."
        color_code = "#22c55e" # Green
        longs = ["S&P 500", "Tech", "Quality Growth"]
        shorts = ["Volatility"]
        alerts.append("✅ STABLE: Buy Dips.")

    # 5. MOMENTUM FALLBACK
    if not longs:
        # Find what's moving
        tradable = ['SPY','QQQ','IWM','BTC','GOLD','OIL','COPPER','BANKS','ENERGY','SEMIS']
        sorted_assets = sorted([(k, get_c(k)) for k in tradable], key=lambda x: x[1], reverse=True)
        
        longs = [f"{k} (Mom)" for k, v in sorted_assets[:2] if v > 0.5]
        shorts = [f"{k} (Mom)" for k, v in sorted_assets[-2:] if v < -0.5]
        
        if not longs: longs = ["Cash / Wait"]
        if not shorts: shorts = ["None"]

    return {
        'regime': regime, 'desc': desc, 'color': color_code,
        'longs': longs, 'shorts': shorts, 'alerts': alerts
    }

# --- 3. UI COMPONENTS ---
def create_nexus_graph(market_data):
    # Solar System Layout
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
        # Inverse for Risk Drivers
        if key in ['US10Y', 'DXY', 'VIX']: col = '#ef4444' if chg > 0 else '#22c55e'

        node_color.append(col)
        node_size.append(45 if key in ['US10Y', 'DXY', 'HYG'] else 35)
        
        ticker = market_data.get(key, {}).get('symbol', key)
        price = market_data.get(key, {}).get('price', 0)
        
        fmt_chg = f"{chg:+.2f}%"
        if key == 'US10Y': fmt_chg = f"{chg:+.1f} bps"
        
        node_text.append(f"<b>{info['label']} ({ticker})</b><br>Price: {price:.2f}<br>Chg: {fmt_chg}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#4b5563'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[n.split('<br>')[0] for n in node_text], textposition="bottom center",
                             hovertext=node_text, hoverinfo="text",
                             marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')),
                             textfont=dict(size=11, color='white')))
    
    fig.update_layout(
        showlegend=False, margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.0, 2.0]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500
    )
    return fig

def create_heatmap_matrix(market_data):
    # Correlation Heatmap Logic
    z_data = [
        [ 0.9,  0.9,  0.4,  0.6,  0.1,  0.2, 0.7], 
        [-0.8, -0.6, -0.9, -0.3,  0.4,  0.6, -0.9], 
        [-0.4, -0.5, -0.9, -0.9, -0.6, -0.1, -0.2], 
        [ 0.8,  0.7,  0.1,  0.8,  0.6,  0.9, 0.8], 
        [ 0.2,  0.3,  0.5,  0.9,  0.9,  0.8, 0.3]  
    ]
    
    x_labels = ['Tech', 'Crypto', 'Gold', 'EM', 'Energy', 'Banks', 'Housing']
    y_labels = ['Liquidity', 'Real Yields', 'Dollar', 'Credit', 'Growth']

    fig = px.imshow(
        z_data, x=x_labels, y=y_labels,
        color_continuous_scale=['#ef4444', '#1e2127', '#22c55e'],
        range_color=[-1, 1], aspect="auto"
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=450)
    return fig

# --- 4. MAIN LAYOUT ---
def main():
    with st.spinner("Initializing MacroNexus Pro..."):
        market_data = fetch_live_data()
        analysis = analyze_market(market_data)

    # Metrics
    st.markdown("### 📡 Market Pulse")
    cols = st.columns(6)
    
    def tile(c, label, key):
        d = market_data.get(key, {})
        val = d.get('price', 0)
        chg = d.get('change', 0)
        sym = d.get('symbol', key)
        color = "#ef4444" if chg < 0 else "#22c55e"
        
        # Logic: Veto/Risk assets Inverse Color
        if key in ['VIX', 'US10Y', 'DXY']: color = "#ef4444" if chg > 0 else "#22c55e"
        
        fmt_chg = f"{chg:+.2f}%"
        if key == 'US10Y': fmt_chg = f"{chg:+.1f} bps"

        c.markdown(f"""
        <div class="metric-container" style="border-left-color: {color};">
            <div class="metric-header"><span class="metric-label">{label}</span><span class="metric-ticker">{sym}</span></div>
            <div><span class="metric-val">{val:.2f}</span><span class="metric-chg" style="color: {color};">{fmt_chg}</span></div>
        </div>
        """, unsafe_allow_html=True)

    tile(cols[0], "Credit", "HYG")
    tile(cols[1], "Volatility", "VIX")
    tile(cols[2], "10Y Yield", "US10Y")
    tile(cols[3], "Dollar", "DXY")
    tile(cols[4], "Oil", "OIL")
    tile(cols[5], "Bitcoin", "BTC")

    # Tabs
    t1, t2, t3, t4 = st.tabs(["🚀 Dashboard", "📊 Heatmap", "🌊 Liquidity", "📖 Playbook"])

    with t1:
        c_g, c_a = st.columns([2.5, 1])
        with c_g: st.plotly_chart(create_nexus_graph(market_data), use_container_width=True)
        with c_a:
            bg = analysis['color']
            st.markdown(f"""
            <div class="regime-badge" style="background-color: {bg}22; border-color: {bg};">
                <div style="color: {bg}; font-weight: bold; font-size: 20px; margin-bottom: 5px;">{analysis['regime']}</div>
                <div style="font-size: 11px; color: #ccc;">{analysis['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.success(f"**LONG:** {', '.join(analysis['longs'])}")
            st.error(f"**AVOID:** {', '.join(analysis['shorts'])}")
            if analysis['alerts']: st.error(analysis['alerts'][0], icon="🚨")

    with t2:
        st.plotly_chart(create_heatmap_matrix(market_data), use_container_width=True)

    with t3:
        st.info("Visualizes how Fed Policy flows downstream to specific sectors.")
        try:
            g = graphviz.Digraph()
            g.attr(rankdir='TB', bgcolor='transparent')
            g.attr('node', shape='box', style='filled, rounded', fontname='Helvetica', fontcolor='white', penwidth='0')
            g.attr('edge', color='#6b7280')
            
            g.node('FED', 'FED & TREASURY', fillcolor='#4f46e5')
            g.node('RATE', 'YIELDS & RATES', fillcolor='#b91c1c')
            g.node('USD', 'DOLLAR (DXY)', fillcolor='#1e3a8a')
            g.node('CRED', 'CREDIT (HYG)', fillcolor='#7e22ce')
            
            g.node('GROWTH', 'TECH / CRYPTO', fillcolor='#1f2937')
            g.node('REAL', 'COMMODITIES', fillcolor='#1f2937')
            g.node('EM', 'EMERGING MKTS', fillcolor='#1f2937')
            
            g.edge('FED','RATE'); g.edge('FED','USD'); g.edge('FED','CRED')
            g.edge('RATE','GROWTH'); g.edge('RATE','REAL'); g.edge('USD','EM')
            g.edge('CRED','GROWTH')
            
            st.graphviz_chart(g, use_container_width=True)
        except:
            st.warning("Graphviz missing.")

    with t4:
        st.markdown("""
        ### 📖 Daily Workflow
        1. **Check Plumbing (HYG, VIX):** If Red, Market is Unsafe.
        2. **Check Regime:** Follow the Badge (Risk Off vs Liquidity).
        3. **Confirm:** Check `US10Y` trends before buying Tech/Gold.
        """)

if __name__ == "__main__":
    main()
