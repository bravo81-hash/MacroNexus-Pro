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
    
    /* Rich Metric Card */
    .metric-container {
        background-color: #1e2127;
        padding: 10px;
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

# --- 1. FULL DATA UNIVERSE (Mapped to Liquid ETFs) ---
# Expanded list to ensure no data is missing from your requirements
TICKERS = {
    # DRIVERS (The Plumbing)
    'US10Y': '^TNX',       # 10Y Yield
    'DXY': 'UUP',          # Dollar ETF 
    'VIX': 'VIXY',         # Volatility ETF
    'HYG': 'HYG',          # Credit High Yield
    'TLT': 'TLT',          # 20Y Bonds
    'SHY': 'SHY',          # 1-3Y Treasury
    
    # COMMODITIES
    'GOLD': 'GLD', 'SILVER': 'SLV', 'OIL': 'USO',
    'NATGAS': 'UNG', 'COPPER': 'CPER', 'AG': 'DBA',
    
    # INDICES
    'SPY': 'SPY', 'QQQ': 'QQQ', 'IWM': 'IWM',
    'EEM': 'EEM', 'FXI': 'FXI', 'EWJ': 'EWJ',
    
    # SECTORS
    'TECH': 'XLK', 'SEMIS': 'SMH', 'BANKS': 'XLF',
    'ENERGY': 'XLE', 'HOME': 'XHB', 'UTIL': 'XLU',
    
    # CRYPTO & FOREX
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD',
    'EURO': 'FXE', 'YEN': 'FXY'
}

@st.cache_data(ttl=300)
def fetch_live_data():
    """Fetches data individually with error handling."""
    data_map = {}
    for key, symbol in TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            # Fetch 10 days to handle weekends/holidays
            hist = ticker.history(period="10d")
            
            # Data Cleaning
            if not hist.empty and len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                
                # Special logic for Yields (Basis points)
                if key == 'US10Y':
                    change = (current - prev) * 10 # Convert to Basis Points
                else:
                    change = ((current - prev) / prev) * 100
                    
                data_map[key] = {'price': current, 'change': change, 'symbol': symbol}
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
    desc = "No clear macro dominance. Follow momentum."
    color_code = "#6b7280" 
    longs, shorts, alerts = [], [], []

    # 1. MACRO REGIME CHECKS
    if hyg < -0.3 or vix > 3.0:
        regime = "RISK OFF"
        desc = "Credit widening or Vol spiking. Cash is King."
        color_code = "#ef4444" # Red
        longs = ["Cash (UUP)", "Vol (VIXY)", "Bonds (TLT)", "Yen (FXY)"]
        shorts = ["Tech (QQQ)", "Crypto", "Small Caps (IWM)", "EM (EEM)", "High Yield (HYG)", "Banks"]
        alerts.append("⛔ CREDIT STRESS: Veto Longs. Reduce Exposure.")

    elif (oil > 0.5 or cop > 0.5) and us10y > 0.5: # Lowered yield threshold for sensitivity
        regime = "REFLATION"
        desc = "Growth + Inflation rising. Real assets outperform."
        color_code = "#f59e0b" # Orange
        longs = ["Energy (XLE)", "Banks (XLF)", "Industrials", "Commodities (Ag/Metals)"]
        shorts = ["Bonds (TLT)", "Tech (Rate Sensitive)", "Homebuilders (XHB)", "Utilities"]
        alerts.append("🔥 INFLATION PULSE: Rotate to Real Assets.")

    elif dxy < -0.1 and btc > 1.0:
        regime = "LIQUIDITY PUMP"
        desc = "Dollar weakness fueling high-beta assets."
        color_code = "#a855f7" # Purple
        longs = ["Bitcoin", "Ethereum", "Nasdaq (QQQ)", "Semis (SMH)", "Gold"]
        shorts = ["Dollar (UUP)", "Cash", "Defensives"]
        alerts.append("🌊 LIQUIDITY ON: Green light for High Beta.")

    elif vix < 0 and abs(us10y) < 2.0:
        regime = "GOLDILOCKS"
        desc = "Low vol, stable rates. Favorable for equities."
        color_code = "#22c55e" # Green
        longs = ["S&P 500", "Tech (XLK)", "Semis (SMH)", "Housing (XHB)", "Small Caps"]
        shorts = ["Volatility (VIX)"]
        alerts.append("✅ STABLE: Buy Dips.")

    # 2. FALLBACK MOMENTUM (Smart Neutral Logic)
    if not longs:
        # Filter tradable assets (exclude drivers like VIX/Yields)
        asset_keys = ['SPY', 'QQQ', 'IWM', 'BTC', 'ETH', 'GOLD', 'SILVER', 'OIL', 'COPPER', 'SEMIS', 'BANKS', 'ENERGY', 'HOME']
        assets = {k: get_c(k) for k in asset_keys}
        
        # Sort by performance
        sorted_assets = sorted(assets.items(), key=lambda x: x[1], reverse=True)
        
        # Pick absolute winners and losers
        top_pick = sorted_assets[0]
        bottom_pick = sorted_assets[-1]
        
        if top_pick[1] > 0.3:
            longs = [f"{top_pick[0]} (+{top_pick[1]:.1f}%)", f"{sorted_assets[1][0]} (+{sorted_assets[1][1]:.1f}%)"]
        else:
            longs = ["Cash / Wait"]
            
        if bottom_pick[1] < -0.3:
            shorts = [f"{bottom_pick[0]} ({bottom_pick[1]:.1f}%)", f"{sorted_assets[-2][0]} ({sorted_assets[-2][1]:.1f}%)"]
        else:
            shorts = ["None"]

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
        'XHB':   {'pos': (-0.8, -0.4), 'label': 'Housing'},
        'XLF':   {'pos': (1.5, -1.0), 'label': 'Banks'},
        'VIX':   {'pos': (0, 1.5), 'label': 'Vol'}
    }
    
    edges = [
        ('US10Y', 'QQQ'), ('US10Y', 'GOLD'), ('US10Y', 'XHB'),
        ('DXY', 'GOLD'), ('DXY', 'OIL'), ('DXY', 'EEM'),
        ('HYG', 'SPY'), ('HYG', 'IWM'), ('HYG', 'XLF'),
        ('QQQ', 'BTC'), ('QQQ', 'SMH'),
        ('COPPER', 'US10Y'), ('OIL', 'XLE'), ('VIX', 'SPY')
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
    
    # Expanded Range to prevent cutoffs
    fig.update_layout(
        showlegend=False, margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.0, 3.0]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=550
    )
    return fig

def create_heatmap_matrix(market_data):
    # Correlation Logic
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
            
            st.success("**LONG**")
            for item in analysis['longs']: st.markdown(f"<small>{item}</small>", unsafe_allow_html=True)
            
            st.error("**AVOID**")
            for item in analysis['shorts']: st.markdown(f"<small>{item}</small>", unsafe_allow_html=True)
            
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
            st.warning("Graphviz missing. Please install it on the server.")

    with t4:
        st.markdown("""
        ### 📖 Daily Workflow
        1. **Check Plumbing (HYG, VIX):** If Red, Market is Unsafe.
        2. **Check Regime:** Follow the Badge (Risk Off vs Liquidity).
        3. **Confirm:** Check `US10Y` trends before buying Tech/Gold.
        """)

if __name__ == "__main__":
    main()
