import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import graphviz
import numpy as np

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="MacroNexus Pro Terminal",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. PROFESSIONAL STYLING (CSS) ---
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp { background-color: #0B0E11; color: #E6E6E6; font-family: 'Inter', sans-serif; }
    
    /* Metrics Cards */
    .metric-card {
        background: rgba(30, 34, 45, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .metric-card:hover { border-color: rgba(255, 255, 255, 0.2); transform: translateY(-2px); }
    .metric-label { font-size: 11px; color: #8B9BB4; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }
    .metric-value { font-size: 20px; font-weight: 700; color: #FFFFFF; margin-top: 4px; }
    .metric-delta { font-size: 12px; font-weight: 600; margin-left: 8px; }
    
    /* Strategy Cards */
    .strat-card {
        background: linear-gradient(145deg, rgba(30,34,45,1) 0%, rgba(20,22,28,1) 100%);
        border: 1px solid #2A2E39;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        height: 100%;
    }
    .strat-title { font-size: 18px; font-weight: 700; color: #FFFFFF; margin-bottom: 8px; border-bottom: 1px solid #2A2E39; padding-bottom: 8px; }
    .strat-subtitle { font-size: 11px; color: #5F6B7C; text-transform: uppercase; letter-spacing: 1px; margin-top: 12px; margin-bottom: 4px; }
    .strat-data { font-size: 14px; color: #D1D5DB; font-family: 'Roboto Mono', monospace; }
    
    /* Control Panel */
    .control-bar {
        background-color: #161920;
        border: 1px solid #2A2E39;
        border-radius: 10px;
        padding: 15px 25px;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* Regime Badge */
    .regime-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 24px;
        text-align: right;
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #2A2E39; }
    .stTabs [data-baseweb="tab"] { height: 45px; border-radius: 6px 6px 0 0; border: none; color: #8B9BB4; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #1E222D; color: #FFF; }
    
    /* Utilities */
    .badge-blue { background: rgba(59, 130, 246, 0.2); color: #3B82F6; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid rgba(59, 130, 246, 0.3); }
    
    /* Helper Text */
    .context-box {
        background: rgba(255,255,255,0.05);
        border-left: 3px solid #3B82F6;
        padding: 10px;
        font-size: 12px;
        color: #B0B8C3;
        margin-top: 10px;
        border-radius: 0 4px 4px 0;
    }
    .context-header {
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
        font-size: 11px;
        text-transform: uppercase;
    }
    
</style>
""", unsafe_allow_html=True)

# --- 3. DATA UNIVERSE ---
TICKERS = {
    # Drivers
    'US10Y': '^TNX', 'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'HYG': 'HYG', 'TLT': 'TLT', 'TIP': 'TIP',
    # Commodities
    'GOLD': 'GLD', 'OIL': 'USO', 'COPPER': 'CPER', 'NATGAS': 'UNG',
    # Indices
    'SPY': 'SPY', 'QQQ': 'QQQ', 'IWM': 'IWM', 'RUT': '^RUT',
    # Sectors
    'TECH': 'XLK', 'SEMIS': 'SMH', 'BANKS': 'XLF', 'ENERGY': 'XLE', 
    'HOME': 'XHB', 'UTIL': 'XLU', 'STAPLES': 'XLP', 'DISC': 'XLY', 
    'IND': 'XLI', 'HEALTH': 'XLV', 'MAT': 'XLB', 'COMM': 'XLC', 'RE': 'XLRE',
    # Crypto
    'BTC': 'BTC-USD'
}
FALLBACKS = {'DXY': 'UUP', 'VIX': 'VIXY', 'RUT': 'IWM'}

# --- 4. DATA ENGINE ---
@st.cache_data(ttl=300)
def fetch_market_data():
    data = {}
    history_data = {} 
    
    for key, symbol in TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            if hist.empty and key in FALLBACKS:
                symbol = FALLBACKS[key]
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="6mo")
            
            hist_clean = hist['Close'].dropna()
            
            if not hist_clean.empty and len(hist_clean) >= 22:
                history_data[key] = hist_clean
                
                curr = hist_clean.iloc[-1]
                prev = hist_clean.iloc[-2]
                prev_w = hist_clean.iloc[-6] 
                prev_m = hist_clean.iloc[-21] 
                
                if key == 'US10Y':
                    chg_d = (curr - prev) * 10 
                    chg_w = (curr - prev_w) * 10
                    chg_m = (curr - prev_m) * 10
                    disp_fmt = "bps"
                else:
                    chg_d = ((curr - prev) / prev) * 100
                    chg_w = ((curr - prev_w) / prev_w) * 100
                    chg_m = ((curr - prev_m) / prev_m) * 100
                    disp_fmt = "%"
                
                data[key] = {
                    'price': curr, 'change': chg_d, 
                    'change_w': chg_w, 'change_m': chg_m,
                    'symbol': symbol, 'fmt': disp_fmt, 'valid': True
                }
            else:
                data[key] = {'price': 0, 'change': 0, 'change_w': 0, 'change_m': 0, 'symbol': symbol, 'fmt': "%", 'valid': False}
        except:
            data[key] = {'price': 0, 'change': 0, 'change_w': 0, 'change_m': 0, 'symbol': symbol, 'fmt': "%", 'valid': False}
            
    df_history = pd.DataFrame(history_data)
    return data, df_history

# --- 5. LOGIC ENGINE ---
def determine_regime(data):
    def g(k): return data.get(k, {}).get('change', 0)
    
    hyg, vix = g('HYG'), g('VIX')
    oil, cop = g('OIL'), g('COPPER')
    us10y, dxy = g('US10Y'), g('DXY')
    btc, banks = g('BTC'), g('BANKS')
    
    if hyg < -0.5 or vix > 5.0: return "RISK OFF"
    if (oil > 1.5 or cop > 1.5) and us10y > 3.0 and banks > 0: return "REFLATION"
    if dxy < -0.3 and btc > 1.5: return "LIQUIDITY"
    if vix < 0 and abs(us10y) < 5.0 and hyg > -0.1: return "GOLDILOCKS"
    return "NEUTRAL"

# --- 6. STRATEGY DATABASE ---
STRATEGIES = {
    "GOLDILOCKS": {
        "desc": "Low Vol + Steady Trend. Market climbing wall of worry.",
        "risk": "1.5%", "bias": "Long",
        "index": {
            "strat": "Directional Diagonal", 
            "dte": "Front 17 / Back 31", 
            "setup": "Buy 70D Call (Back) / Sell 30D Call (Front)", 
            "notes": "Stock replacement strategy. Combines delta (trend) and theta (decay). Use on SPY/QQQ."
        },
        "longs": "TECH, SEMIS, DISC", "shorts": "VIX, TLT"
    },
    "LIQUIDITY": {
        "desc": "High Liquidity / Dollar Weakness. Drift Up environment.",
        "risk": "1.0%", "bias": "Aggressive Long",
        "index": {
            "strat": "Flyagonal (Drift)", 
            "dte": "Entry 7-14 DTE", 
            "setup": "Call BWB (Upside) + Put Diagonal (Downside)", 
            "notes": "Captures the drift. Upside tent (Call BWB) funds the downside floor (Put Diagonal). Close at 4% profit."
        },
        "longs": "BTC, SEMIS, QQQ", "shorts": "DXY, CASH"
    },
    "REFLATION": {
        "desc": "Inflation / Rates Rising. Real Assets outperform Tech.",
        "risk": "1.0%", "bias": "Cyclical Long",
        "index": {
            "strat": "Call Spread (Cyclicals)", 
            "dte": "45 DTE", 
            "setup": "Buy 60D / Sell 30D", 
            "notes": "Focus on Russell 2000 (IWM), Energy (XLE), and Banks (XLF). Avoid long duration Tech."
        },
        "longs": "ENERGY, BANKS, IND", "shorts": "TLT, TECH"
    },
    "NEUTRAL": {
        "desc": "Chop / Range Bound. No clear direction.",
        "risk": "Income Size", "bias": "Neutral/Theta",
        "index": {
            "strat": "TimeEdge / TimeZone", 
            "dte": "Entry 15 / Exit 7", 
            "setup": "Put Calendar Spread (ATM)", 
            "notes": "Pure Theta play. Sell 15 DTE / Buy 22+ DTE. Requires VIX < 20 and no binary events."
        },
        "longs": "INCOME, CASH", "shorts": "MOMENTUM"
    },
    "RISK OFF": {
        "desc": "High Volatility / Credit Stress. Preservation mode.",
        "risk": "0.5%", "bias": "Short/Hedge",
        "index": {
            "strat": "A14 Put BWB", 
            "dte": "Entry 14 / Exit 7", 
            "setup": "Long ATM / Short -40 / (Skip) / Long -60", 
            "notes": "Crash Catcher. Zero upside risk if filled for credit. Profit tent expands into the crash."
        },
        "longs": "VIX, DXY", "shorts": "SPY, IWM, HYG"
    }
}

# --- 7. DYNAMIC HELPER ---
def get_val(data, key, timeframe):
    """Retrieve value based on timeframe selection"""
    d = data.get(key, {})
    if timeframe == 'Tactical (Daily)':
        return d.get('change', 0)
    else: # Structural (Weekly)
        return d.get('change_w', 0)

# --- 8. VISUALIZATION FUNCTIONS ---

def plot_nexus_graph(data, timeframe):
    # Dynamic Colors based on selected timeframe
    dot = graphviz.Digraph(comment='The Macro Machine')
    dot.attr(rankdir='LR', bgcolor='#0e1117')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontcolor='white')
    dot.attr('edge', color='#555555', arrowsize='0.8')

    def get_col(k, invert=False):
        c = get_val(data, k, timeframe)
        if invert: return '#ef4444' if c > 0 else '#22c55e'
        return '#22c55e' if c > 0 else '#ef4444'

    # Get values based on timeframe
    us10y_c = get_val(data, "US10Y", timeframe)
    dxy_c = get_val(data, "DXY", timeframe)
    hyg_c = get_val(data, "HYG", timeframe)
    
    # Label suffix
    lbl = "1d" if timeframe == 'Tactical (Daily)' else "1w"

    # NODES
    dot.node('FED', 'FED POLICY\n(Liquidity)', fillcolor='#3b82f6')
    
    dot.node('US10Y', f'YIELDS ({lbl})\n{us10y_c:+.1f} bps', fillcolor=get_col('US10Y', True))
    dot.node('DXY', f'DOLLAR ({lbl})\n{dxy_c:+.2f}%', fillcolor=get_col('DXY', True))
    dot.node('HYG', f'CREDIT ({lbl})\n{hyg_c:+.2f}%', fillcolor=get_col('HYG', False))
    
    dot.node('TECH', 'TECH (QQQ)', fillcolor=get_col('QQQ', False))
    dot.node('GOLD', 'GOLD', fillcolor=get_col('GOLD', False))
    dot.node('CRYPTO', 'CRYPTO', fillcolor=get_col('BTC', False))
    dot.node('EM', 'EMERGING', fillcolor=get_col('EEM', False))
    dot.node('CYCL', 'CYCLICALS', fillcolor=get_col('ENERGY', False))

    # EDGES
    dot.edge('FED', 'US10Y', 'Rates')
    dot.edge('FED', 'DXY', 'Currency')
    dot.edge('US10Y', 'TECH', 'Cost of Capital', color='#ef4444')
    dot.edge('US10Y', 'GOLD', 'Real Rates', color='#ef4444')
    dot.edge('DXY', 'EM', 'Debt Pressure', color='#ef4444')
    dot.edge('DXY', 'GOLD', 'Pricing', color='#ef4444')
    dot.edge('US10Y', 'HYG', 'Refinancing')
    dot.edge('HYG', 'TECH', 'Risk On')
    dot.edge('HYG', 'CRYPTO', 'Liquidity Spillover')
    
    return dot

def plot_rrg(data, category, view):
    items = []
    
    if category == 'SECTORS':
        keys = ['TECH','SEMIS','BANKS','ENERGY','HOME','UTIL','STAPLES','DISC','IND','HEALTH','MAT','COMM','RE']
    else: # ASSETS
        keys = ['SPY','QQQ','IWM','GOLD','BTC','TLT','DXY','HYG','OIL']
        
    for k in keys:
        d = data.get(k, {})
        # Tactical: X=Weekly, Y=Daily | Structural: X=Monthly, Y=Weekly
        if view == 'Tactical (Daily)':
            x = d.get('change_w', 0)
            y = d.get('change', 0)
            x_lab, y_lab = "Weekly Trend", "Daily Momentum"
        else:
            x = d.get('change_m', 0)
            y = d.get('change_w', 0)
            x_lab, y_lab = "Monthly Trend", "Weekly Momentum"
            
        c = 'gray'
        if x>0 and y>0: c='#22c55e'
        elif x<0 and y>0: c='#3b82f6'
        elif x>0 and y<0: c='#f59e0b'
        elif x<0 and y<0: c='#ef4444'
        
        items.append({'Symbol': k, 'Trend': x, 'Momentum': y, 'Color': c})
        
    df = pd.DataFrame(items)
    
    fig = px.scatter(df, x='Trend', y='Momentum', text='Symbol', color='Color', color_discrete_map="identity")
    fig.update_traces(textposition='top center', marker=dict(size=14, line=dict(width=1, color='white')))
    
    fig.add_hline(y=0, line_dash="dot", line_color="#555")
    fig.add_vline(x=0, line_dash="dot", line_color="#555")
    
    limit = max(df['Trend'].abs().max(), df['Momentum'].abs().max()) * 1.1 if not df.empty else 1
    fig.update_layout(
        xaxis=dict(range=[-limit, limit], zeroline=False, showgrid=True, gridcolor='#333', title=x_lab),
        yaxis=dict(range=[-limit, limit], zeroline=False, showgrid=True, gridcolor='#333', title=y_lab),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'), showlegend=False, height=450, margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

def plot_sankey_sectors(data, timeframe):
    # DYNAMIC: Uses timeframe for calculation
    sector_keys = ['TECH','SEMIS','BANKS','ENERGY','HOME','UTIL','HEALTH','MAT','COMM']
    sectors = {k: get_val(data, k, timeframe) for k in sector_keys}
    
    df = pd.DataFrame(list(sectors.items()), columns=['id', 'val']).sort_values('val', ascending=False)
    
    winners = df.head(3)
    losers = df.tail(3)
    
    labels = list(losers['id']) + list(winners['id'])
    sources, targets, values, colors = [], [], [], []
    
    for i in range(len(losers)):
        for j in range(len(winners)):
            sources.append(i) 
            targets.append(len(losers) + j)
            # Size = Magnitude of divergence
            flow_size = abs(losers.iloc[i]['val']) + abs(winners.iloc[j]['val'])
            values.append(flow_size)
            colors.append('rgba(59, 130, 246, 0.2)') 
            
    node_colors = ['#ef4444']*3 + ['#22c55e']*3 
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=node_colors),
        link=dict(source=sources, target=targets, value=values, color=colors)
    )])
    
    title = f"Sector Rotation Flow ({'Daily' if timeframe=='Tactical (Daily)' else 'Weekly'})"
    fig.update_layout(title_text=title, font=dict(color='white'), paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plot_sankey_assets(data, timeframe):
    # DYNAMIC: Uses timeframe for calculation
    asset_keys = ['SPY','TLT','DXY','GOLD','BTC','OIL','HYG']
    assets = {k: get_val(data, k, timeframe) for k in asset_keys}
    
    df = pd.DataFrame(list(assets.items()), columns=['id', 'val']).sort_values('val', ascending=False)
    
    winners = df.head(3)
    losers = df.tail(3)
    
    labels = list(losers['id']) + list(winners['id'])
    sources, targets, values, colors = [], [], [], []
    
    for i in range(len(losers)):
        for j in range(len(winners)):
            sources.append(i) 
            targets.append(len(losers) + j)
            flow_size = abs(losers.iloc[i]['val']) + abs(winners.iloc[j]['val'])
            values.append(flow_size)
            colors.append('rgba(168, 85, 247, 0.2)') 
            
    node_colors = ['#ef4444']*3 + ['#22c55e']*3 
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=node_colors),
        link=dict(source=sources, target=targets, value=values, color=colors)
    )])
    
    title = f"Macro Asset Rotation ({'Daily' if timeframe=='Tactical (Daily)' else 'Weekly'})"
    fig.update_layout(title_text=title, font=dict(color='white'), paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plot_correlation_heatmap(history_df):
    if history_df.empty: return go.Figure()
    
    corr = history_df.pct_change().corr()
    subset = ['US10Y', 'DXY', 'VIX', 'HYG', 'SPY', 'QQQ', 'IWM', 'BTC', 'GOLD', 'OIL']
    cols = [c for c in subset if c in corr.columns]
    corr_subset = corr.loc[cols, cols]
    
    fig = px.imshow(
        corr_subset, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ccc'), height=400)
    return fig

# --- 9. MAIN APPLICATION ---
def main():
    # --- LOAD DATA ---
    with st.spinner("Connecting to MacroNexus Core..."):
        market_data, history_df = fetch_market_data()
        
    # --- TOP METRICS BAR ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    def metric_tile(col, label, key, invert=False):
        d = market_data.get(key, {})
        val = d.get('price', 0); chg = d.get('change', 0); fmt = d.get('fmt', "%")
        is_up = chg > 0
        if invert: color = "#F43F5E" if is_up else "#10B981" 
        else: color = "#10B981" if is_up else "#F43F5E" 
        fmt_chg = f"{chg:+.1f} bps" if key == 'US10Y' else f"{chg:+.2f}%"
        
        # FIXED: Removed all indentation in HTML string to prevent code block rendering
        col.markdown(f"""
<div class="metric-card" style="border-left: 3px solid {color};">
<div class="metric-label">{label}</div>
<div class="metric-value">{val:.2f}
<span class="metric-delta" style="color: {color};">{fmt_chg}</span>
</div>
</div>
""", unsafe_allow_html=True)

    metric_tile(c1, "Credit (HYG)", "HYG")
    metric_tile(c2, "Volatility (VIX)", "VIX", invert=True)
    metric_tile(c3, "10Y Yield", "US10Y", invert=True)
    metric_tile(c4, "Dollar (DXY)", "DXY", invert=True)
    metric_tile(c5, "Oil", "OIL")
    metric_tile(c6, "Bitcoin", "BTC")

    # --- MAIN CONTROLS ---
    auto_regime = determine_regime(market_data)

    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    
    with ctrl1:
        override = st.checkbox("Manual Override", value=False)
        if override:
            active_regime = st.selectbox("Force Regime", list(STRATEGIES.keys()), label_visibility="collapsed")
        else:
            active_regime = auto_regime
            
    with ctrl2:
        # TIMEFRAME TOGGLE
        timeframe = st.selectbox("Analytic View", ["Tactical (Daily)", "Structural (Weekly)"], label_visibility="collapsed")
        
    with ctrl3:
        r_colors = {"GOLDILOCKS": "#10B981", "LIQUIDITY": "#A855F7", "REFLATION": "#F59E0B", "NEUTRAL": "#6B7280", "RISK OFF": "#EF4444"}
        rc = r_colors.get(active_regime, "#6B7280")
        # FIXED: HTML Formatting
        st.markdown(f"""
<div style="text-align: right; display: flex; align-items: center; justify-content: flex-end; gap: 15px;">
<div style="text-align: right;">
<div style="font-size: 10px; color: #8B9BB4; letter-spacing: 1px;">SYSTEM STATUS</div>
<div style="font-size: 14px; font-weight: bold; color: {rc};">ACTIVE</div>
</div>
<div class="regime-badge" style="background: {rc}22; color: {rc}; border: 1px solid {rc};">
{active_regime}
</div>
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- TABS ---
    tab_mission, tab_pulse, tab_macro, tab_playbook = st.tabs([
        "🚀 MISSION CONTROL", 
        "📊 MARKET PULSE", 
        "🕸️ MACRO MACHINE", 
        "📖 STRATEGY PLAYBOOK"
    ])

    # === TAB 1: MISSION CONTROL ===
    with tab_mission:
        with st.expander("🎛️ SPX Income Reactor Telemetry (Manual Input)", expanded=True):
            tc1, tc2, tc3 = st.columns(3)
            iv_rank = tc1.slider("IV Rank (Percentile)", 0, 100, 45, help="Low < 30 | High > 50")
            skew_rank = tc2.slider("Skew Rank", 0, 100, 50, help="High Skew (>80) indicates Crash Risk")
            adx_val = tc3.slider("Trend ADX", 0, 60, 20, help="< 20 is Chop/Range. > 30 is Strong Trend")

        st.divider()

        strat_data = STRATEGIES[active_regime]
        
        # REACTOR LOGIC
        reactor_output = {}
        if active_regime == "RISK OFF":
            reactor_output = STRATEGIES["RISK OFF"]["index"]
        elif iv_rank > 50: 
            if skew_rank > 80:
                reactor_output = {"strat": "Put BWB (High Skew)", "dte": "21-30 DTE", "setup": "Long ATM / Short -40 / Skip / Long -60", "notes": "REACTOR WARNING: High Skew detected. Crash risk elevated. Use BWB to eliminate upside risk and profit from crash."}
            else:
                reactor_output = {"strat": "Iron Condor", "dte": "30-45 DTE", "setup": "Delta 15 Wings", "notes": "REACTOR: Volatility is high but Skew is normal. Classic reversion to mean play."}
        else: 
            if adx_val > 25:
                if active_regime == "LIQUIDITY":
                    reactor_output = STRATEGIES["LIQUIDITY"]["index"] 
                else:
                    reactor_output = {"strat": "Directional Diagonal", "dte": "Front 17 / Back 31", "setup": "Buy Back ITM / Sell Front OTM", "notes": "REACTOR: Trend detected (ADX > 25). Do not cap upside."}
            else:
                reactor_output = STRATEGIES["NEUTRAL"]["index"]

        col_L, col_R = st.columns([1, 2])
        
        with col_L:
            # FIXED: HTML Formatting
            st.markdown(f"""
<div class="strat-card">
<div class="strat-title" style="color: {rc}">CONTEXT</div>
<div class="strat-subtitle">DESCRIPTION</div>
<div class="strat-data">{strat_data['desc']}</div>
<div class="strat-subtitle">RISK SIZE</div>
<div class="strat-data" style="font-size: 24px; color: {rc}">{strat_data['risk']}</div>
<div class="strat-subtitle">BIAS</div>
<div class="strat-data">{strat_data['bias']}</div>
<div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #333;">
<div style="color: #10B981; font-size: 12px; margin-bottom: 4px;"><strong>TARGETS:</strong> {strat_data.get('longs', '')}</div>
<div style="color: #EF4444; font-size: 12px;"><strong>AVOID:</strong> {strat_data.get('shorts', '')}</div>
</div>
</div>
""", unsafe_allow_html=True)

        with col_R:
            st.markdown(f'<div class="strat-title">TACTICAL EXECUTION</div>', unsafe_allow_html=True)
            # FIXED: HTML Formatting
            st.markdown(f"""
<div class="strat-card" style="border-color: {rc}; background: linear-gradient(145deg, {rc}11 0%, rgba(20,22,28,1) 100%);">
<div class="strat-title" style="color: {rc}; border-color: {rc}44;">{reactor_output['strat']}</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 15px;">
<div>
<div class="strat-subtitle">TIMING (DTE)</div>
<div class="strat-data" style="font-weight: 700;">{reactor_output['dte']}</div>
</div>
<div>
<div class="strat-subtitle">STRUCTURE</div>
<div class="strat-data" style="font-weight: 700;">{reactor_output['setup']}</div>
</div>
</div>
<div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1);">
<div class="strat-subtitle" style="margin-top: 0;">THE LOGIC</div>
<div class="strat-data" style="font-size: 13px; color: #9CA3AF;">{reactor_output['notes']}</div>
</div>
<div style="margin-top: 15px; display: flex; gap: 10px;">
<span class="badge-blue">IV Rank: {iv_rank}</span>
<span class="badge-blue">Skew: {skew_rank}</span>
<span class="badge-blue">ADX: {adx_val}</span>
</div>
</div>
""", unsafe_allow_html=True)

    # === TAB 2: MARKET PULSE ===
    with tab_pulse:
        
        # ROW 1: SANKEY SECTORS
        st.subheader("🌊 Capital Flow: Sectors")
        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            st.plotly_chart(plot_sankey_sectors(market_data, timeframe), use_container_width=True)
        with col_s2:
            st.markdown(f"""
<div class="context-box">
<div class="context-header">💡 How to Read</div>
<div>Sankey diagrams visualize the flow of capital. The "Source" (Left) are the worst performing sectors. The lines show where that capital is theoretically rotating into (The "Winners" on the Right).</div>
<br>
<div class="context-header">🔎 Analyst Note</div>
<div>Thicker lines indicate stronger conviction in the rotation. If defensives (Util/Staples) are on the Right, it signals Fear.</div>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # ROW 2: SANKEY ASSETS
        st.subheader("🌍 Capital Flow: Macro Assets")
        col_a1, col_a2 = st.columns([3, 1])
        with col_a1:
            st.plotly_chart(plot_sankey_assets(market_data, timeframe), use_container_width=True)
        with col_a2:
            st.markdown(f"""
<div class="context-box">
<div class="context-header">💡 How to Read</div>
<div>Tracks rotation between Asset Classes. Are traders dumping Bonds (TLT) to buy Stocks (SPY)? Or dumping Stocks to buy Gold?</div>
<br>
<div class="context-header">🔎 Analyst Note</div>
<div>Risk-On flows usually show DXY/TLT on the Left and SPY/BTC on the Right.</div>
</div>
""", unsafe_allow_html=True)
            
        st.divider()
        
        # ROW 3: CORRELATION
        st.subheader("🔥 Inter-Correlation Matrix")
        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            st.plotly_chart(plot_correlation_heatmap(history_df), use_container_width=True)
        with col_c2:
            st.markdown(f"""
<div class="context-box">
<div class="context-header">💡 How to Read</div>
<div>Red = Inverse Correlation (Hedge). Blue = Positive Correlation (Moving Together).</div>
<br>
<div class="context-header">🔎 Analyst Note</div>
<div>Look for breakdowns. If SPY and TLT are both Blue, stock-bond correlation is positive (Risk Parity pain). Ideally, you want DXY to be Red vs SPY.</div>
</div>
""", unsafe_allow_html=True)

        st.divider()
        
        # ROW 4: RRG
        st.subheader("🎯 Momentum Quadrants (RRG)")
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("**SECTORS**")
            st.plotly_chart(plot_rrg(market_data, 'SECTORS', timeframe), use_container_width=True)
            st.markdown(f"""
<div class="context-box">
<div class="context-header">💡 RRG Logic</div>
<div><strong>Leading (Green):</strong> Strong Trend + Momentum.<br><strong>Weakening (Yellow):</strong> Strong Trend, Momentum fading.<br><strong>Lagging (Red):</strong> Downtrend.<br><strong>Improving (Blue):</strong> Downtrend ending, momentum building.</div>
</div>
""", unsafe_allow_html=True)

        with q2:
            st.markdown("**MACRO ASSETS**")
            st.plotly_chart(plot_rrg(market_data, 'ASSETS', timeframe), use_container_width=True)
            st.markdown(f"""
<div class="context-box">
<div class="context-header">🔎 Rotation Watch</div>
<div>The best trades are often found in the <strong>Improving (Blue)</strong> quadrant as they cross into Leading. Avoid assets deep in the Lagging (Red) quadrant unless you are short.</div>
</div>
""", unsafe_allow_html=True)

    # === TAB 3: MACRO MACHINE ===
    with tab_macro:
        st.subheader("🕸️ The Macro Transmission Mechanism")
        
        col_graph, col_legend = st.columns([3, 1])
        
        with col_graph:
            try:
                # Dynamic Graph using timeframe
                st.graphviz_chart(plot_nexus_graph(market_data, timeframe), use_container_width=True)
            except:
                st.warning("Graphviz executable not found.")
            
        with col_legend:
            st.markdown(f"""
<div class="context-box" style="margin-top: 0;">
<div class="context-header">💡 The Plumbing</div>
<div>This graph visualizes the causal chain of the economy.</div>
<br>
<div class="context-header">🟢 Green Node</div>
<div>Asset is rising in the selected timeframe ({timeframe}).</div>
<br>
<div class="context-header">🔴 Red Node</div>
<div>Asset is falling in the selected timeframe ({timeframe}).</div>
<br>
<div class="context-header">🔎 The Flow</div>
<div>1. <strong>Fed Policy:</strong> Starts at US10Y and DXY.<br>2. <strong>The Pipe:</strong> HYG (Credit) transmits the signal.<br>3. <strong>The Bucket:</strong> Risk assets (Tech, Crypto) catch the flow.</div>
</div>
""", unsafe_allow_html=True)

    # === TAB 4: PLAYBOOK ===
    with tab_playbook:
        st.subheader("📚 Strategy Reference Library")
        
        with st.expander("🟣 TIMEZONE (RUT Income)", expanded=False):
            st.markdown("""
            **Concept:** High probability income strategy for RUT (Russell 2000).
            * **Entry:** Thursday Afternoon (15 DTE in front month).
            * **Structure:** 1. Put Credit Spread (Income). 2. Put Calendar (Hedge).
            * **Target:** 5-7% Return on Capital.
            * **Hard Stop:** Exit by 7 DTE.
            """)
            
        with st.expander("🔵 TIMEEDGE (SPX Neutral)", expanded=False):
            st.markdown("""
            **Concept:** Pure Theta play utilizing decay differential.
            * **Structure:** Double Calendar (Sell 15 DTE / Buy 22 DTE).
            * **Entry:** Thursday @ 3:30 PM.
            * **Strikes:** ATM or slightly OTM.
            """)
            
        with st.expander("🌊 FLYAGONAL (Liquidity/Drift)", expanded=False):
            st.markdown("""
            **Concept:** Hybrid directional trade. Call BWB (Upside) + Put Diagonal (Downside).
            * **Entry:** 7-10 DTE.
            * **Exit:** Close at >4% Profit (Flash Win).
            """)
            
        with st.expander("🔴 A14 (Risk Off Hedge)", expanded=False):
            st.markdown("""
            **Concept:** Crash Catcher. Uses a Put Broken Wing Butterfly to finance downside protection.
            * **Structure:** Put BWB.
            * **Why:** Zero upside risk if filled for credit.
            """)

if __name__ == "__main__":
    main()
