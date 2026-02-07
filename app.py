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
    .badge-green { background: rgba(16, 185, 129, 0.2); color: #10B981; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-red { background: rgba(239, 68, 68, 0.2); color: #EF4444; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-blue { background: rgba(59, 130, 246, 0.2); color: #3B82F6; padding: 2px 6px; border-radius: 4px; font-size: 10px; border: 1px solid rgba(59, 130, 246, 0.3); }
    
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
# Fallbacks for known reliable tickers if Yahoo fails on specific symbols
FALLBACKS = {'DXY': 'UUP', 'VIX': 'VIXY', 'RUT': 'IWM'}

# --- 4. DATA ENGINE ---
@st.cache_data(ttl=300)
def fetch_market_data():
    data = {}
    history_data = {} # Store full history for correlation
    
    for key, symbol in TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            # Fallback logic
            if hist.empty and key in FALLBACKS:
                symbol = FALLBACKS[key]
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="6mo")
            
            hist_clean = hist['Close'].dropna()
            
            if not hist_clean.empty and len(hist_clean) >= 22:
                # Store history for correlation matrix
                history_data[key] = hist_clean
                
                curr = hist_clean.iloc[-1]
                prev = hist_clean.iloc[-2]
                prev_w = hist_clean.iloc[-6] # Weekly (~5 trading days)
                prev_m = hist_clean.iloc[-21] # Monthly (~20 trading days)
                
                # Logic: Yields are in basis points for display, but raw for logic
                if key == 'US10Y':
                    # Yahoo often returns e.g. 4.25 for 4.25%. 
                    chg_d = (curr - prev) * 100 # Basis Points
                    chg_w = (curr - prev_w) * 100
                    chg_m = (curr - prev_m) * 100
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
            
    # Create DataFrame for Correlations
    df_history = pd.DataFrame(history_data)
    return data, df_history

# --- 5. LOGIC ENGINE (THE BRAIN) ---
def determine_regime(data):
    # Safe getter for daily change
    def g(k): return data.get(k, {}).get('change', 0)
    
    hyg, vix = g('HYG'), g('VIX')
    oil, cop = g('OIL'), g('COPPER')
    us10y, dxy = g('US10Y'), g('DXY')
    btc, banks = g('BTC'), g('BANKS')
    
    # 1. RISK OFF (Priority 1: The Veto Check)
    # Logic: HYG Breaking Down (< -0.5%) OR VIX Spiking (> 5%)
    if hyg < -0.5 or vix > 5.0: return "RISK OFF"
    
    # 2. REFLATION (Growth + Yields + Commodities)
    if (oil > 1.5 or cop > 1.5) and us10y > 3.0 and banks > 0: return "REFLATION"
    
    # 3. LIQUIDITY (Dollar Down + Beta/Crypto Up)
    if dxy < -0.3 and btc > 1.5: return "LIQUIDITY"
    
    # 4. GOLDILOCKS (Vol Down + Yields Calm + Credit Stable)
    if vix < 0 and abs(us10y) < 5.0 and hyg > -0.1: return "GOLDILOCKS"
    
    return "NEUTRAL"

# --- 6. STRATEGY DATABASE (The Playbook) ---
# Merges concepts from A14, Flyagonal, TimeEdge, TimeZone documents
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

# --- 7. VISUALIZATION FUNCTIONS ---

def plot_nexus_graph(data):
    # Colors based on Daily Change. Red/Green.
    # Logic: Fed -> Rates/DXY -> Assets
    
    dot = graphviz.Digraph(comment='The Macro Machine')
    dot.attr(rankdir='LR', bgcolor='#0e1117') # Left to Right flow
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontcolor='white')
    dot.attr('edge', color='#555555', arrowsize='0.8')

    def get_col(k, invert=False):
        c = data.get(k, {}).get('change', 0)
        if invert: return '#ef4444' if c > 0 else '#22c55e' # Red if up (Bad)
        return '#22c55e' if c > 0 else '#ef4444' # Green if up (Good)

    # NODES
    dot.node('FED', 'FED POLICY\n(Liquidity)', fillcolor='#3b82f6')
    
    # Transmission
    us10y_c = data.get("US10Y", {}).get('change', 0)
    dxy_c = data.get("DXY", {}).get('change', 0)
    hyg_c = data.get("HYG", {}).get('change', 0)
    
    dot.node('US10Y', f'YIELDS\n{us10y_c:+.1f} bps', fillcolor=get_col('US10Y', True))
    dot.node('DXY', f'DOLLAR\n{dxy_c:+.2f}%', fillcolor=get_col('DXY', True))
    
    # Risk Gate
    dot.node('HYG', f'CREDIT (HYG)\n{hyg_c:+.2f}%', fillcolor=get_col('HYG', False))
    
    # Assets
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
        # Tactical (Daily): X=WeeklyTrend, Y=DailyMom
        # Structural (Weekly): X=MonthlyTrend, Y=WeeklyMom
        if view == 'Tactical (Daily)':
            x = d.get('change_w', 0)
            y = d.get('change', 0)
        else:
            x = d.get('change_m', 0)
            y = d.get('change_w', 0)
            
        # Color Logic
        c = 'gray'
        if x>0 and y>0: c='#22c55e' # Leading
        elif x<0 and y>0: c='#3b82f6' # Improving
        elif x>0 and y<0: c='#f59e0b' # Weakening
        elif x<0 and y<0: c='#ef4444' # Lagging
        
        items.append({'Symbol': k, 'Trend': x, 'Momentum': y, 'Color': c})
        
    df = pd.DataFrame(items)
    
    fig = px.scatter(df, x='Trend', y='Momentum', text='Symbol', color='Color', color_discrete_map="identity")
    fig.update_traces(textposition='top center', marker=dict(size=14, line=dict(width=1, color='white')))
    
    # Quadrant Lines
    fig.add_hline(y=0, line_dash="dot", line_color="#555")
    fig.add_vline(x=0, line_dash="dot", line_color="#555")
    
    limit = max(df['Trend'].abs().max(), df['Momentum'].abs().max()) * 1.1 if not df.empty else 1
    fig.update_layout(
        xaxis=dict(range=[-limit, limit], zeroline=False, showgrid=True, gridcolor='#333', title="Trend (Lagging)"),
        yaxis=dict(range=[-limit, limit], zeroline=False, showgrid=True, gridcolor='#333', title="Momentum (Leading)"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'),
        showlegend=False,
        height=450,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

def plot_sankey(data):
    # Calculate flows: Losers (Left) -> Winners (Right)
    # Using Sector data
    sector_keys = ['TECH','SEMIS','BANKS','ENERGY','HOME','UTIL','HEALTH','MAT','COMM']
    sectors = {k: data.get(k, {}).get('change', 0) for k in sector_keys}
    df = pd.DataFrame(list(sectors.items()), columns=['id', 'val']).sort_values('val', ascending=False)
    
    winners = df.head(3)
    losers = df.tail(3)
    
    labels = list(losers['id']) + list(winners['id'])
    sources, targets, values, colors = [], [], [], []
    
    for i in range(len(losers)):
        for j in range(len(winners)):
            sources.append(i) 
            targets.append(len(losers) + j)
            # Size of flow based on divergence magnitude
            flow_size = abs(losers.iloc[i]['val']) + abs(winners.iloc[j]['val'])
            values.append(flow_size)
            colors.append('rgba(255, 255, 255, 0.1)') 
            
    node_colors = ['#ef4444']*3 + ['#22c55e']*3 # Red left, Green right
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(source=sources, target=targets, value=values, color=colors)
    )])
    
    fig.update_layout(title_text="Intraday Capital Rotation Flow", font=dict(color='white'), paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plot_correlation_heatmap(history_df):
    if history_df.empty: return go.Figure()
    
    corr = history_df.pct_change().corr()
    
    subset = ['US10Y', 'DXY', 'VIX', 'HYG', 'SPY', 'QQQ', 'IWM', 'BTC', 'GOLD', 'OIL']
    # Filter only available columns
    cols = [c for c in subset if c in corr.columns]
    corr_subset = corr.loc[cols, cols]
    
    fig = px.imshow(
        corr_subset,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r", # Red = Neg (Hedge), Blue = Pos (Correlated)
        zmin=-1, zmax=1
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#ccc'),
        height=400
    )
    return fig

# --- 8. MAIN APPLICATION ---
def main():
    # --- LOAD DATA ---
    with st.spinner("Connecting to MacroNexus Core..."):
        market_data, history_df = fetch_market_data()
        
    # --- TOP METRICS BAR ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    def metric_tile(col, label, key, invert=False):
        d = market_data.get(key, {})
        val = d.get('price', 0); chg = d.get('change', 0); fmt = d.get('fmt', "%")
        
        # Color Logic
        is_up = chg > 0
        if invert: color = "#F43F5E" if is_up else "#10B981" # Red if up (Bad)
        else: color = "#10B981" if is_up else "#F43F5E" # Green if up
        
        fmt_chg = f"{chg:+.1f} bps" if key == 'US10Y' else f"{chg:+.2f}%"
        
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

    # --- MAIN DASHBOARD CONTROLS ---
    # Determine Auto Regime
    auto_regime = determine_regime(market_data)

    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    
    with ctrl1:
        # Manual Override
        override = st.checkbox("Manual Override", value=False)
        if override:
            active_regime = st.selectbox("Force Regime", list(STRATEGIES.keys()), label_visibility="collapsed")
        else:
            active_regime = auto_regime
            
    with ctrl2:
        # Timeframe for Analytics
        timeframe = st.selectbox("Analytic View", ["Tactical (Daily)", "Structural (Weekly)"], label_visibility="collapsed")
        
    with ctrl3:
        # Regime Display
        r_colors = {"GOLDILOCKS": "#10B981", "LIQUIDITY": "#A855F7", "REFLATION": "#F59E0B", "NEUTRAL": "#6B7280", "RISK OFF": "#EF4444"}
        rc = r_colors.get(active_regime, "#6B7280")
        
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
        # 1. SPX REACTOR TELEMETRY INPUTS
        # Used to refine the generic regime into a specific tactical trade
        with st.expander("🎛️ SPX Income Reactor Telemetry (Manual Input)", expanded=True):
            tc1, tc2, tc3 = st.columns(3)
            iv_rank = tc1.slider("IV Rank (Percentile)", 0, 100, 45, help="Low < 30 | High > 50")
            skew_rank = tc2.slider("Skew Rank", 0, 100, 50, help="High Skew (>80) indicates Crash Risk")
            adx_val = tc3.slider("Trend ADX", 0, 60, 20, help="< 20 is Chop/Range. > 30 is Strong Trend")

        st.divider()

        # 2. STRATEGY LOGIC ENGINE
        strat_data = STRATEGIES[active_regime]
        
        # --- THE REACTOR LOGIC (Dynamic Refinement) ---
        # This overrides the generic regime strategy based on specific Vol/Skew inputs
        reactor_output = {}
        
        if active_regime == "RISK OFF":
            reactor_output = STRATEGIES["RISK OFF"]["index"]
        elif iv_rank > 50: # High Volatility Environment
            if skew_rank > 80:
                reactor_output = {"strat": "Put BWB (High Skew)", "dte": "21-30 DTE", "setup": "Long ATM / Short -40 / Skip / Long -60", "notes": "REACTOR WARNING: High Skew detected. Crash risk elevated. Use BWB to eliminate upside risk and profit from crash."}
            else:
                reactor_output = {"strat": "Iron Condor", "dte": "30-45 DTE", "setup": "Delta 15 Wings", "notes": "REACTOR: Volatility is high but Skew is normal. Classic reversion to mean play."}
        else: # Low/Normal Volatility
            if adx_val > 25:
                if active_regime == "LIQUIDITY":
                    reactor_output = STRATEGIES["LIQUIDITY"]["index"] # Flyagonal
                else:
                    reactor_output = {"strat": "Directional Diagonal", "dte": "Front 17 / Back 31", "setup": "Buy Back ITM / Sell Front OTM", "notes": "REACTOR: Trend detected (ADX > 25). Do not cap upside."}
            else:
                # Low Vol + Low Trend = Calendars
                reactor_output = STRATEGIES["NEUTRAL"]["index"]

        # DISPLAY CARDS
        col_L, col_R = st.columns([1, 2])
        
        with col_L:
            # Context Card
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
            # Tactical Execution Card
            st.markdown(f'<div class="strat-title">TACTICAL EXECUTION</div>', unsafe_allow_html=True)
            
            # RENDER CARD
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
        row1_1, row1_2 = st.columns(2)
        
        with row1_1:
            st.subheader("🌊 Capital Flow (Sankey)")
            st.plotly_chart(plot_sankey(market_data), use_container_width=True)
            
        with row1_2:
            st.subheader("🔥 Inter-Correlation Matrix")
            st.plotly_chart(plot_correlation_heatmap(history_df), use_container_width=True)
            
        st.divider()
        
        st.subheader("🎯 Momentum Quadrants (RRG)")
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("**SECTORS**")
            st.plotly_chart(plot_rrg(market_data, 'SECTORS', timeframe), use_container_width=True)
        with q2:
            st.markdown("**MACRO ASSETS**")
            st.plotly_chart(plot_rrg(market_data, 'ASSETS', timeframe), use_container_width=True)

    # === TAB 3: MACRO MACHINE ===
    with tab_macro:
        st.subheader("🕸️ The Macro Transmission Mechanism")
        st.markdown("Visualizing how Fed Policy (Rates/USD) flows downstream to Risk Assets.")
        
        col_graph, col_legend = st.columns([3, 1])
        
        with col_graph:
            try:
                st.graphviz_chart(plot_nexus_graph(market_data), use_container_width=True)
            except:
                st.warning("Graphviz executable not found. Please install Graphviz to view the transmission graph.")
            
        with col_legend:
            st.info("""
            **LEGEND:**
            
            🟢 **Green Node:** Asset is Rising
            🔴 **Red Node:** Asset is Falling
            
            **THE FLOW:**
            1. **Fed Policy:** Look at `US10Y` and `DXY`. If these are RED (Rising), they restrict flow.
            2. **The Pipe:** `HYG` (Credit) is the pipe. If RED, the pipe is clogged (Risk Off).
            3. **The Bucket:** If flow is good, `TECH`, `CRYPTO` fill up.
            """)

    # === TAB 4: PLAYBOOK ===
    with tab_playbook:
        st.subheader("📚 Strategy Reference Library")
        
        with st.expander("🟣 TIMEZONE (RUT Income)", expanded=False):
            st.markdown("""
            **Concept:** High probability income strategy for RUT (Russell 2000).
            
            * **Entry:** Thursday Afternoon (15 DTE in front month).
            * **Structure:** 1. **Put Credit Spread (Income):** Sell 14 Delta / Buy 20pts lower. (2 Contracts)
                2. **Put Calendar (Hedge):** Sell Front Month (15 DTE) / Buy Back Month (~45 DTE) at same strike (~40 Delta). (2 Contracts)
            * **Target:** 5-7% Return on Capital.
            * **Hard Stop:** Exit by 7 DTE (gamma risk). Max loss ~5%.
            * **Adjustments:**
                * *Rally:* Reverse Diagonal (Sell back month long / Buy lower strike front month put).
                * *Drop:* Add Put Debit Spread or simple Long Put hedge.
            """)
            
        with st.expander("🔵 TIMEEDGE (SPX Neutral)", expanded=False):
            st.markdown("""
            **Concept:** Pure Theta play utilizing decay differential between front and back months.
            
            * **Structure:** Double Calendar (Sell 15 DTE / Buy 22 DTE) OR Put Calendar.
            * **Entry:** Thursday @ 3:30 PM.
            * **Strikes:** ATM or slightly OTM (35 Delta).
            * **Constraint:** Back month IV cannot be >1pt higher than Front month (Vol Skew check).
            * **Management:** * One adjustment only (Roll strikes or convert to diagonal).
                * Hard exit at 7 DTE.
            """)
            
        with st.expander("🌊 FLYAGONAL (Liquidity/Drift)", expanded=False):
            st.markdown("""
            **Concept:** Hybrid directional trade merging Call BWB (Upside) + Put Diagonal (Downside). Designed for "Liquidity Pump" regimes where market drifts up.
            
            * **Call Side (Upside Tent):** Call Broken Wing Butterfly.
                * 1 Long (ATM+10) / 2 Short (ATM+50) / 1 Long (ATM+60).
            * **Put Side (Downside Floor):** Put Diagonal.
                * Sell 1 Put (ATM-30) / Buy 1 Put (ATM-40) in later expiry.
            * **Entry:** 7-10 DTE.
            * **Exit:** Close at >4% Profit (Flash Win).
            """)
            
        with st.expander("🔴 A14 (Risk Off Hedge)", expanded=False):
            st.markdown("""
            **Concept:** Crash Catcher. Uses a Put Broken Wing Butterfly to finance downside protection.
            
            * **Structure:** Put BWB.
            * **Entry:** Friday Morning (14 DTE).
            * **Strikes:** * Long 1x ATM Put
                * Short 2x OTM Puts (-40 pts)
                * Long 1x OTM Put (-60 pts) -- *Note: The skip creates the broken wing.*
            * **Why:** Zero upside risk if filled for credit. Profit tent expands massively if market crashes into the short strikes.
            * **Rule:** Exit at 7 DTE.
            """)

if __name__ == "__main__":
    main()
