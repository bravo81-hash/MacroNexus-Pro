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
</style>
""", unsafe_allow_html=True)

# --- 3. DATA UNIVERSE ---
TICKERS = {
    # Drivers
    'US10Y': '^TNX', 'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'HYG': 'HYG', 'TLT': 'TLT',
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
                prev_w = hist_clean.iloc[-6] # Weekly
                prev_m = hist_clean.iloc[-21] # Monthly
                
                # Logic: Yields are in basis points
                if key == 'US10Y':
                    chg_d = (curr - prev) * 10
                    chg_w = (curr - prev_w) * 10
                    chg_m = (curr - prev_m) * 10
                else:
                    chg_d = ((curr - prev) / prev) * 100
                    chg_w = ((curr - prev_w) / prev_w) * 100
                    chg_m = ((curr - prev_m) / prev_m) * 100
                
                data[key] = {
                    'price': curr, 'change': chg_d, 
                    'change_w': chg_w, 'change_m': chg_m,
                    'symbol': symbol, 'valid': True
                }
            else:
                data[key] = {'price': 0, 'change': 0, 'change_w': 0, 'change_m': 0, 'symbol': symbol, 'valid': False}
        except:
            data[key] = {'price': 0, 'change': 0, 'change_w': 0, 'change_m': 0, 'symbol': symbol, 'valid': False}
            
    # Create DataFrame for Correlations
    df_history = pd.DataFrame(history_data)
    return data, df_history

# --- 5. LOGIC ENGINE (THE BRAIN) ---
def determine_regime(data):
    # Safe getter
    def g(k): return data.get(k, {}).get('change', 0)
    
    hyg, vix = g('HYG'), g('VIX')
    oil, cop = g('OIL'), g('COPPER')
    us10y, dxy = g('US10Y'), g('DXY')
    btc, banks = g('BTC'), g('BANKS')
    
    # 1. RISK OFF (Priority 1)
    if hyg < -0.5 or vix > 5.0: return "RISK OFF"
    # 2. REFLATION (Growth + Yields)
    if (oil > 1.5 or cop > 1.5) and us10y > 3.0 and banks > 0: return "REFLATION"
    # 3. LIQUIDITY (Dollar Down + Beta Up)
    if dxy < -0.3 and btc > 2.0: return "LIQUIDITY"
    # 4. GOLDILOCKS (Vol Down + Yields Calm + Credit Stable)
    if vix < 0 and abs(us10y) < 5.0 and hyg > -0.1: return "GOLDILOCKS"
    
    return "NEUTRAL"

# --- 6. STRATEGY DATABASE (The Playbook) ---
STRATEGIES = {
    "GOLDILOCKS": {
        "desc": "Low Vol + Steady Trend",
        "risk": "1.5%", "bias": "Long",
        "index": {"strat": "Directional Diagonal", "dte": "Front 17 / Back 31", "setup": "Buy 70D Call / Sell 30D Call", "notes": "Stock replacement. Low vol favors diagonals."},
        "stock": {"strat": "Call Debit Spread", "dte": "45-60 DTE", "setup": "Buy 60D / Sell 30D", "filter": "Price > SMA50 | RSI 50-70"}
    },
    "LIQUIDITY": {
        "desc": "High Liquidity / Drift Up",
        "risk": "1.0%", "bias": "Aggressive Long",
        "index": {"strat": "Flyagonal (Drift)", "dte": "Entry 7-14 DTE", "setup": "Call BWB (Upside) + Put Diagonal (Downside)", "notes": "Captures the drift. Close at 4% profit."},
        "stock": {"strat": "Zebra / Long Call", "dte": "60+ DTE", "setup": "Buy 2x 70D / Sell 1x 50D", "filter": "ADX > 25 | High Relative Vol"}
    },
    "REFLATION": {
        "desc": "Inflation / Rates Rising",
        "risk": "1.0%", "bias": "Cyclical Long",
        "index": {"strat": "Directional Diagonal (IWM)", "dte": "Front 17 / Back 31", "setup": "Long 70D / Short 30D", "notes": "Focus on Russell 2000 (IWM) over Tech."},
        "stock": {"strat": "Call Spread (Cyclicals)", "dte": "45 DTE", "setup": "ATM Strikes", "filter": "Energy / Banks / Industrials"}
    },
    "NEUTRAL": {
        "desc": "Chop / Range Bound",
        "risk": "Income Size", "bias": "Neutral/Theta",
        "index": {"strat": "TimeEdge Calendar / TimeZone", "dte": "Entry 15 / Exit 7", "setup": "Double Calendar or Put Calendar (ATM)", "notes": "Low Vol Required. Hard exit at 7 DTE."},
        "stock": {"strat": "Iron Condor", "dte": "30-45 DTE", "setup": "Short 20D Wings", "filter": "ADX < 20 | BB Width > 0.1"}
    },
    "RISK OFF": {
        "desc": "High Volatility / Stress",
        "risk": "0.5%", "bias": "Short/Hedge",
        "index": {"strat": "A14 Put BWB", "dte": "Entry 14 / Exit 7", "setup": "ATM / -40 / -60 (Skip Strike)", "notes": "Zero upside risk. Profit tent catches crash."},
        "stock": {"strat": "Put Debit Spread", "dte": "60-90 DTE", "setup": "Buy 40D / Sell 15D", "filter": "Price < SMA50 | High Beta"}
    }
}

# --- 7. VISUALIZATION FUNCTIONS ---
def plot_nexus_graph(data):
    # Logic: Colors based on Daily Change. Red/Green.
    # Flow: Fed -> Rates/DXY -> Assets
    
    # 1. Create Graph
    dot = graphviz.Digraph(comment='The Macro Machine')
    dot.attr(rankdir='LR', bgcolor='#0e1117') # Left to Right flow
    dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontcolor='white')
    dot.attr('edge', color='#555555', arrowsize='0.8')

    def get_col(k, invert=False):
        c = data.get(k, {}).get('change', 0)
        if invert: return '#ef4444' if c > 0 else '#22c55e' # Red if up (Bad)
        return '#22c55e' if c > 0 else '#ef4444' # Green if up (Good)

    # NODES
    # Source
    dot.node('FED', 'FED POLICY\n(Liquidity)', fillcolor='#3b82f6')
    
    # Transmission
    dot.node('US10Y', f'YIELDS\n{data["US10Y"]["change"]:+.1f} bps', fillcolor=get_col('US10Y', True))
    dot.node('DXY', f'DOLLAR\n{data["DXY"]["change"]:+.2f}%', fillcolor=get_col('DXY', True))
    
    # Risk Gate
    dot.node('HYG', f'CREDIT (HYG)\n{data["HYG"]["change"]:+.2f}%', fillcolor=get_col('HYG', False))
    
    # Assets
    dot.node('TECH', 'TECH (QQQ)', fillcolor=get_col('QQQ', False))
    dot.node('GOLD', 'GOLD', fillcolor=get_col('GOLD', False))
    dot.node('CRYPTO', 'CRYPTO', fillcolor=get_col('BTC', False))
    dot.node('EM', 'EMERGING', fillcolor=get_col('EEM', False))
    dot.node('CYCL', 'CYCLICALS', fillcolor=get_col('ENERGY', False))

    # EDGES
    dot.edge('FED', 'US10Y', 'Rates')
    dot.edge('FED', 'DXY', 'Currency')
    
    dot.edge('US10Y', 'TECH', 'Cost of Capital', color='#ef4444') # Inverse relationship usually
    dot.edge('US10Y', 'GOLD', 'Real Rates', color='#ef4444')
    
    dot.edge('DXY', 'EM', 'Debt Pressure', color='#ef4444')
    dot.edge('DXY', 'GOLD', 'Pricing', color='#ef4444')
    
    dot.edge('US10Y', 'HYG', 'Refinancing')
    dot.edge('HYG', 'TECH', 'Risk On')
    dot.edge('HYG', 'CRYPTO', 'Liquidity Spillover')
    
    return dot

def plot_rrg(data, category, view):
    # Prepare Data
    items = []
    
    if category == 'SECTORS':
        keys = ['TECH','SEMIS','BANKS','ENERGY','HOME','UTIL','STAPLES','DISC','IND','HEALTH','MAT','COMM','RE']
    else: # ASSETS
        keys = ['SPY','QQQ','IWM','GOLD','BTC','TLT','DXY','HYG','OIL']
        
    for k in keys:
        d = data.get(k, {})
        # Tactical (Daily): X=Weekly, Y=Daily
        # Structural (Weekly): X=Monthly, Y=Weekly
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
    
    # Annotations
    limit = max(df['Trend'].abs().max(), df['Momentum'].abs().max()) * 1.1
    fig.update_layout(
        xaxis=dict(range=[-limit, limit], zeroline=False, showgrid=True, gridcolor='#333'),
        yaxis=dict(range=[-limit, limit], zeroline=False, showgrid=True, gridcolor='#333'),
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
    sectors = {k: data.get(k, {}).get('change', 0) for k in ['TECH','SEMIS','BANKS','ENERGY','HOME','UTIL','HEALTH','MAT','COMM']}
    df = pd.DataFrame(list(sectors.items()), columns=['id', 'val']).sort_values('val', ascending=False)
    
    winners = df.head(3)
    losers = df.tail(3)
    
    labels = list(losers['id']) + list(winners['id'])
    sources, targets, values, colors = [], [], [], []
    
    # Link every loser to every winner (visualizing rotation)
    for i in range(len(losers)):
        for j in range(len(winners)):
            sources.append(i) 
            targets.append(len(losers) + j)
            # Size of flow based on magnitude of divergence
            flow_size = abs(losers.iloc[i]['val']) + abs(winners.iloc[j]['val'])
            values.append(flow_size)
            colors.append('rgba(255, 255, 255, 0.1)') # Subtle flow lines
            
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
    
    fig.update_layout(title_text="Capital Rotation Flow (Intraday)", font=dict(color='white'), paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plot_correlation_heatmap(history_df):
    # Calculate Correlation
    corr = history_df.pct_change().corr()
    
    # Select key assets for cleaner view
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
        val = d.get('price', 0); chg = d.get('change', 0)
        
        # Color Logic
        is_up = chg > 0
        if invert: color = "#F43F5E" if is_up else "#10B981" # Red if up
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
    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    
    with ctrl1:
        # Manual Override
        override = st.checkbox("Manual Override", value=False)
        auto_regime = determine_regime(market_data)
        
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
        rc = r_colors[active_regime]
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
        # 1. TELEMETRY INPUTS
        with st.expander("🎛️ SPX Income Reactor Telemetry (Input from TradingView)", expanded=True):
            tc1, tc2, tc3 = st.columns(3)
            iv_rank = tc1.slider("IV Rank (Percentile)", 0, 100, 45)
            skew_rank = tc2.slider("Skew Rank", 0, 100, 50)
            adx_val = tc3.slider("Trend ADX", 0, 60, 20)

        st.divider()

        # 2. LOGIC & EXECUTION
        strat_data = STRATEGIES[active_regime]
        
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
                    <div style="color: #10B981; font-size: 12px; margin-bottom: 4px;"><strong>TARGET:</strong> {strat_data.get('longs', 'See Strategy')}</div>
                    <div style="color: #EF4444; font-size: 12px;"><strong>AVOID:</strong> {strat_data.get('shorts', 'See Strategy')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_R:
            # Tactical Execution Card
            st.markdown(f'<div class="strat-title">TACTICAL EXECUTION</div>', unsafe_allow_html=True)
            mode = st.radio("Asset Class", ["INDEX (SPX/RUT)", "STOCKS"], horizontal=True)
            
            sel_strat = {}
            
            # --- THE REACTOR LOGIC ---
            if mode == "INDEX (SPX/RUT)":
                if active_regime == "RISK OFF":
                    sel_strat = STRATEGIES["RISK OFF"]["index"]
                elif active_regime == "NEUTRAL":
                    # TimeZone Check
                    sel_strat = STRATEGIES["NEUTRAL"]["index"]
                    if "RUT" in TICKERS: sel_strat['name_alt'] = "TimeZone (RUT)"
                elif iv_rank > 50: # High Vol
                    if skew_rank > 80:
                        sel_strat = {"strat": "Put BWB (High Skew)", "dte": "21-30 DTE", "setup": "OTM Puts", "notes": "High skew warns of crash. Use BWB."}
                    else:
                        sel_strat = {"strat": "Iron Condor", "dte": "30-45 DTE", "setup": "Delta 15 Wings", "notes": "Classic Vol Crush."}
                else: # Low Vol
                    if adx_val > 25:
                        sel_strat = STRATEGIES["GOLDILOCKS"]["index"]
                    elif active_regime == "LIQUIDITY":
                        sel_strat = STRATEGIES["LIQUIDITY"]["index"]
                    else:
                        sel_strat = {"strat": "Double Diagonal", "dte": "Front 17 / Back 31", "setup": "OTM", "notes": "Low Vol Expansion play."}
            else:
                # Stock Logic is simpler
                sel_strat = strat_data["stock"]

            # RENDER CARD
            st.markdown(f"""
            <div class="strat-card" style="border-color: {rc}; background: linear-gradient(145deg, {rc}11 0%, rgba(20,22,28,1) 100%);">
                <div class="strat-title" style="color: {rc}; border-color: {rc}44;">{sel_strat['strat']}</div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 15px;">
                    <div>
                        <div class="strat-subtitle">TIMING (DTE)</div>
                        <div class="strat-data" style="font-weight: 700;">{sel_strat['dte']}</div>
                    </div>
                    <div>
                        <div class="strat-subtitle">STRUCTURE</div>
                        <div class="strat-data" style="font-weight: 700;">{sel_strat['setup']}</div>
                    </div>
                </div>
                
                <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1);">
                    <div class="strat-subtitle" style="margin-top: 0;">THE LOGIC</div>
                    <div class="strat-data" style="font-size: 13px; color: #9CA3AF;">{sel_strat['notes'] if 'notes' in sel_strat else sel_strat.get('logic', '')}</div>
                </div>
                
                {f'<div style="margin-top: 10px; font-family: monospace; font-size: 12px; color: {rc};">🔍 SCREENER: {sel_strat["filter"]}</div>' if 'filter' in sel_strat else ''}
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
            st.graphviz_chart(plot_nexus_graph(market_data), use_container_width=True)
            
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
            **Concept:** High probability income strategy for RUT.
            * **Entry:** Thursday Afternoon (15 DTE).
            * **Structure:** Put Credit Spread (Income) + Put Calendar (Hedge).
            * **Ratio:** 2x PCS / 2x Calendar.
            * **Hard Stop:** Exit by 7 DTE (gamma risk).
            * **Target:** 5-7% Return on Capital.
            """)
            
        with st.expander("🔵 TIMEEDGE (SPX Neutral)", expanded=False):
            st.markdown("""
            **Concept:** Pure Theta play utilizing decay differential.
            * **Structure:** Double Calendar or Put Calendar.
            * **Entry:** Thursday @ 3:30 PM (15 DTE Front / 22 DTE Back).
            * **Strikes:** ATM.
            * **Constraint:** Back month IV cannot be >1pt higher than Front month.
            """)
            
        with st.expander("🌊 FLYAGONAL (Liquidity/Drift)", expanded=False):
            st.markdown("""
            **Concept:** Hybrid directional trade merging Call BWB (Upside) + Put Diagonal (Downside).
            * **Call Side:** 1 Long (ATM+10) / 2 Short (ATM+50) / 1 Long (ATM+60).
            * **Put Side:** Sell 1 Put (ATM-30) / Buy 1 Put (ATM-40) in later expiry.
            * **Entry:** 7-10 DTE.
            * **Management:** Close at >4% Profit (Flash Win).
            """)
            
        with st.expander("🔴 A14 (Risk Off Hedge)", expanded=False):
            st.markdown("""
            **Concept:** Crash Catcher.
            * **Structure:** Put BWB (Broken Wing).
            * **Entry:** Friday Morning (14 DTE).
            * **Strikes:** Long ATM / Short -40pts / Skip / Long -60pts.
            * **Why:** Zero upside risk if filled for credit. Catches the crash.
            """)

if __name__ == "__main__":
    main()
