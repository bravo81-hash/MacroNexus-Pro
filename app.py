import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import graphviz
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(
    page_title="MacroNexus Pro Terminal v11",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Metrics */
    .metric-container {
        background-color: #1e2127;
        padding: 10px 12px;
        border-radius: 6px;
        border-left: 4px solid #4b5563;
        margin-bottom: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; }
    .metric-label { font-size: 10px; color: #9ca3af; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
    .metric-val { font-size: 18px; font-weight: bold; color: #f3f4f6; }
    .metric-chg { font-size: 12px; font-weight: bold; margin-left: 6px; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; border-bottom: 1px solid #2e3039; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: 600; font-size: 14px; }
    
    /* Regime Badge */
    .regime-badge { padding: 15px; border-radius: 8px; text-align: center; border: 1px solid; margin-bottom: 20px; background: #1e2127; }
    
    /* Strategy Cards */
    .strat-card { background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #374151; margin-bottom: 20px; }
    .strat-header { font-size: 16px; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #4b5563; padding-bottom: 5px; color: #fff; }
    .strat-sub { font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .strat-val { font-size: 14px; color: #e0e0e0; font-family: monospace; font-weight: 600; margin-bottom: 12px; }
    
    /* Control Panel */
    .control-panel { background-color: #1e2127; padding: 15px; border-radius: 8px; border: 1px solid #374151; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 1. FULL DATA UNIVERSE ---
TICKERS = {
    # DRIVERS
    'US10Y': '^TNX',       # 10Y Yield (CBOE)
    'DXY': 'DX-Y.NYB',     # Primary Index
    'VIX': '^VIX',         # Primary Spot VIX
    'HYG': 'HYG',          # Credit High Yield
    'TLT': 'TLT',          # 20Y Bonds
    
    # COMMODITIES
    'GOLD': 'GLD', 'SILVER': 'SLV', 'OIL': 'USO',
    'NATGAS': 'UNG', 'COPPER': 'CPER', 'AG': 'DBA',
    
    # INDICES
    'SPY': 'SPY', 'QQQ': 'QQQ', 'IWM': 'IWM',
    'EEM': 'EEM', 'FXI': 'FXI',
    
    # SECTORS
    'TECH': 'XLK', 'SEMIS': 'SMH', 'BANKS': 'XLF',
    'ENERGY': 'XLE', 'HOME': 'XHB', 'UTIL': 'XLU',
    'STAPLES': 'XLP', 'DISC': 'XLY', 'IND': 'XLI',
    'HEALTH': 'XLV', 'MAT': 'XLB', 'COMM': 'XLC', 'RE': 'XLRE',
    
    # CRYPTO
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD'
}

FALLBACKS = {'DXY': 'UUP', 'VIX': 'VIXY'}

# --- 2. DATA FETCHING ---
@st.cache_data(ttl=300)
def fetch_live_data():
    data_map = {}
    for key, symbol in TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo") # 6mo for RRG calculations
            
            if hist.empty and key in FALLBACKS:
                symbol = FALLBACKS[key]
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="6mo")
            
            hist_clean = hist['Close'].dropna()

            if not hist_clean.empty and len(hist_clean) >= 22:
                current = hist_clean.iloc[-1]
                prev = hist_clean.iloc[-2]
                prev_week = hist_clean.iloc[-6]
                prev_month = hist_clean.iloc[-21]
                
                if key == 'US10Y':
                    change = (current - prev) * 10
                    change_w = (current - prev_week) * 10
                    change_m = (current - prev_month) * 10
                else:
                    change = ((current - prev) / prev) * 100
                    change_w = ((current - prev_week) / prev_week) * 100
                    change_m = ((current - prev_month) / prev_month) * 100
                
                data_map[key] = {
                    'price': current, 'change': change, 'change_w': change_w,
                    'change_m': change_m, 'symbol': symbol, 'error': False
                }
            else:
                data_map[key] = {'price': 0.0, 'change': 0.0, 'change_w': 0.0, 'change_m': 0.0, 'symbol': symbol, 'error': True}
        except Exception:
            data_map[key] = {'price': 0.0, 'change': 0.0, 'change_w': 0.0, 'change_m': 0.0, 'symbol': symbol, 'error': True}
            
    return data_map

# --- 3. ANALYTICS ENGINE ---
def get_regime(data):
    def get_c(k): return data.get(k, {}).get('change', 0)
    hyg, vix = get_c('HYG'), get_c('VIX')
    oil, cop = get_c('OIL'), get_c('COPPER')
    us10y, dxy = get_c('US10Y'), get_c('DXY')
    btc, banks = get_c('BTC'), get_c('BANKS')

    if hyg < -0.5 or vix > 5.0: return "RISK OFF"
    if (oil > 2.0 or cop > 2.0) and us10y > 5.0 and banks > 0: return "REFLATION"
    if dxy < -0.4 and btc > 3.0: return "LIQUIDITY"
    if vix < 0 and abs(us10y) < 5.0 and hyg > -0.1: return "GOLDILOCKS"
    return "NEUTRAL"

def get_quadrants(data, view):
    points = []
    # SECTORS + ASSETS
    keys = ['TECH', 'SEMIS', 'BANKS', 'ENERGY', 'HOME', 'UTIL', 'STAPLES', 'DISC', 'IND', 'HEALTH', 'MAT', 'COMM', 'RE', 'SPY', 'QQQ', 'IWM', 'GOLD', 'BTC', 'TLT', 'DXY']
    
    for k in keys:
        d = data.get(k, {})
        # Tactical (Daily): X=Weekly Trend, Y=Daily Mom
        # Structural (Weekly): X=Monthly Trend, Y=Weekly Mom
        if view == 'Tactical (Daily)':
            x = d.get('change_w', 0)
            y = d.get('change', 0)
        else:
            x = d.get('change_m', 0)
            y = d.get('change_w', 0)
            
        color = '#6b7280'
        if x > 0 and y > 0: color = '#22c55e' # Leading
        elif x < 0 and y > 0: color = '#3b82f6' # Improving
        elif x > 0 and y < 0: color = '#f59e0b' # Weakening
        elif x < 0 and y < 0: color = '#ef4444' # Lagging
        
        points.append({'id': k, 'x': x, 'y': y, 'color': color})
        
    return pd.DataFrame(points)

# --- 4. VISUALIZATION FUNCTIONS ---
def create_nexus_graph(market_data):
    # Logic from v1.5
    nodes = {
        'US10Y': {'pos': (0, 0), 'label': 'Rates'}, 'DXY': {'pos': (0.8, 0.8), 'label': 'Dollar'},
        'SPY': {'pos': (-0.8, 0.8), 'label': 'S&P 500'}, 'QQQ': {'pos': (-1.2, 0.4), 'label': 'Nasdaq'},
        'GOLD': {'pos': (0.8, -0.8), 'label': 'Gold'}, 'HYG': {'pos': (-0.4, -0.8), 'label': 'Credit'},
        'BTC': {'pos': (-1.5, 1.5), 'label': 'Bitcoin'}, 'OIL': {'pos': (1.5, -0.4), 'label': 'Oil'},
        'COPPER': {'pos': (1.2, -1.2), 'label': 'Copper'}, 'IWM': {'pos': (-1.2, -1.0), 'label': 'Russell'},
        'SMH': {'pos': (-1.8, 0.8), 'label': 'Semis'}, 'XLE': {'pos': (1.8, -0.8), 'label': 'Energy'},
        'EEM': {'pos': (-0.5, -1.5), 'label': 'EM'}, 'XHB': {'pos': (-0.8, -0.4), 'label': 'Housing'},
        'XLF': {'pos': (1.5, -1.0), 'label': 'Banks'}, 'VIX': {'pos': (0, 1.5), 'label': 'Vol'}
    }
    edges = [('US10Y','QQQ'), ('US10Y','GOLD'), ('US10Y','XHB'), ('DXY','GOLD'), ('DXY','OIL'), ('DXY','EEM'), ('HYG','SPY'), ('HYG','IWM'), ('HYG','XLF'), ('QQQ','BTC'), ('QQQ','SMH'), ('COPPER','US10Y'), ('OIL','XLE'), ('VIX','SPY')]
    
    edge_x, edge_y = [], []
    for u, v in edges:
        x0, y0 = nodes[u]['pos']; x1, y1 = nodes[v]['pos']
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for key, info in nodes.items():
        x, y = info['pos']; node_x.append(x); node_y.append(y)
        d = market_data.get(key, {}); chg = d.get('change', 0)
        col = '#22c55e' if chg > 0 else '#ef4444'
        if key in ['US10Y', 'DXY', 'VIX']: col = '#ef4444' if chg > 0 else '#22c55e' # Inverse assets
        node_color.append(col); node_size.append(40)
        node_text.append(f"{info['label']} ({chg:+.2f}%)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#4b5563'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="bottom center", marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white'))))
    fig.update_layout(showlegend=False, margin=dict(b=0,l=0,r=0,t=0), xaxis=dict(showgrid=False, visible=False), yaxis=dict(showgrid=False, visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450)
    return fig

def create_sankey(data):
    sectors = {k: data.get(k, {}).get('change', 0) for k in ['TECH', 'SEMIS', 'BANKS', 'ENERGY', 'HOME', 'UTIL', 'HEALTH', 'IND', 'MAT']}
    df = pd.DataFrame(list(sectors.items()), columns=['id', 'val']).sort_values('val', ascending=False)
    winners = df.head(3); losers = df.tail(3)
    
    labels = list(losers['id']) + list(winners['id'])
    sources, targets, values, colors = [], [], [], []
    
    for i in range(len(losers)):
        for j in range(len(winners)):
            sources.append(i); targets.append(len(losers) + j)
            values.append(10); colors.append('rgba(100,100,100,0.3)')
            
    fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=['#ef4444']*3 + ['#22c55e']*3), link=dict(source=sources, target=targets, value=values, color=colors))])
    fig.update_layout(title_text="Capital Flow (Losers -> Winners)", font=dict(color='white'), paper_bgcolor='rgba(0,0,0,0)', height=300)
    return fig

def create_rrg(df, title):
    fig = px.scatter(df, x='x', y='y', text='id', color='color', color_discrete_map="identity")
    fig.add_hline(y=0, line_dash="dot", line_color="gray"); fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_traces(textposition='top center', marker=dict(size=12))
    fig.update_layout(title=title, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), xaxis=dict(title="Trend", zeroline=False), yaxis=dict(title="Momentum", zeroline=False), showlegend=False, height=400)
    return fig

# --- 5. MAIN APP ---
def main():
    # --- DATA & STATE ---
    market_data = fetch_live_data()
    
    # --- HEADER TILES ---
    cols = st.columns(6)
    metrics = [("Credit", 'HYG'), ("Volatility", 'VIX'), ("10Y Yield", 'US10Y'), ("Dollar", 'DXY'), ("Oil", 'OIL'), ("Bitcoin", 'BTC')]
    for i, (lbl, key) in enumerate(metrics):
        d = market_data.get(key, {})
        val = d.get('price', 0); chg = d.get('change', 0)
        color = "#ef4444" if chg < 0 else "#22c55e"
        if key in ['VIX', 'US10Y', 'DXY']: color = "#ef4444" if chg > 0 else "#22c55e" # Inverted
        cols[i].markdown(f"""<div class="metric-container" style="border-left-color: {color};"><div class="metric-header"><span class="metric-label">{lbl}</span></div><div><span class="metric-val">{val:.2f}</span><span class="metric-chg" style="color: {color};">{chg:+.2f}%</span></div></div>""", unsafe_allow_html=True)

    # --- CONTROL PANEL ---
    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 2])
        
        # 1. Override
        with c1:
            manual_on = st.checkbox("Manual Override", value=False)
            auto_regime = get_regime(market_data)
            active_regime = st.selectbox("Force Regime", ["GOLDILOCKS", "LIQUIDITY", "REFLATION", "NEUTRAL", "RISK OFF"], index=["GOLDILOCKS", "LIQUIDITY", "REFLATION", "NEUTRAL", "RISK OFF"].index(auto_regime)) if manual_on else auto_regime
        
        # 2. Timeframe
        with c2:
            time_view = st.radio("Quadrant View", ["Tactical (Daily)", "Structural (Weekly)"], horizontal=True)
            
        # 3. Status Display
        with c3:
            color_map = {"GOLDILOCKS": "#22c55e", "LIQUIDITY": "#a855f7", "REFLATION": "#f59e0b", "NEUTRAL": "#6b7280", "RISK OFF": "#ef4444"}
            rc = color_map[active_regime]
            st.markdown(f"""<div style="text-align:right;"><span style="font-size:12px; color:#888;">ACTIVE REGIME</span><br><span style="font-size:24px; font-weight:bold; color:{rc};">{active_regime}</span></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TABS ---
    t1, t2, t3, t4 = st.tabs(["🚀 Mission Control", "📊 Market Pulse", "🕸️ Macro Machine", "📖 Strategy Playbook"])

    # === TAB 1: MISSION CONTROL ===
    with t1:
        # SPX REACTOR INPUTS (Telemetry)
        st.markdown("##### 🎛️ SPX Income Reactor Telemetry")
        with st.expander("Input TradingView Data (IV Rank / Skew / ADX)", expanded=True):
            tc1, tc2, tc3 = st.columns(3)
            iv_rank = tc1.slider("IV Rank", 0, 100, 45)
            skew_rank = tc2.slider("Skew Rank", 0, 100, 50)
            adx_val = tc3.slider("ADX (Trend)", 0, 60, 20)
        
        # LOGIC ENGINE
        st.divider()
        
        # 1. Sizing & Logic
        risk_pct = "1.0%"
        focus = "Longs"
        if active_regime == "RISK OFF": risk_pct = "0.5%"; focus = "Cash/Hedge"
        elif active_regime == "GOLDILOCKS": risk_pct = "1.5%"; focus = "Aggressive Longs"
        elif active_regime == "NEUTRAL": risk_pct = "Inc"; focus = "Income"
        
        mc1, mc2 = st.columns([1, 2])
        
        with mc1:
            st.markdown(f"""
            <div class="strat-card">
                <div class="strat-header" style="color: {rc};">COMMAND CENTER</div>
                <div class="strat-sub">RISK ALLOCATION</div>
                <div class="strat-val" style="color: {rc}; font-size: 20px;">{risk_pct} per trade</div>
                <div class="strat-sub">FOCUS</div>
                <div class="strat-val">{focus}</div>
                <div class="strat-sub">VETO STATUS</div>
                <div class="strat-val">
                    HYG: {market_data['HYG']['change']:+.2f}% <br>
                    VIX: {market_data['VIX']['change']:+.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Target Acquisition
            targets = {
                "GOLDILOCKS": ("Tech, Semis", "Defensives"),
                "LIQUIDITY": ("Crypto, Nasdaq", "Cash"),
                "REFLATION": ("Energy, Banks", "Tech, Bonds"),
                "NEUTRAL": ("Income Plays", "Directional"),
                "RISK OFF": ("Cash, Volatility", "Everything Else")
            }
            longs, shorts = targets[active_regime]
            st.success(f"**TARGET:** {longs}")
            st.error(f"**AVOID:** {shorts}")

        # 2. TACTICAL EXECUTION (Reactor Logic)
        with mc2:
            st.subheader("⚔️ Tactical Execution")
            view_mode = st.radio("Asset Class", ["INDEX (SPX)", "STOCKS"], horizontal=True, label_visibility="collapsed")
            
            # Logic Tree
            strat_name, strat_dte, strat_strikes, strat_logic = "", "", "", ""
            
            if view_mode == "INDEX (SPX)":
                # SPX REACTOR LOGIC (Matches HTML v10.0)
                if active_regime == "RISK OFF":
                    strat_name = "A14 Put BWB"; strat_dte = "14 DTE Entry / 7 DTE Exit"; strat_strikes = "ATM / -40 / -60 (Skip)"; strat_logic = "Crash catcher. Zero upside risk."
                elif active_regime == "NEUTRAL":
                    strat_name = "TimeEdge Calendar"; strat_dte = "15 DTE / 7 DTE"; strat_strikes = "ATM Puts"; strat_logic = "Theta play. Low Vol environment."
                elif iv_rank > 50: # High Vol
                    if skew_rank > 80:
                        strat_name = "Put BWB (Skew Play)"; strat_dte = "21-30 DTE"; strat_strikes = "OTM Puts"; strat_logic = "High skew favors OTM butterflies."
                    else:
                        strat_name = "Iron Condor"; strat_dte = "30-45 DTE"; strat_strikes = "15 Delta Wings"; strat_logic = "Classic volatility crush."
                else: # Low Vol + Trend
                    if adx_val > 25:
                        strat_name = "Directional Diagonal"; strat_dte = "Front 17 / Back 31"; strat_strikes = "Long 70D / Short 30D"; strat_logic = "Trend replacement."
                    elif active_regime == "LIQUIDITY":
                        strat_name = "Flyagonal"; strat_dte = "7-14 DTE"; strat_strikes = "Call BWB + Put Diag"; strat_logic = "Capture drift."
                    else:
                        strat_name = "Double Diagonal"; strat_dte = "Front 17 / Back 31"; strat_strikes = "OTM"; strat_logic = "Low vol expansion play."
            else:
                # STOCK LOGIC
                if active_regime == "GOLDILOCKS":
                    strat_name = "Call Debit Spread"; strat_dte = "45-60 DTE"; strat_strikes = "60D / 30D"; strat_logic = "Directional efficiency."
                elif active_regime == "LIQUIDITY":
                    strat_name = "Zebra / Long Call"; strat_dte = "60+ DTE"; strat_strikes = "70D"; strat_logic = "Aggressive delta."
                elif active_regime == "NEUTRAL":
                    strat_name = "Iron Condor"; strat_dte = "30-45 DTE"; strat_strikes = "20 Delta"; strat_logic = "Range capture. Check earnings."
                elif active_regime == "RISK OFF":
                    strat_name = "Put Spread"; strat_dte = "60-90 DTE"; strat_strikes = "OTM"; strat_logic = "Hedge/Short."
                elif active_regime == "REFLATION":
                    strat_name = "Call Spread (Cyclicals)"; strat_dte = "45 DTE"; strat_strikes = "ATM"; strat_logic = "Rotation play."

            st.markdown(f"""
            <div class="strat-card" style="border-color: {rc};">
                <div class="strat-header" style="color: {rc};">{strat_name}</div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                    <div><div class="strat-sub">DTE</div><div class="strat-val">{strat_dte}</div></div>
                    <div><div class="strat-sub">STRIKES</div><div class="strat-val">{strat_strikes}</div></div>
                </div>
                <div style="background:#111; padding:10px; border-radius:6px; font-size:13px; color:#ccc;">
                    <strong>LOGIC:</strong> {strat_logic}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # SCREENER CRITERIA
            st.markdown("**🔍 Screener Logic:**")
            if active_regime == "GOLDILOCKS": st.code("Price > SMA50 | RSI 50-70 | EPS > 0")
            elif active_regime == "LIQUIDITY": st.code("ADX > 25 | RS > 80 | High Beta")
            elif active_regime == "NEUTRAL": st.code("ADX < 20 | BB Width > 0.1 | IVR > 30")
            elif active_regime == "RISK OFF": st.code("Price < SMA50 | Rel Vol > 2 | Debt/Eq > 2")

    # === TAB 2: MARKET PULSE ===
    with t2:
        mp1, mp2 = st.columns(2)
        with mp1:
            st.markdown("##### 🌊 Asset Flow (Sankey)")
            st.plotly_chart(create_sankey(market_data), use_container_width=True)
        with mp2:
            st.markdown("##### 🌡️ Heatmap")
            try:
                # Simple Sector Heatmap
                sec_data = {k: market_data[k]['change'] for k in ['TECH','SEMIS','BANKS','ENERGY','HOME','UTIL','HEALTH','MAT']}
                df_hm = pd.DataFrame(list(sec_data.items()), columns=['Sector', 'Change'])
                fig_hm = px.treemap(df_hm, path=['Sector'], values=[1]*len(df_hm), color='Change', color_continuous_scale=['red', 'black', 'green'], color_continuous_midpoint=0)
                st.plotly_chart(fig_hm, use_container_width=True)
            except: st.error("Heatmap Data Error")
            
        st.divider()
        st.markdown("##### 🎯 Momentum Quadrants (RRG)")
        st.info(f"Currently Viewing: {time_view}")
        rrg_data = get_quadrants(market_data, time_view)
        st.plotly_chart(create_rrg(rrg_data, "Sector & Asset Rotation"), use_container_width=True)

    # === TAB 3: MACRO MACHINE ===
    with t3:
        st.markdown("### 🕸️ The Macro Transmission Mechanism")
        c_g, c_e = st.columns([3, 1])
        with c_g:
            st.plotly_chart(create_nexus_graph(market_data), use_container_width=True)
        with c_e:
            st.markdown("""
            **How to read:**
            
            1. **The Source (Rates/DXY):**
               If these Nodes are GREEN (Dropping), it feeds Liquidity.
               If RED (Rising), it restricts Liquidity.
               
            2. **The Flow:**
               Follow the lines. Rising Rates -> Hurts Tech/Gold.
               Rising Dollar -> Hurts EM/Oil.
               
            3. **The Destination:**
               Check Credit (HYG). If Credit is Broken (RED), the transmission is broken. Risk Off.
            """)

    # === TAB 4: PLAYBOOK ===
    with t4:
        st.markdown("### 📚 Strategy Reference Library")
        
        with st.expander("🟣 TIMEZONE (Short-Term Income)", expanded=False):
            st.markdown("""
            **Concept:** High probability income strategy for RUT.
            
            **Setup:**
            * **Entry:** Thursday Afternoon (15 DTE).
            * **Structure:** Put Credit Spread (Income) + Put Calendar (Hedge).
            * **Ratio:** 2x PCS / 2x Calendar.
            
            **Rules:**
            * **Hard Stop:** Exit by 7 DTE (gamma risk).
            * **Target:** 5-7% Return on Capital.
            * **Adjustment:** If market drops, roll calendar down.
            """)
            
        with st.expander("🔵 TIMEEDGE (Double Calendar)", expanded=False):
            st.markdown("""
            **Concept:** Pure Theta play utilizing decay differential.
            
            **Setup (SPX):**
            * **Structure:** Double Calendar or Put Calendar.
            * **Entry:** Thursday @ 3:30 PM (15 DTE Front / 22 DTE Back).
            * **Strikes:** ATM.
            
            **Constraint:**
            * Back month IV cannot be >1pt higher than Front month.
            * Avoid Earnings.
            """)
            
        with st.expander("🌊 FLYAGONAL (Liquidity/Drift)", expanded=False):
            st.markdown("""
            **Concept:** Hybrid directional trade merging Call BWB (Upside) + Put Diagonal (Downside).
            
            **Setup (SPX):**
            * **Call Side:** 1 Long (ATM+10) / 2 Short (ATM+50) / 1 Long (ATM+60).
            * **Put Side:** Sell 1 Put (ATM-30) / Buy 1 Put (ATM-40) in later expiry.
            * **Entry:** 7-10 DTE.
            """)
            
        with st.expander("🔴 A14 (Risk Off Hedge)", expanded=False):
            st.markdown("""
            **Concept:** Crash Catcher.
            
            **Setup:**
            * **Structure:** Put BWB (Broken Wing).
            * **Entry:** Friday Morning (14 DTE).
            * **Strikes:** Long ATM / Short -40pts / Skip / Long -60pts.
            
            **Why:** Zero upside risk if filled for credit. Catches the crash.
            """)

if __name__ == "__main__":
    main()
