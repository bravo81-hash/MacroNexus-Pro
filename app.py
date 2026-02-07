import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import graphviz

# --- CONFIGURATION ---
st.set_page_config(
    page_title="MacroNexus Pro Terminal",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .strat-tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; margin-right: 5px; }
    
    /* Quadrant Colors */
    .quad-leading { border-left: 3px solid #22c55e; }
    .quad-improving { border-left: 3px solid #3b82f6; }
    .quad-weakening { border-left: 3px solid #f59e0b; }
    .quad-lagging { border-left: 3px solid #ef4444; }
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
    'TIP': 'TIP',          # TIPS (Real Yield Proxy)
    
    # COMMODITIES
    'GOLD': 'GLD', 'SILVER': 'SLV', 'OIL': 'USO',
    'NATGAS': 'UNG', 'COPPER': 'CPER', 'AG': 'DBA',
    
    # INDICES
    'SPY': 'SPY', 'QQQ': 'QQQ', 'IWM': 'IWM',
    'EEM': 'EEM', 'FXI': 'FXI', 'EWJ': 'EWJ',
    
    # SECTORS
    'TECH': 'XLK', 'SEMIS': 'SMH', 'BANKS': 'XLF',
    'ENERGY': 'XLE', 'HOME': 'XHB', 'UTIL': 'XLU',
    'STAPLES': 'XLP', 'DISC': 'XLY', 'IND': 'XLI',
    'HEALTH': 'XLV', 'MAT': 'XLB', 'COMM': 'XLC', 'RE': 'XLRE',
    
    # CRYPTO & FOREX
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD',
    'EURO': 'FXE', 'YEN': 'FXY'
}

FALLBACKS = {'DXY': 'UUP', 'VIX': 'VIXY'}

# --- 2. EXPERT KNOWLEDGE BASE (v10.0 Logic) ---
STRATEGY_DB = {
    "GOLDILOCKS": {
        "desc": "Trend + Low Vol. Favor Directional Longs.",
        "risk": "1.5%",
        "color": "#22c55e",
        "index": {
            "strategy": "Directional Diagonal",
            "dte": "Front 17 / Back 31 DTE",
            "strikes": "Long: >70 Delta (Deep ITM) | Short: ~30 Delta (OTM)",
            "logic": "Stock replacement. Deep ITM long mimics stock ownership; short OTM call reduces cost basis and harvests Theta in low vol."
        },
        "stock": {
            "strategy": "Call Debit Spread",
            "dte": "45-60 DTE",
            "strikes": "Long: 60 Delta | Short: 30 Delta",
            "logic": "Reduces cost of directional trade. 45-60 DTE gives time for the trend to play out without rapid decay.",
            "screener": "Price > SMA50 | RSI 50-70 | EPS Gr > 0% | No Earnings < 14d"
        }
    },
    "LIQUIDITY": {
        "desc": "Aggressive Trend (Drift). Favor Beta.",
        "risk": "1.0%",
        "color": "#a855f7",
        "index": {
            "strategy": "Flyagonal (Call BWB + Put Diag)",
            "dte": "Entry: 7-14 DTE",
            "strikes": "Call BWB: ATM+10/+50/+60 | Put Diag: -30/-40",
            "logic": "Pure Delta/Gamma play. Call BWB captures the drift (profit tent), Put Diagonal anchors downside."
        },
        "stock": {
            "strategy": "Long Call / Zebra",
            "dte": "60-90 DTE",
            "strikes": "Zebra: Buy 2x 70D / Sell 1x 50D (Zero Extrinsic)",
            "logic": "Stock replacement with zero time decay. Best for aggressive liquidity pumps where you want 100 delta exposure.",
            "screener": "ADX > 25 | Relative Strength | High Beta | Crypto Proxies"
        }
    },
    "REFLATION": {
        "desc": "Cyclical Rotation. Yields Rising.",
        "risk": "1.0%",
        "color": "#f59e0b",
        "index": {
            "strategy": "Directional Diagonal (IWM Focus)",
            "dte": "Front 17 / Back 31 DTE",
            "strikes": "Long 70D / Short 30D",
            "logic": "Reflation favors small caps (IWM). Use Diagonals to leverage the rotation out of Tech into Cyclicals."
        },
        "stock": {
            "strategy": "Call Debit Spread",
            "dte": "45-60 DTE",
            "strikes": "Long: 55 Delta | Short: 25 Delta",
            "logic": "Targeting Energy/Banks. Slightly wider strikes to capture volatility expansion in these sectors.",
            "screener": "Focus: Energy / Banks / Ind | Div Yield > 2% | PEG < 1.5"
        }
    },
    "NEUTRAL": {
        "desc": "Chop / Range. Income Mode.",
        "risk": "0.0% (Stock) / Income Size",
        "color": "#6b7280",
        "index": {
            "strategy": "TimeEdge Double Calendar",
            "dte": "Entry: 15 DTE / Exit: 7 DTE",
            "strikes": "Put Cal: Sell 15 DTE / Buy 22 DTE (ATM)",
            "logic": "TimeEdge specific: Maximizes Theta decay curve (15-7 DTE). Avoids earnings. Requires Low Vol (<20 ADX)."
        },
        "stock": {
            "strategy": "Iron Condor / Calendar",
            "dte": "30-45 DTE",
            "strikes": "Short P: 20 Delta / Short C: 20 Delta | Wings: 10pts Wide",
            "logic": "Classic range capture. 20 Delta is the sweet spot for single stocks. Mandatory Earnings Filter.",
            "screener": "ADX < 20 | BB Width > 0.10 | IV Rank > 30 | No Earnings < 21d"
        }
    },
    "RISK OFF": {
        "desc": "High Vol / Credit Stress. Hedge or Short.",
        "risk": "0.5%",
        "color": "#ef4444",
        "index": {
            "strategy": "A14 Put Broken Wing Butterfly",
            "dte": "Entry: 14 DTE (Fri) / Exit: 7 DTE",
            "strikes": "Long ATM / Short -40pts / Long -60pts (Skip Strike)",
            "logic": "A14 Strategy: Designed to catch the crash. Zero upside risk. Hard exit at 7 DTE to avoid Gamma."
        },
        "stock": {
            "strategy": "Put Debit Spread",
            "dte": "60-90 DTE",
            "strikes": "Long: 40 Delta (OTM) | Short: 15 Delta",
            "logic": "Buying OTM puts is cheaper. We go longer duration (60-90) to avoid getting crushed by IV contraction.",
            "screener": "Price < SMA50 | High Relative Vol | Beta > 1.5 | Debt/Eq > 2.0"
        }
    }
}

# --- 3. DATA FETCHING ---
@st.cache_data(ttl=300)
def fetch_live_data():
    """Robust data fetching with holiday gap fix."""
    data_map = {}
    
    for key, symbol in TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if hist.empty and key in FALLBACKS:
                symbol = FALLBACKS[key]
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="3mo")
            
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

# --- 4. LOGIC ENGINE ---
def analyze_market_auto(data):
    if not data: return None
    
    def get_c(k): return data.get(k, {}).get('change', 0)
    
    hyg, vix = get_c('HYG'), get_c('VIX')
    oil, cop = get_c('OIL'), get_c('COPPER')
    us10y, dxy = get_c('US10Y'), get_c('DXY')
    btc, banks = get_c('BTC'), get_c('BANKS')

    regime = "NEUTRAL"
    
    # Logic matching v10.0 HTML
    if hyg < -0.5 or vix > 5.0:
        regime = "RISK OFF"
    elif (oil > 2.0 or cop > 2.0) and us10y > 5.0 and banks > 0:
        regime = "REFLATION"
    elif dxy < -0.4 and btc > 3.0:
        regime = "LIQUIDITY"
    elif vix < 0 and abs(us10y) < 5.0 and hyg > -0.1:
        regime = "GOLDILOCKS"
        
    return regime, data

# --- 5. VISUALIZATION HELPERS ---
def create_nexus_graph(market_data):
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
    edges = [('US10Y','QQQ'), ('US10Y','GOLD'), ('US10Y','XHB'), ('DXY','GOLD'), ('DXY','OIL'), ('DXY','EEM'), 
             ('HYG','SPY'), ('HYG','IWM'), ('HYG','XLF'), ('QQQ','BTC'), ('QQQ','SMH'), ('COPPER','US10Y'), 
             ('OIL','XLE'), ('VIX','SPY')]
    
    edge_x, edge_y = [], []
    for u, v in edges:
        if u in nodes and v in nodes:
            x0, y0 = nodes[u]['pos']; x1, y1 = nodes[v]['pos']
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for key, info in nodes.items():
        x, y = info['pos']; node_x.append(x); node_y.append(y)
        d = market_data.get(key, {}); chg = d.get('change', 0)
        col = '#22c55e' if chg > 0 else '#ef4444'
        if chg == 0: col = '#6b7280'
        if key in ['US10Y', 'DXY', 'VIX']: col = '#ef4444' if chg > 0 else '#22c55e'
        node_color.append(col); node_size.append(45 if key in ['US10Y', 'DXY', 'HYG'] else 35)
        ticker = d.get('symbol', key); price = d.get('price', 0)
        fmt_chg = f"{chg:+.1f} bps" if key == 'US10Y' else f"{chg:+.2f}%"
        node_text.append(f"<b>{info['label']} ({ticker})</b><br>Price: {price:.2f}<br>Chg: {fmt_chg}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#4b5563'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[n.split('<br>')[0] for n in node_text], 
                             textposition="bottom center", hovertext=node_text, hoverinfo="text", 
                             marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')), 
                             textfont=dict(size=11, color='white')))
    fig.update_layout(showlegend=False, margin=dict(b=0,l=0,r=0,t=0), 
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]), 
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.0, 2.0]), 
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)
    return fig

def create_sankey_flow(market_data):
    # Simplified Logic for Sector Flow
    sectors = {k: market_data.get(k, {}).get('change', 0) for k in ['TECH', 'SEMIS', 'BANKS', 'ENERGY', 'HOME', 'UTIL', 'HEALTH', 'IND', 'MAT', 'COMM', 'DISC', 'STAPLES', 'RE']}
    df = pd.DataFrame(list(sectors.items()), columns=['id', 'val']).sort_values('val', ascending=False)
    
    winners = df.head(3)
    losers = df.tail(3).sort_values('val', ascending=True)
    total_flow = winners['val'].sum() + abs(losers['val'].sum())
    
    labels = list(losers['id']) + list(winners['id'])
    sources, targets, values, colors = [], [], [], []
    
    for i, loser_row in losers.iterrows():
        idx_src = list(losers['id']).index(loser_row['id'])
        for j, winner_row in winners.iterrows():
            idx_tgt = len(losers) + list(winners['id']).index(winner_row['id'])
            weight = (abs(loser_row['val']) * winner_row['val']) / total_flow * 10
            sources.append(idx_src); targets.append(idx_tgt); values.append(weight); colors.append('rgba(75, 85, 99, 0.4)')

    node_colors = ['#ef4444'] * len(losers) + ['#22c55e'] * len(winners)
    fig = go.Figure(data=[go.Sankey(node = dict(pad = 15, thickness = 20, line = dict(color = "black", width = 0.5), label = labels, color = node_colors), link = dict(source = sources, target = targets, value = values, color = colors))])
    fig.update_layout(title_text="<b>Money Flow (Day)</b>", font=dict(size=12, color='white'), paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# --- 6. MAIN APP ---
def main():
    # --- SIDEBAR OVERRIDE ---
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("Execution Timing: 3:00 PM EST")
        
        st.divider()
        st.subheader("📡 Regime Override")
        manual_active = st.checkbox("Enable Manual Override", value=False, help="Force the system into a specific regime if Yahoo Finance data is delayed.")
        
        if manual_active:
            override_regime = st.selectbox("Force Regime To:", list(STRATEGY_DB.keys()))
            st.warning(f"SYSTEM FORCED: {override_regime}")
        else:
            st.caption("Status: Auto-Pilot (Live Data)")

    # --- DATA INIT ---
    with st.spinner("Connecting to Mission Control..."):
        market_data = fetch_live_data()
        auto_regime, _ = analyze_market_auto(market_data)
        
    # Determine Active Regime
    active_regime = override_regime if manual_active else auto_regime
    db = STRATEGY_DB.get(active_regime, STRATEGY_DB["NEUTRAL"])

    # --- HEADER ---
    st.markdown(f"### 🛰️ MacroNexus Pro: {active_regime}")
    
    # --- TILE ROW ---
    cols = st.columns(6)
    def tile(c, label, key):
        d = market_data.get(key, {})
        val = d.get('price', 0); chg = d.get('change', 0)
        color = "#ef4444" if chg < 0 else "#22c55e"
        if key in ['VIX', 'US10Y', 'DXY']: color = "#ef4444" if chg > 0 else "#22c55e"
        fmt_chg = f"{chg:+.1f} bps" if key == 'US10Y' else f"{chg:+.2f}%"
        c.markdown(f"""<div class="metric-container" style="border-left-color: {color};"><div class="metric-header"><span class="metric-label">{label}</span></div><div><span class="metric-val">{val:.2f}</span><span class="metric-chg" style="color: {color};">{fmt_chg}</span></div></div>""", unsafe_allow_html=True)

    tile(cols[0], "Credit", "HYG"); tile(cols[1], "Volatility", "VIX"); tile(cols[2], "10Y Yield", "US10Y")
    tile(cols[3], "Dollar", "DXY"); tile(cols[4], "Oil", "OIL"); tile(cols[5], "Bitcoin", "BTC")

    # --- MAIN TABS ---
    tab_mc, tab_db, tab_lib, tab_macro = st.tabs(["🚀 Mission Control", "📊 Market Pulse", "📖 Strategy Playbook", "🌐 Macro Network"])

    # --- TAB 1: MISSION CONTROL (v10.0 Logic) ---
    with tab_mc:
        # Phase 1: Context
        c1, c2 = st.columns([1, 2])
        with c1:
            bg_col = db['color']
            st.markdown(f"""
            <div class="regime-badge" style="border-color: {bg_col}; background: {bg_col}11;">
                <h2 style="color: {bg_col}; margin:0;">{active_regime}</h2>
                <p style="color: #ccc; font-size: 14px; margin: 5px 0 0 0;">{db['desc']}</p>
                <div style="margin-top: 10px; font-weight: bold;">RISK ALLOCATION: {db['risk']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Veto Check Visuals
            hyg_chg = market_data.get('HYG', {}).get('change', 0)
            vix_chg = market_data.get('VIX', {}).get('change', 0)
            
            st.markdown("##### 🛡️ Veto Check")
            col_v1, col_v2 = st.columns(2)
            col_v1.metric("HYG (Credit)", f"{hyg_chg:+.2f}%", delta_color="normal")
            col_v2.metric("VIX (Fear)", f"{vix_chg:+.2f}%", delta_color="inverse")
            
            if hyg_chg < -0.5: st.error("⛔ CREDIT STRESS: No Longs")
            elif vix_chg > 5.0: st.error("⛔ VIX SPIKE: Hedge/Cash")
            else: st.success("✅ SYSTEMS NORMAL")

        with c2:
            st.markdown("##### 🎯 Target Acquisition")
            mode_toggle = st.radio("Tactical Mode", ["INDEX (SPX/RUT)", "STOCKS (Equity)"], horizontal=True, label_visibility="collapsed")
            
            strat = db['index'] if "INDEX" in mode_toggle else db['stock']
            
            st.markdown(f"""
            <div class="strat-card">
                <div class="strat-header" style="color: {bg_col};">{strat['strategy']}</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
                    <div>
                        <div style="font-size: 10px; color: #888; text-transform: uppercase;">DTE / Expiration</div>
                        <div style="font-weight: bold; font-family: monospace;">{strat['dte']}</div>
                    </div>
                    <div>
                        <div style="font-size: 10px; color: #888; text-transform: uppercase;">Structure / Strikes</div>
                        <div style="font-weight: bold; font-family: monospace;">{strat['strikes']}</div>
                    </div>
                </div>
                <div style="background: #111; padding: 10px; border-radius: 6px; font-size: 13px; color: #ccc;">
                    <strong>💡 The Logic:</strong> {strat['logic']}
                </div>
                {"<div style='margin-top: 10px; border-top: 1px solid #444; padding-top: 5px; font-family: monospace; font-size: 12px; color: " + bg_col + ";'>🔍 SCREENER: " + strat.get('screener', 'N/A') + "</div>" if 'screener' in strat else ""}
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        
        # Phase 5: Trade Architect Logic (Dynamic)
        st.markdown("##### 🏗️ Trade Architect (Execution Protocol)")
        c_a, c_b, c_c = st.columns(3)
        
        # Directional Card
        opacity_dir = "1.0" if active_regime in ["GOLDILOCKS", "LIQUIDITY", "REFLATION", "RISK OFF"] else "0.4"
        dir_title = "BEARISH (Puts)" if active_regime == "RISK OFF" else "BULLISH (Calls)"
        with c_a:
            st.markdown(f"""<div class="strat-card" style="opacity: {opacity_dir}; text-align: center;">
                <div style="color: #f59e0b; font-weight: bold;">DIRECTIONAL (ADX > 30)</div>
                <div style="font-size: 12px; margin-top: 5px;">{dir_title}</div>
                <div style="margin-top: 5px; font-size: 11px; color: #888;">Spreads / Stock</div>
            </div>""", unsafe_allow_html=True)

        # Drift Card
        opacity_drift = "1.0" if active_regime == "LIQUIDITY" else "0.4"
        with c_b:
            st.markdown(f"""<div class="strat-card" style="opacity: {opacity_drift}; text-align: center;">
                <div style="color: #06b6d4; font-weight: bold;">DRIFT (ADX 20-30)</div>
                <div style="font-size: 12px; margin-top: 5px;">FLYAGONAL</div>
                <div style="margin-top: 5px; font-size: 11px; color: #888;">Diagonals</div>
            </div>""", unsafe_allow_html=True)

        # Range Card
        opacity_range = "1.0" if active_regime in ["NEUTRAL", "GOLDILOCKS"] else "0.4"
        range_strat = "TimeEdge Calendar" if active_regime == "NEUTRAL" else "Diagonals"
        with c_c:
            st.markdown(f"""<div class="strat-card" style="opacity: {opacity_range}; text-align: center;">
                <div style="color: #22c55e; font-weight: bold;">RANGE (ADX < 20)</div>
                <div style="font-size: 12px; margin-top: 5px;">{range_strat}</div>
                <div style="margin-top: 5px; font-size: 11px; color: #888;">Theta Plays</div>
            </div>""", unsafe_allow_html=True)

    # --- TAB 2: MARKET PULSE (Original Visuals) ---
    with tab_db:
        col_sec, col_macro = st.columns([1, 1])
        with col_sec:
            st.plotly_chart(create_sankey_flow(market_data), use_container_width=True)
        with col_macro:
            # Heatmap of sectors
            try:
                sectors = {k: market_data.get(k, {}).get('change', 0) for k in ['TECH', 'SEMIS', 'BANKS', 'ENERGY', 'HOME', 'UTIL', 'HEALTH', 'IND', 'MAT']}
                df_sec = pd.DataFrame(list(sectors.items()), columns=['Sector', 'Change']).sort_values('Change', ascending=False)
                fig_bar = px.bar(df_sec, x='Change', y='Sector', orientation='h', color='Change', color_continuous_scale=['#ef4444', '#1e2127', '#22c55e'])
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=350)
                st.plotly_chart(fig_bar, use_container_width=True)
            except: st.error("Data Error")

    # --- TAB 3: STRATEGY PLAYBOOK (Deep Dive) ---
    with tab_lib:
        st.markdown("### 📚 The Arsenal")
        
        with st.expander("🟣 THE FLYAGONAL (Liquidity / Drift Strategy)"):
            st.markdown("""
            **Concept:** A hybrid income/directional trade merging a Call Broken Wing Butterfly (Upside) with a Put Diagonal (Downside).
            
            **Setup (SPX Example):**
            * **Call Side (Income):** 1 Long (ATM+10) / 2 Short (ATM+50) / 1 Long (ATM+60). *Broken Wing.*
            * **Put Side (Anchor):** Sell 1 Put (ATM-30) / Buy 1 Put (ATM-40) in a later expiry (Diagonal).
            * **Timing:** Entry 7-10 DTE.
            
            **Management:**
            * **Flash Win:** >4% Profit in 1-2 days -> CLOSE.
            * **Defense:** If market drops >10%, add Put Calendar or roll shorts down.
            """)
            
        with st.expander("🟡 TIMEEDGE (Neutral / Range Strategy)"):
            st.markdown("""
            **Concept:** Pure Theta play utilizing the decay curve differential between 15 DTE and 7 DTE.
            
            **Setup (SPX/RUT):**
            * **Structure:** Double Calendar or Put Calendar.
            * **Entry:** Thursday @ 3:30 PM (15 DTE Front / 22 DTE Back).
            * **Strikes:** At-The-Money (ATM).
            
            **Rules:**
            * **Constraint:** Volatility of Back Month must NOT be >1pt higher than Front Month.
            * **Hard Stop:** Exit at 7 DTE. Never hold into Gamma week.
            * **Target:** 10% on Margin.
            """)
            
        with st.expander("🔴 A14 (Risk Off / Hedge Strategy)"):
            st.markdown("""
            **Concept:** A "Crash Catcher" structure with zero upside risk if filled for a credit.
            
            **Setup (SPX):**
            * **Structure:** Put Broken Wing Butterfly (BWB).
            * **Entry:** Friday Morning (14 DTE).
            * **Strikes:** 1 Long ATM / 2 Short OTM (-40pts) / Skip Strike / 1 Long OTM (-60pts).
            
            **Logic:**
            * Creates a "Profit Tent" to the downside.
            * If market rallies, trade expires worthless (or small credit).
            * If market crashes, it flies into the tent.
            """)

    # --- TAB 4: MACRO NETWORK ---
    with tab_macro:
        st.plotly_chart(create_nexus_graph(market_data), use_container_width=True)
        st.info("Visualizes how Fed Policy (Yields/DXY) flows downstream to specific sectors.")

if __name__ == "__main__":
    main()
