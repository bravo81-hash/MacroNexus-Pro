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
    .metric-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; }
    .metric-label { font-size: 10px; color: #9ca3af; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
    .metric-ticker { font-size: 9px; color: #6b7280; font-family: monospace; background: #262730; padding: 1px 4px; border-radius: 3px; }
    .metric-val { font-size: 18px; font-weight: bold; color: #f3f4f6; }
    .metric-chg { font-size: 12px; font-weight: bold; margin-left: 6px; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 20px; border-bottom: 1px solid #2e3039; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: 600; font-size: 14px; }
    .regime-badge { padding: 15px; border-radius: 8px; text-align: center; border: 1px solid; margin-bottom: 20px; background: #1e2127; }
    
    /* Rotation Quadrant Colors */
    .quad-leading { border-left: 3px solid #22c55e; padding-left: 10px; background: rgba(34, 197, 94, 0.1); border-radius: 0 4px 4px 0; }
    .quad-improving { border-left: 3px solid #3b82f6; padding-left: 10px; background: rgba(59, 130, 246, 0.1); border-radius: 0 4px 4px 0; }
    .quad-weakening { border-left: 3px solid #f59e0b; padding-left: 10px; background: rgba(245, 158, 11, 0.1); border-radius: 0 4px 4px 0; }
    .quad-lagging { border-left: 3px solid #ef4444; padding-left: 10px; background: rgba(239, 68, 68, 0.1); border-radius: 0 4px 4px 0; }
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

# Fallbacks for cloud environments
FALLBACKS = {'DXY': 'UUP', 'VIX': 'VIXY'}

@st.cache_data(ttl=300)
def fetch_live_data():
    """Robust data fetching with holiday gap fix."""
    data_map = {}
    failed_tickers = []
    
    for key, symbol in TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            # Fetch 3mo days to ensure we have enough buffer for weekly/monthly calcs
            hist = ticker.history(period="3mo")
            
            # Try fallback if primary empty
            if hist.empty and key in FALLBACKS:
                symbol = FALLBACKS[key]
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="3mo")
            
            # CRITICAL FIX: Drop NaNs to fix "Monday/Holiday" bug
            hist_clean = hist['Close'].dropna()

            if not hist_clean.empty and len(hist_clean) >= 22:
                current = hist_clean.iloc[-1]
                prev = hist_clean.iloc[-2]         # 1 Day ago
                prev_week = hist_clean.iloc[-6]    # 5 Days ago
                prev_month = hist_clean.iloc[-21]  # 20 Days ago
                
                # Metric Calculation
                if key == 'US10Y':
                    # AUDIT FIX: Yahoo ^TNX already expresses changes in basis points.
                    # NO MULTIPLIER REQUIRED.
                    change = (current - prev)
                    change_w = (current - prev_week)
                    change_m = (current - prev_month)
                else:
                    # Standard Percent Change
                    change = ((current - prev) / prev) * 100
                    change_w = ((current - prev_week) / prev_week) * 100
                    change_m = ((current - prev_month) / prev_month) * 100
                
                data_map[key] = {
                    'price': current, 
                    'change': change, 
                    'change_w': change_w,
                    'change_m': change_m,
                    'symbol': symbol, 
                    'error': False
                }
            else:
                data_map[key] = {'price': 0.0, 'change': 0.0, 'change_w': 0.0, 'change_m': 0.0, 'symbol': symbol, 'error': True, 'msg': "No Data"}
                failed_tickers.append(key)
        except Exception as e:
            data_map[key] = {'price': 0.0, 'change': 0.0, 'change_w': 0.0, 'change_m': 0.0, 'symbol': symbol, 'error': True, 'msg': str(e)}
            failed_tickers.append(key)
            
    return data_map

# --- 2. LOGIC ENGINE ---
def analyze_market(data):
    if not data: return None
    
    # Check Critical Feeds
    critical_errors = []
    for k in ['HYG', 'VIX', 'US10Y']:
        if data.get(k, {}).get('error'):
            critical_errors.append(k)
            
    if critical_errors:
        return {
            'regime': 'DATA ERROR',
            'desc': f"Critical feeds down: {', '.join(critical_errors)}. Cannot analyze.",
            'color': '#ef4444',
            'longs': ['CASH'], 'shorts': [], 'alerts': ['⚠️ DATA INTEGRITY FAILURE - STAND DOWN']
        }

    def get_c(k): return data.get(k, {}).get('change', 0)
    
    hyg, vix = get_c('HYG'), get_c('VIX')
    oil, cop = get_c('OIL'), get_c('COPPER')
    us10y, dxy = get_c('US10Y'), get_c('DXY')
    btc, banks = get_c('BTC'), get_c('BANKS')

    regime = "NEUTRAL"
    desc = "No clear macro dominance. Follow price momentum."
    color_code = "#6b7280"
    longs, shorts, alerts = [], [], []

    # 1. RISK OFF (The Veto)
    # Thresholds: HYG < -0.5% (Tighter per audit) OR VIX > 5%
    if hyg < -0.5 or vix > 5.0:
        regime = "RISK OFF"
        desc = "Credit Stress or Volatility Spike. Cash is King."
        color_code = "#ef4444" # Red
        longs = ["Cash (UUP)", "Vol (VIX)"]
        shorts = ["Tech", "Crypto", "Small Caps", "High Yield"]
        alerts.append("⛔ CREDIT VETO: HYG Breaking Down. Stop all long risk.")

    # 2. REFLATION (Growth + Yields + Banks)
    # AUDIT FIX: US10Y threshold adjusted to 5.0 (represents 5 bps in Yahoo data)
    elif (oil > 2.0 or cop > 2.0) and us10y > 5.0 and banks > 0:
        regime = "REFLATION"
        desc = "Inflationary Growth. Real Assets outperform."
        color_code = "#f59e0b" # Orange
        longs = ["Energy (XLE)", "Banks (XLF)", "Industrials"]
        shorts = ["Bonds (TLT)", "Tech (Rate Sensitive)"]
        alerts.append("🔥 INFLATION PULSE: Rotate to Cyclicals.")

    # 3. LIQUIDITY PUMP (Risk On)
    # Tighter thresholds: DXY < -0.4%, BTC > 3.0%
    elif dxy < -0.4 and btc > 3.0:
        regime = "LIQUIDITY PUMP"
        desc = "Dollar weakness fueling high-beta assets."
        color_code = "#a855f7" # Purple
        longs = ["Bitcoin", "Nasdaq (QQQ)", "Semis (SMH)"]
        shorts = ["Dollar (DXY)", "Defensives"]
        alerts.append("🌊 LIQUIDITY ON: Green light for Beta.")

    # 4. GOLDILOCKS (Stability)
    # AUDIT FIX: US10Y threshold adjusted to abs < 5.0 (represents < 5 bps)
    elif vix < 0 and abs(us10y) < 5.0 and hyg > -0.1:
        regime = "GOLDILOCKS"
        desc = "Low vol, stable rates. Favorable for equities."
        color_code = "#22c55e" # Green
        longs = ["S&P 500", "Tech", "Quality Growth"]
        shorts = ["Volatility"]
        alerts.append("✅ STABLE: Buy Dips.")

    # 5. MOMENTUM FALLBACK (Smart Neutral)
    if not longs and regime == "NEUTRAL":
        asset_keys = ['SPY', 'QQQ', 'IWM', 'BTC', 'GOLD', 'OIL', 'COPPER', 'BANKS', 'ENERGY', 'SEMIS']
        assets = {k: get_c(k) for k in asset_keys}
        sorted_assets = sorted(assets.items(), key=lambda x: x[1], reverse=True)
        
        top_pick = sorted_assets[0]
        bottom_pick = sorted_assets[-1]
        
        # Bear Market Check: If even the best asset is flat/red, don't buy it
        if top_pick[1] > 0.3:
            longs = [f"{top_pick[0]} (+{top_pick[1]:.1f}%)", f"{sorted_assets[1][0]} (+{sorted_assets[1][1]:.1f}%)"]
        else:
            longs = ["Cash / Wait"]
            alerts.append("⚠️ NO MOMENTUM: Market is flat/down. Sit on hands.")
            
        shorts = [f"{bottom_pick[0]} ({bottom_pick[1]:.1f}%)"]

    return {
        'regime': regime, 'desc': desc, 'color': color_code,
        'longs': longs, 'shorts': shorts, 'alerts': alerts
    }

# --- 3. ROTATION ENGINE ---
def analyze_rotation_specifics(data, period='change'):
    def get_p(k): return data.get(k, {}).get(period, 0)
    
    # SECTORS ONLY
    sector_keys = ['TECH', 'SEMIS', 'BANKS', 'ENERGY', 'HOME', 'UTIL', 'HEALTH', 'IND', 'MAT', 'COMM', 'DISC', 'STAPLES', 'RE']
    sectors = [{'id': k, 'val': get_p(k)} for k in sector_keys]
    df_sec = pd.DataFrame(sectors).sort_values('val', ascending=False)
    
    # ASSETS ONLY
    asset_keys = ['GOLD', 'BTC', 'OIL', 'TLT', 'SPY', 'DXY', 'EEM']
    assets = [{'id': k, 'val': get_p(k)} for k in asset_keys]
    df_ast = pd.DataFrame(assets).sort_values('val', ascending=False)
    
    winner_sec = df_sec.iloc[0]['id']; loser_sec = df_sec.iloc[-1]['id']
    narrative = f"Rotating from **{loser_sec}** into **{winner_sec}**."
    implication = "Market is directionless."
    
    defensives = ['UTIL', 'STAPLES', 'HEALTH', 'RE']
    cyclicals = ['ENERGY', 'BANKS', 'IND', 'MAT']
    growth = ['TECH', 'SEMIS', 'DISC', 'COMM']
    
    if winner_sec in defensives: implication = "🛡️ **DEFENSIVE ROTATION:** Fear is rising. Capital hiding in safety."
    elif winner_sec in cyclicals: implication = "⚙️ **CYCLICAL ROTATION:** Betting on economic growth/inflation."
    elif winner_sec in growth: implication = "🚀 **GROWTH ROTATION:** Risk appetite is high."
        
    return df_sec, df_ast, narrative, implication

def analyze_quadrant_data(data, view_type):
    # Momentum Quadrant Logic
    points_sectors, points_macro = [], []
    
    sector_list = ['TECH', 'SEMIS', 'BANKS', 'ENERGY', 'HOME', 'UTIL', 'STAPLES', 'DISC', 'IND', 'HEALTH', 'MAT', 'COMM', 'RE', 'SPY', 'QQQ', 'IWM']
    macro_list = ['GOLD', 'SILVER', 'OIL', 'NATGAS', 'COPPER', 'BTC', 'ETH', 'SOL', 'EURO', 'YEN', 'TLT', 'DXY', 'EEM']
    
    for k in sector_list + macro_list:
        d = data.get(k, {})
        
        # FIX: Explicit separation of timeframes for RRG logic
        if view_type == 'Daily (Tactical)':
            # X = Weekly Trend (5d), Y = Daily Momentum (1d)
            x_val = d.get('change_w', 0)
            y_val = d.get('change', 0)
        else: # Weekly (Structural)
            # X = Monthly Trend (20d), Y = Weekly Momentum (5d)
            x_val = d.get('change_m', 0)
            y_val = d.get('change_w', 0)
        
        quad = 'IMPROVING'
        color = '#3b82f6'
        # Quadrant Logic: Leading (Both +), Lagging (Both -), Weakening (Trend + / Mom -), Improving (Trend - / Mom +)
        if x_val > 0 and y_val > 0: quad = 'LEADING'; color = '#22c55e'
        elif x_val > 0 and y_val < 0: quad = 'WEAKENING'; color = '#f59e0b'
        elif x_val < 0 and y_val < 0: quad = 'LAGGING'; color = '#ef4444'
        
        item = {'id': k, 'x': x_val, 'y': y_val, 'quad': quad, 'color': color}
        if k in sector_list: points_sectors.append(item)
        else: points_macro.append(item)
        
    return pd.DataFrame(points_sectors), pd.DataFrame(points_macro)

# --- 4. VISUALIZATION COMPONENTS ---
def create_sankey_flow(df, title_prefix):
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
    fig.update_layout(title_text=f"<b>{title_prefix} Flow</b>", font=dict(size=12, color='white'), paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def create_rrg_scatter(df, title, x_label, y_label, range_val=5):
    if df.empty: return go.Figure()
    fig = px.scatter(df, x='x', y='y', text='id', color='color', color_discrete_map="identity", title=f"<b>{title}</b>")
    fig.add_hline(y=0, line_dash="dot", line_color="gray"); fig.add_vline(x=0, line_dash="dot", line_color="gray")
    
    pos = range_val * 0.7
    fig.add_annotation(x=pos, y=pos, text="LEADING", showarrow=False, font=dict(color="#22c55e", size=12))
    fig.add_annotation(x=-pos, y=-pos, text="LAGGING", showarrow=False, font=dict(color="#ef4444", size=12))
    fig.add_annotation(x=-pos, y=pos, text="IMPROVING", showarrow=False, font=dict(color="#3b82f6", size=10))
    fig.add_annotation(x=pos, y=-pos, text="WEAKENING", showarrow=False, font=dict(color="#f59e0b", size=10))
    
    fig.update_traces(textposition='top center', marker=dict(size=14, line=dict(width=1, color='white')))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                      xaxis=dict(title=x_label, zeroline=False, range=[-range_val, range_val]), 
                      yaxis=dict(title=y_label, zeroline=False, range=[-range_val, range_val]),
                      showlegend=False, height=450)
    return fig

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
    edges = [('US10Y','QQQ'), ('US10Y','GOLD'), ('US10Y','XHB'), ('DXY','GOLD'), ('DXY','OIL'), ('DXY','EEM'), ('HYG','SPY'), ('HYG','IWM'), ('HYG','XLF'), ('QQQ','BTC'), ('QQQ','SMH'), ('COPPER','US10Y'), ('OIL','XLE'), ('VIX','SPY')]
    
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
        if d.get('error'): col = '#374151'
        node_color.append(col); node_size.append(45 if key in ['US10Y', 'DXY', 'HYG'] else 35)
        ticker = d.get('symbol', key); price = d.get('price', 0)
        
        # AUDIT FIX: Display Logic
        # Correctly display bps without double-counting
        if key == 'US10Y':
            fmt_chg = f"{chg:+.1f} bps"
        else:
            fmt_chg = f"{chg:+.2f}%"
            
        node_text.append(f"<b>{info['label']} ({ticker})</b><br>Price: {price:.2f}<br>Chg: {fmt_chg}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#4b5563'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[n.split('<br>')[0] for n in node_text], textposition="bottom center", hovertext=node_text, hoverinfo="text", marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')), textfont=dict(size=11, color='white')))
    fig.update_layout(showlegend=False, margin=dict(b=0,l=0,r=0,t=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.0, 2.0]), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)
    return fig

def create_heatmap_matrix():
    z_data = [[0.9, 0.9, 0.4, 0.6, 0.1, 0.2, 0.7], [-0.8, -0.6, -0.9, -0.3, 0.4, 0.6, -0.9], [-0.4, -0.5, -0.9, -0.9, -0.6, -0.1, -0.2], [0.8, 0.7, 0.1, 0.8, 0.6, 0.9, 0.8], [0.2, 0.3, 0.5, 0.9, 0.9, 0.8, 0.3]]
    x_labels = ['Tech', 'Crypto', 'Gold', 'EM', 'Energy', 'Banks', 'Housing']
    y_labels = ['Liquidity', 'Real Yields', 'Dollar', 'Credit', 'Growth']
    fig = px.imshow(z_data, x=x_labels, y=y_labels, color_continuous_scale=['#ef4444', '#1e2127', '#22c55e'], range_color=[-1, 1], aspect="auto")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=450)
    return fig

# --- 4. MAIN LAYOUT ---
def main():
    st.sidebar.warning("⚠️ **EXECUTION TIMING**: This tool uses daily data. For 3:00 PM EST execution, cross-reference with live price action. If Credit (HYG) is crashing intraday (> -1.5%), override to RISK OFF manually.")
    
    with st.spinner("Initializing MacroNexus Pro..."):
        market_data = fetch_live_data()
        analysis = analyze_market(market_data)

    st.markdown("### 📡 Market Pulse")
    if analysis and analysis['regime'] == 'DATA ERROR': st.error(analysis['desc'], icon="🚨")
    
    # AUDIT RECOMMENDATION: Sanity Check
    if market_data and 'US10Y' in market_data:
        us10y_chg = market_data['US10Y']['change']
        if abs(us10y_chg) > 25:
             st.warning(f"⚠️ US10Y Move Unusually Large ({us10y_chg:.1f} bps). Verify data source.")
    
    cols = st.columns(6)
    def tile(c, label, key):
        d = market_data.get(key, {})
        val = d.get('price', 0); chg = d.get('change', 0); err = d.get('error', False)
        if err: color = "#374151"; fmt_chg = "ERR"
        else:
            color = "#ef4444" if chg < 0 else "#22c55e"
            if key in ['VIX', 'US10Y', 'DXY']: color = "#ef4444" if chg > 0 else "#22c55e"
            
            # AUDIT FIX: Display
            fmt_chg = f"{chg:+.2f}%"
            if key == 'US10Y': fmt_chg = f"{chg:+.1f} bps"
            
        c.markdown(f"""<div class="metric-container" style="border-left-color: {color};"><div class="metric-header"><span class="metric-label">{label}</span></div><div><span class="metric-val">{val:.2f}</span><span class="metric-chg" style="color: {color};">{fmt_chg}</span></div></div>""", unsafe_allow_html=True)

    tile(cols[0], "Credit", "HYG"); tile(cols[1], "Volatility", "VIX"); tile(cols[2], "10Y Yield", "US10Y")
    tile(cols[3], "Dollar", "DXY"); tile(cols[4], "Oil", "OIL"); tile(cols[5], "Bitcoin", "BTC")

    if not analysis: return

    t1, t2, t3, t4, t5 = st.tabs(["🚀 Dashboard", "🔄 Money Rotation", "📊 Heatmap", "🌊 Liquidity", "📖 Playbook"])

    with t1:
        c_g, c_a = st.columns([2.5, 1])
        with c_g: st.plotly_chart(create_nexus_graph(market_data), use_container_width=True)
        with c_a:
            bg = analysis['color']
            st.markdown(f"""<div class="regime-badge" style="background-color: {bg}22; border-color: {bg};"><div style="color: {bg}; font-weight: bold; font-size: 20px; margin-bottom: 5px;">{analysis['regime']}</div><div style="font-size: 11px; color: #ccc;">{analysis['desc']}</div></div>""", unsafe_allow_html=True)
            st.success("**LONG**")
            for item in analysis['longs']: st.markdown(f"<small>• {item}</small>", unsafe_allow_html=True)
            st.error("**AVOID**")
            for item in analysis['shorts']: st.markdown(f"<small>• {item}</small>", unsafe_allow_html=True)
            if analysis['alerts']: st.error(analysis['alerts'][0], icon="🚨")

    with t2:
        c_ctrl, c_info = st.columns([1, 4])
        with c_ctrl:
            st.markdown("#### ⚙️ View")
            timeframe = st.radio("Period", ["Daily (Tactical)", "Weekly (Structural)"])
            
            if timeframe == "Daily (Tactical)":
                tf_key = 'change'
                x_lab, y_lab = "Weekly Trend (%)", "Daily Momentum (%)"
            else:
                tf_key = 'change_w'
                x_lab, y_lab = "Monthly Trend (%)", "Weekly Momentum (%)"
            
            zoom = st.slider("🔍 Zoom (%)", 1.0, 20.0, 5.0, 1.0)
            
            df_sec, df_ast, narrative, implication = analyze_rotation_specifics(market_data, tf_key)
            df_sec_q, df_macro_q = analyze_quadrant_data(market_data, timeframe)

            st.markdown("---")
            st.info(f"**Flow:** {narrative}")
            st.warning(f"**Theme:** {implication}")
            
        with c_info:
            col_sec, col_macro = st.columns(2)
            with col_sec:
                # AUDIT FIX: Renamed RRG to "Momentum Quadrant"
                st.markdown("##### 🏢 Sector Momentum Quadrant")
                st.plotly_chart(create_rrg_scatter(df_sec_q, "Sector Rotation", x_lab, y_lab, range_val=zoom), use_container_width=True)
            with col_macro:
                st.markdown("##### 🌍 Macro Asset Momentum Quadrant")
                st.plotly_chart(create_rrg_scatter(df_macro_q, "Asset Rotation", x_lab, y_lab, range_val=zoom*2), use_container_width=True)
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(create_sankey_flow(df_sec, "Sector"), use_container_width=True)
            with c2: st.plotly_chart(create_sankey_flow(df_ast, "Asset Class"), use_container_width=True)

    with t3: st.plotly_chart(create_heatmap_matrix(), use_container_width=True)

    with t4:
        st.markdown("### 🌊 The Macro Transmission Mechanism")
        st.info("Visualizes how Fed Policy flows downstream to specific sectors.")
        col_flow, col_expl = st.columns([2, 1])
        with col_flow:
            try:
                g = graphviz.Digraph()
                g.attr(rankdir='TB', bgcolor='transparent'); g.attr('node', shape='box', style='filled, rounded', fontname='Helvetica', fontcolor='white', penwidth='0'); g.attr('edge', color='#6b7280')
                g.node('FED', 'FED & TREASURY\n(Liquidity)', fillcolor='#4f46e5')
                g.node('RATE', 'YIELDS & RATES\n(Cost of Money)', fillcolor='#b91c1c'); g.node('USD', 'DOLLAR (DXY)\n(Collateral)', fillcolor='#1e3a8a'); g.node('CRED', 'CREDIT (HYG)\n(Risk Appetite)', fillcolor='#7e22ce')
                g.node('GROWTH', 'TECH / CRYPTO\n(Long Duration)', fillcolor='#1f2937'); g.node('REAL', 'COMMODITIES\n(Real Assets)', fillcolor='#1f2937'); g.node('EM', 'EMERGING MKTS\n(Dollar Sensitive)', fillcolor='#1f2937'); g.node('CYCL', 'BANKS / ENERGY\n(Growth Sensitive)', fillcolor='#1f2937')
                g.node('SEMIS', 'Semis (SMH)', fillcolor='#111827', fontsize='9'); g.node('HOUSING', 'Housing (XHB)', fillcolor='#111827', fontsize='9'); g.node('BTC', 'Bitcoin', fillcolor='#111827', fontsize='9'); g.node('IND', 'Industrials', fillcolor='#111827', fontsize='9')
                g.edge('FED','RATE'); g.edge('FED','USD'); g.edge('FED','CRED')
                g.edge('RATE','GROWTH'); g.edge('RATE','REAL'); g.edge('USD','EM')
                g.edge('CRED','GROWTH'); g.edge('CRED','CYCL')
                g.edge('GROWTH','SEMIS', style='dashed'); g.edge('GROWTH','BTC', style='dashed'); g.edge('CYCL','IND', style='dashed'); g.edge('RATE', 'HOUSING', style='dashed')
                st.graphviz_chart(g, use_container_width=True)
            except: st.warning("Graphviz missing.")
        with col_expl:
            with st.expander("1. The Source (Liquidity)", expanded=True): st.markdown("* **Liquidity (WALCL):** When Fed buys assets (QE) or Treasury spends cash (TGA), liquidity rises. Pumps **Bitcoin** & **Tech**.")
            with st.expander("2. The Transmission (Cost)", expanded=True): st.markdown("* **Real Yields:** If rates > inflation, kills valuations. **Gold** & **Tech** drop.\n* **Dollar:** Strong DXY wrecks **EM** & **Commodities**.")
            with st.expander("3. The Destination (Risk)", expanded=True): st.markdown("* **Credit (HYG):** If companies borrow cheaply, buy **Stocks**. If Credit breaks, SELL.\n* **Cyclicals:** If Growth real, buy **Energy** & **Banks**.")

    with t5:
        st.markdown("""
# 📖 MacroNexus Pro: Daily Trader's Playbook

This guide explains how to use the interactive map as a decision-support engine.

## ⏰ The 5-Minute Morning Routine

Before you look at a single stock chart, open the MacroNexus and perform this "Health Check."

### 1. Diagnose the "Plumbing" (The Veto Check)

**Goal:** Determine if it is safe to take risk today.

* **Click on `HYG` (High Yield Credit)**

  * *Question:* Is HYG stable or rising?

  * *Logic:* HYG measures corporate stress.

  * *Decision:* If HYG is crashing, **DO NOT** buy the dip in Stocks (`SPY`, `IWM`). The rally is likely a trap.

* **Click on `RealYields` (TIPS)**

  * *Question:* Are Real Yields spiking?

  * *Logic:* High real yields kill "duration" assets (Gold, Tech, Crypto).

  * *Decision:* If Real Yields are surging, **DO NOT** go long Gold or Nasdaq today.

### 2. Identify the Regime (The Tailwind Check)

**Goal:** Align your trades with the current wind direction.

Use the Regime Buttons (Keys `1`-`4`) to see what is currently favored:

| **If Market Feels Like...** | **Select Regime** | **Actionable Strategy** | 
| :--- | :--- | :--- |
| **"Bad news is good news"** | **LIQUIDITY** | **Focus:** Crypto (`BTC`), Tech (`QQQ`).  **Ignore:** Value stocks, Commodities. | 
| **"Everything is selling off"** | **RISK-OFF** | **Focus:** Cash (`DXY`), Volatility (`VIX`).  **Avoid:** Small Caps (`Russell`), Emerging Markets. | 
| **"Growth is booming"** | **GOLDILOCKS** | **Focus:** Everything works, but `Copper` and `Semis` lead.  **Avoid:** Defensive plays (Utilities, Bonds). | 
| **"Prices up, Growth down"** | **REFLATION** | **Focus:** Energy (`XLE`), Banks (`XLF`).  **Avoid:** Tech (`QQQ`) - it hates inflation. | 

## 🚦 "What To Do" vs. "What NOT To Do"

The tool is best used to filter your ideas. Here are specific examples:

### Scenario A: You want to buy NVIDIA or Tech (QQQ)

1. **Check `US10Y` & `RealYields`:**

   * *Tool View:* Click `RealYields`. Follow the thick red line to `Nasdaq`.

   * *Verdict:* If the line source (Yields) is UP, the target (Nasdaq) usually goes DOWN.

   * *Action:* **WAIT.** Don't fight the Fed.

### Scenario B: You want to buy the dip in Crypto (BTC)

1. **Check `Liquidity` & `TGA`:**

   * *Tool View:* Click `Liquidity`. Follow the green line to `Bitcoin`.

   * *Verdict:* Is the Fed draining liquidity (QT)? Or is the TGA refilling?

   * *Action:* If Liquidity is dropping, Crypto has no fuel. **NO TRADE.**

### Scenario C: You want to trade a "China Reopening" (FXI/Copper)

1. **Check `DXY` (Dollar):**

   * *Tool View:* Click `DXY`. Look at the red line to `EmgMkts` and `China`.

   * *Verdict:* A strong dollar crushes emerging markets (because of dollar-denominated debt).

   * *Action:* Only buy China if the DXY is weakening.

## ⚡ How to Spot Opportunity (Divergences)

The biggest trades happen when the market "breaks" a correlation temporarily. Use the tool to spot these:

* **The "Coil" Setup:**

  * If **Copper** rips higher (Growth signal)...

  * But **Oil** and **Rates** haven't moved yet...

  * *Trade:* The market is lagging. Look for **Energy (`XLE`)** or **Industrials** to play catch-up.

* **The "Fakeout" Setup:**

  * If **S&P 500** makes a new high...

  * But **HYG (Credit)** makes a lower high...

  * *Trade:* This is a bearish divergence. Credit isn't confirming the move. **Short the S&P 500.**

## 🛠 Execution Checklist (3:00 PM EST)

1. **Open Tool.**

2. **Press `2` (Risk-Off Mode).**

   * Are `DXY` and `VIX` actively rising on your charts?

   * **YES:** Sit on hands or short.

   * **NO:** Proceed to step 3.

3. **Click your target asset (e.g., `Gold`).**

   * Look at the lines connected to it (`RealYields`, `DXY`).

   * Are those drivers moving in the *opposite* direction of your trade?

   * **YES:** You have a green light.

   * **NO:** You are fighting the macro current. Reduce position size or wait.
        """)

if __name__ == "__main__":
    main()
