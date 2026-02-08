import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
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
        background: linear-gradient(180deg, rgba(22, 25, 33, 1) 0%, rgba(15, 17, 22, 1) 100%);
        border: 1px solid #2A2E39;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-bottom: 20px;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    .strat-header {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 15px; margin-bottom: 15px;
    }
    .strat-title { font-size: 20px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px; }
    .strat-tag { font-size: 10px; font-weight: 700; padding: 4px 8px; border-radius: 4px; text-transform: uppercase; }
    
    .strat-section { margin-bottom: 15px; }
    .strat-subtitle { font-size: 11px; color: #6B7280; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 6px; }
    .strat-data { font-size: 15px; color: #E5E7EB; font-family: 'Inter', sans-serif; font-weight: 500; line-height: 1.4; }
    
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
        padding: 6px 16px;
        border-radius: 6px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 18px;
        display: inline-block;
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #2A2E39; }
    .stTabs [data-baseweb="tab"] { height: 45px; border-radius: 6px 6px 0 0; border: none; color: #8B9BB4; font-weight: 600; font-size: 14px; }
    .stTabs [aria-selected="true"] { background-color: #1E222D; color: #FFF; border-bottom: 2px solid #3B82F6; }
    
    /* Utilities */
    .badge-blue { background: rgba(59, 130, 246, 0.15); color: #60A5FA; padding: 2px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(59, 130, 246, 0.3); font-family: monospace; }
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #34D399; padding: 2px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(16, 185, 129, 0.3); font-family: monospace; }
    .badge-red { background: rgba(239, 68, 68, 0.15); color: #F87171; padding: 2px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(239, 68, 68, 0.3); font-family: monospace; }
    
    .context-box {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #3B82F6;
        padding: 15px;
        font-size: 13px;
        color: #9CA3AF;
        margin-top: 0px;
        border-radius: 0 6px 6px 0;
        height: 100%;
    }
    .context-header { font-weight: 700; color: #E5E7EB; margin-bottom: 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Expander Styling */
    .streamlit-expanderHeader { font-weight: 700; color: #E5E7EB; background-color: #161920; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA UNIVERSE ---
TICKERS = {
    # Drivers
    'US10Y': '^TNX', 'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'HYG': 'HYG', 'TLT': 'TLT', 
    # Commodities
    'GOLD': 'GLD', 'OIL': 'USO', 'COPPER': 'CPER', 
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
            
            if not hist_clean.empty and len(hist_clean) >= 50:
                history_data[key] = hist_clean
                
                curr = hist_clean.iloc[-1]
                prev = hist_clean.iloc[-2]
                
                # Calculations for RRG (Trend vs Momentum)
                # Trend = Price vs 20 SMA
                sma20 = hist_clean.rolling(window=20).mean().iloc[-1]
                trend_score = ((curr / sma20) - 1) * 100
                
                # Momentum = Price vs 5 SMA
                sma5 = hist_clean.rolling(window=5).mean().iloc[-1]
                mom_score = ((curr / sma5) - 1) * 100
                
                # Standard Changes
                prev_w = hist_clean.iloc[-6] 
                
                if key == 'US10Y':
                    chg_d = (curr - prev) * 10 
                    chg_w = (curr - prev_w) * 10
                    disp_fmt = "bps"
                else:
                    chg_d = ((curr - prev) / prev) * 100
                    chg_w = ((curr - prev_w) / prev_w) * 100
                    disp_fmt = "%"
                
                data[key] = {
                    'price': curr, 'change': chg_d, 
                    'change_w': chg_w,
                    'trend_score': trend_score, 'mom_score': mom_score,
                    'symbol': symbol, 'fmt': disp_fmt, 'valid': True
                }
            else:
                data[key] = {'price': 0, 'change': 0, 'change_w': 0, 'trend_score':0, 'mom_score':0, 'symbol': symbol, 'fmt': "%", 'valid': False}
        except:
            data[key] = {'price': 0, 'change': 0, 'change_w': 0, 'trend_score':0, 'mom_score':0, 'symbol': symbol, 'fmt': "%", 'valid': False}
            
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

# --- 6. STRATEGY DATABASE (Expanded) ---
STRATEGIES = {
    "GOLDILOCKS": {
        "desc": "Low Vol + Steady Trend. Market climbing wall of worry.",
        "risk": "1.5%", "bias": "Long",
        "index": {
            "strat": "Directional Diagonal", 
            "dte": "Front 17 / Back 31", 
            "setup": "Buy Back ITM (70D) / Sell Front OTM (30D)", 
            "notes": "Stock replacement. Trend (Delta) + Decay (Theta). Upside is uncapped."
        },
        "stock": {
            "strat": "Call Debit Spreads",
            "dte": "45-60 DTE",
            "setup": "Buy 60D / Sell 30D (Spread)",
            "notes": "Focus on Relative Strength Leaders (Tech, Semis). Use pullbacks to EMA21."
        },
        "longs": "TECH, SEMIS, DISC", "shorts": "VIX, TLT"
    },
    "LIQUIDITY": {
        "desc": "High Liquidity / Dollar Weakness. Drift Up environment.",
        "risk": "1.0%", "bias": "Aggressive Long",
        "index": {
            "strat": "Flyagonal (Drift)", 
            "dte": "Entry 7-10 DTE", 
            "setup": "Upside: Call BWB (Long +10 / Short 2x +50 / Long +60). Downside: Put Diagonal.", 
            "notes": "Captures the drift. Upside tent funds the downside floor. Target 4% Flash Win."
        },
        "stock": {
            "strat": "Risk Reversals",
            "dte": "60 DTE",
            "setup": "Sell OTM Put / Buy OTM Call",
            "notes": "Funding long delta with short volatility. Best for High Beta (Crypto proxies)."
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
            "notes": "Focus on Russell 2000 (IWM). Avoid long duration Tech (QQQ) as rates rise."
        },
        "stock": {
            "strat": "Cash Secured Puts",
            "dte": "30-45 DTE",
            "setup": "Sell 30D Puts on Energy/Banks",
            "notes": "Energy (XLE) and Banks (XLF) benefit from rising yields. Sell premium to acquire."
        },
        "longs": "ENERGY, BANKS, IND", "shorts": "TLT, TECH"
    },
    "NEUTRAL": {
        "desc": "Chop / Range Bound. No clear direction.",
        "risk": "Income Size", "bias": "Neutral/Theta",
        "index": {
            "strat": "TimeEdge (SPX) / TimeZone (RUT)", 
            "dte": "Entry 15 / Exit 7", 
            "setup": "Put Calendar Spread (ATM) or Double Calendar", 
            "notes": "Pure Theta play. Sell 15 DTE / Buy 22+ DTE. Requires VIX < 20."
        },
        "stock": {
            "strat": "Iron Condor",
            "dte": "30-45 DTE",
            "setup": "Sell 20D Call / Sell 20D Put (Wings 10 wide)",
            "notes": "Delta neutral income. Best on low beta stocks (KO, PEP) during chop."
        },
        "longs": "INCOME, CASH", "shorts": "MOMENTUM"
    },
    "RISK OFF": {
        "desc": "High Volatility / Credit Stress. Preservation mode.",
        "risk": "0.5%", "bias": "Short/Hedge",
        "index": {
            "strat": "A14 Put BWB", 
            "dte": "Entry 14 / Exit 7", 
            "setup": "Long ATM / Short 2x -40 / (Skip) / Long -60", 
            "notes": "Crash Catcher. Zero upside risk. Profit tent expands into the crash. Enter Friday AM."
        },
        "stock": {
            "strat": "Put Debit Spreads",
            "dte": "60 DTE",
            "setup": "Buy 40D / Sell 15D",
            "notes": "Directional downside. Selling the 15D put reduces cost and offsets IV crush."
        },
        "longs": "VIX, DXY", "shorts": "SPY, IWM, HYG"
    }
}

# --- 7. DYNAMIC HELPER ---
def get_val(data, key, timeframe):
    d = data.get(key, {})
    if timeframe == 'Tactical (Daily)':
        return d.get('change', 0)
    else: 
        return d.get('change_w', 0)

# --- 8. VISUALIZATION FUNCTIONS ---

def plot_nexus_graph_dots(data, timeframe):
    """
    Creates the 'Dots' Plotly Network Graph requested in original specs.
    Replaces Graphviz with Plotly for specific aesthetic.
    """
    # 1. Define Nodes & Positions (The "Map")
    nodes = {
        'US10Y': {'pos': (0, 0), 'label': 'Rates (^TNX)'}, 
        'DXY': {'pos': (0.8, 0.8), 'label': 'Dollar (DXY)'},
        'SPY': {'pos': (-0.8, 0.8), 'label': 'S&P 500 (SPY)'}, 
        'QQQ': {'pos': (-1.2, 0.4), 'label': 'Nasdaq (QQQ)'},
        'GOLD': {'pos': (0.8, -0.8), 'label': 'Gold (GLD)'}, 
        'HYG': {'pos': (-0.4, -0.8), 'label': 'Credit (HYG)'},
        'BTC': {'pos': (-1.5, 1.5), 'label': 'Bitcoin'}, 
        'OIL': {'pos': (1.5, -0.4), 'label': 'Oil (USO)'},
        'COPPER': {'pos': (1.2, -1.2), 'label': 'Copper'}, 
        'IWM': {'pos': (-1.2, -1.0), 'label': 'Russell (IWM)'},
        'SEMIS': {'pos': (-1.8, 0.8), 'label': 'Semis (SMH)'}, 
        'ENERGY': {'pos': (1.8, -0.8), 'label': 'Energy (XLE)'},
        'HOME': {'pos': (-0.8, -0.4), 'label': 'Housing (XHB)'},
        'BANKS': {'pos': (1.5, -1.0), 'label': 'Banks (XLF)'}, 
        'VIX': {'pos': (0, 1.5), 'label': 'Vol (^VIX)'}
    }
    
    # 2. Define Edges (The "Plumbing")
    edges = [
        ('US10Y','QQQ'), ('US10Y','GOLD'), ('US10Y','HOME'), 
        ('DXY','GOLD'), ('DXY','OIL'), 
        ('HYG','SPY'), ('HYG','IWM'), ('HYG','BANKS'), 
        ('QQQ','BTC'), ('QQQ','SEMIS'), 
        ('COPPER','US10Y'), ('OIL','ENERGY'), ('VIX','SPY')
    ]
    
    # 3. Build Plotly Data
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
        node_x.append(x); node_y.append(y)
        
        # Get Value
        val = get_val(data, key, timeframe)
        
        # Color Logic
        col = '#22c55e' if val > 0 else '#ef4444' # Green Up, Red Down
        if val == 0: col = '#6b7280' # Grey
        
        # Invert for Drivers (Rates, DXY, VIX up is usually 'Bad' or Red contextually, 
        # but to keep simple visual consistency with price change, we keep Green = Price Up)
        
        node_color.append(col)
        node_size.append(45 if key in ['US10Y', 'DXY', 'HYG'] else 35)
        
        fmt_val = f"{val:+.2f}%"
        if key == 'US10Y': fmt_val = f"{val:+.1f} bps"
            
        node_text.append(f"<b>{info['label']}</b><br>Chg: {fmt_val}")

    # 4. Create Figure
    fig = go.Figure()
    
    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, 
        mode='lines', 
        line=dict(width=1, color='#4b5563'), 
        hoverinfo='none'
    ))
    
    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, 
        mode='markers+text', 
        text=[n.split('<br>')[0] for n in node_text], # Short label for chart
        textposition="bottom center", 
        hovertext=node_text, 
        hoverinfo="text", 
        marker=dict(
            size=node_size, 
            color=node_color, 
            line=dict(width=2, color='white')
        ), 
        textfont=dict(size=11, color='white')
    ))
    
    fig.update_layout(
        showlegend=False, 
        margin=dict(b=0,l=0,r=0,t=0), 
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]), 
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.0, 2.0]), 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        height=500
    )
    return fig

def plot_sankey_sectors(data, timeframe):
    sector_keys = ['TECH','SEMIS','BANKS','ENERGY','HOME','UTIL','HEALTH','MAT','COMM']
    sectors = {k: get_val(data, k, timeframe) for k in sector_keys}
    df = pd.DataFrame(list(sectors.items()), columns=['id', 'val']).sort_values('val', ascending=False)
    winners = df.head(3); losers = df.tail(3)
    labels = list(losers['id']) + list(winners['id'])
    sources, targets, values, colors = [], [], [], []
    for i in range(len(losers)):
        for j in range(len(winners)):
            sources.append(i); targets.append(len(losers) + j)
            values.append(abs(losers.iloc[i]['val']) + abs(winners.iloc[j]['val']))
            colors.append('rgba(59, 130, 246, 0.2)')
    node_colors = ['#ef4444']*3 + ['#22c55e']*3
    fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=node_colors), link=dict(source=sources, target=targets, value=values, color=colors))])
    fig.update_layout(title_text=f"Sector Rotation ({'Daily' if timeframe=='Tactical (Daily)' else 'Weekly'})", font=dict(color='white'), paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plot_sankey_assets(data, timeframe):
    asset_keys = ['SPY','TLT','DXY','GOLD','BTC','OIL','HYG']
    assets = {k: get_val(data, k, timeframe) for k in asset_keys}
    df = pd.DataFrame(list(assets.items()), columns=['id', 'val']).sort_values('val', ascending=False)
    winners = df.head(3); losers = df.tail(3)
    labels = list(losers['id']) + list(winners['id'])
    sources, targets, values, colors = [], [], [], []
    for i in range(len(losers)):
        for j in range(len(winners)):
            sources.append(i); targets.append(len(losers) + j)
            values.append(abs(losers.iloc[i]['val']) + abs(winners.iloc[j]['val']))
            colors.append('rgba(168, 85, 247, 0.2)')
    node_colors = ['#ef4444']*3 + ['#22c55e']*3
    fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=node_colors), link=dict(source=sources, target=targets, value=values, color=colors))])
    fig.update_layout(title_text=f"Asset Rotation ({'Daily' if timeframe=='Tactical (Daily)' else 'Weekly'})", font=dict(color='white'), paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plot_rrg(data, category, view):
    items = []
    if category == 'SECTORS': keys = ['TECH','SEMIS','BANKS','ENERGY','HOME','UTIL','STAPLES','DISC','IND','HEALTH','MAT','COMM','RE']
    else: keys = ['SPY','QQQ','IWM','GOLD','BTC','TLT','DXY','HYG','OIL']
    
    for k in keys:
        d = data.get(k, {})
        # RRG LOGIC FIX:
        # X-Axis = Trend (Longer MA comparison)
        # Y-Axis = Momentum (Shorter MA comparison)
        # This provides a more accurate "Rotation" view than just price change
        x = d.get('trend_score', 0) 
        y = d.get('mom_score', 0)
        
        # Quadrant Color Logic
        if x > 0 and y > 0: c = '#22c55e' # LEADING (Green)
        elif x < 0 and y > 0: c = '#3b82f6' # IMPROVING (Blue)
        elif x > 0 and y < 0: c = '#f59e0b' # WEAKENING (Yellow)
        else: c = '#ef4444' # LAGGING (Red)
        
        items.append({'Symbol': k, 'Trend': x, 'Momentum': y, 'Color': c})
        
    df = pd.DataFrame(items)
    fig = px.scatter(df, x='Trend', y='Momentum', text='Symbol', color='Color', color_discrete_map="identity")
    fig.update_traces(textposition='top center', marker=dict(size=14, line=dict(width=1, color='white')))
    
    # Quadrant Lines
    fig.add_hline(y=0, line_dash="dot", line_color="#555")
    fig.add_vline(x=0, line_dash="dot", line_color="#555")
    
    limit = 5 # Fixed scale for better visualization of rotation
    fig.update_layout(
        xaxis=dict(range=[-limit, limit], zeroline=False, showgrid=True, gridcolor='#333', title="Trend (Price vs SMA20)"),
        yaxis=dict(range=[-limit, limit], zeroline=False, showgrid=True, gridcolor='#333', title="Momentum (Price vs SMA5)"),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'), showlegend=False, height=450, 
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

def plot_correlation_heatmap(history_df):
    if history_df.empty: return go.Figure()
    corr = history_df.pct_change().corr()
    subset = ['US10Y', 'DXY', 'VIX', 'HYG', 'SPY', 'QQQ', 'IWM', 'BTC', 'GOLD', 'OIL']
    cols = [c for c in subset if c in corr.columns]; corr_subset = corr.loc[cols, cols]
    fig = px.imshow(corr_subset, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
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
        col.markdown(f"""<div class="metric-card" style="border-left: 3px solid {color};"><div class="metric-label">{label}</div><div class="metric-value">{val:.2f}<span class="metric-delta" style="color: {color};">{fmt_chg}</span></div></div>""", unsafe_allow_html=True)

    metric_tile(c1, "Credit (HYG)", "HYG"); metric_tile(c2, "Volatility (VIX)", "VIX", invert=True); metric_tile(c3, "10Y Yield", "US10Y", invert=True)
    metric_tile(c4, "Dollar (DXY)", "DXY", invert=True); metric_tile(c5, "Oil", "OIL"); metric_tile(c6, "Bitcoin", "BTC")

    # --- MAIN CONTROLS ---
    auto_regime = determine_regime(market_data)

    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        override = st.checkbox("Manual Override", value=False)
        active_regime = st.selectbox("Force Regime", list(STRATEGIES.keys()), label_visibility="collapsed") if override else auto_regime
    with ctrl2:
        timeframe = st.selectbox("Analytic View", ["Tactical (Daily)", "Structural (Weekly)"], label_visibility="collapsed")
    with ctrl3:
        r_colors = {"GOLDILOCKS": "#10B981", "LIQUIDITY": "#A855F7", "REFLATION": "#F59E0B", "NEUTRAL": "#6B7280", "RISK OFF": "#EF4444"}
        rc = r_colors.get(active_regime, "#6B7280")
        st.markdown(f"""<div style="text-align: right; display: flex; align-items: center; justify-content: flex-end; gap: 15px;"><div style="text-align: right;"><div style="font-size: 10px; color: #8B9BB4; letter-spacing: 1px;">SYSTEM STATUS</div><div style="font-size: 14px; font-weight: bold; color: {rc};">ACTIVE</div></div><div class="regime-badge" style="background: {rc}22; color: {rc}; border: 1px solid {rc};">{active_regime}</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- TABS ---
    tab_mission, tab_workflow, tab_pulse, tab_macro, tab_playbook = st.tabs([
        "🚀 MISSION CONTROL", 
        "📋 WORKFLOW",
        "📊 MARKET PULSE", 
        "🕸️ MACRO MACHINE", 
        "📖 STRATEGY PLAYBOOK"
    ])

    # === TAB 1: MISSION CONTROL ===
    with tab_mission:
        # TELEMETRY
        with st.expander("🎛️ SPX Income Reactor Telemetry (Manual Input)", expanded=True):
            tc1, tc2, tc3, tc4 = st.columns(4)
            asset_mode = tc1.radio("Asset Class", ["INDEX (SPX/RUT)", "STOCKS"], horizontal=True) # TOGGLE IS HERE
            iv_rank = tc2.slider("IV Rank (Percentile)", 0, 100, 45)
            skew_rank = tc3.slider("Skew Rank", 0, 100, 50)
            adx_val = tc4.slider("Trend ADX", 0, 60, 20)

        st.divider()

        strat_data = STRATEGIES[active_regime]
        
        # --- REACTOR LOGIC ---
        reactor_output = {}
        
        if asset_mode == "STOCKS":
            # Stock Logic relies purely on Regime
            reactor_output = strat_data["stock"]
        else:
            # Index Logic uses the Vol/Skew Reactor
            if active_regime == "RISK OFF":
                reactor_output = STRATEGIES["RISK OFF"]["index"]
            elif iv_rank > 50: 
                if skew_rank > 80:
                    reactor_output = {"strat": "Put BWB (High Skew)", "dte": "21-30 DTE", "setup": "Long ATM / Short -40 / Skip / Long -60", "notes": "REACTOR WARNING: High Skew detected (>80). Crash risk elevated. Use BWB to eliminate upside risk and profit from crash."}
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
            st.markdown(f"""
<div class="strat-card">
<div class="strat-header">
    <div class="strat-title" style="color: {rc}">CONTEXT</div>
    <div class="strat-tag" style="background: {rc}22; color: {rc}">{active_regime}</div>
</div>
<div class="strat-section">
    <div class="strat-subtitle">DESCRIPTION</div>
    <div class="strat-data">{strat_data['desc']}</div>
</div>
<div class="strat-section">
    <div class="strat-subtitle">RISK SIZE</div>
    <div class="strat-data" style="font-size: 24px; color: {rc}">{strat_data['risk']}</div>
</div>
<div class="strat-section">
    <div class="strat-subtitle">BIAS</div>
    <div class="strat-data">{strat_data['bias']}</div>
</div>
<div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
    <div style="margin-bottom: 8px;"><span class="badge-green">TARGETS</span> <span style="font-size: 13px; color: #D1D5DB;">{strat_data.get('longs', '')}</span></div>
    <div><span class="badge-red">AVOID</span> <span style="font-size: 13px; color: #D1D5DB;">{strat_data.get('shorts', '')}</span></div>
</div>
</div>
""", unsafe_allow_html=True)

        with col_R:
            st.markdown(f"""
<div class="strat-card" style="border-color: {rc};">
<div class="strat-header">
    <div class="strat-title" style="color: {rc};">TACTICAL EXECUTION</div>
    <div class="strat-tag" style="border: 1px solid {rc}; color: {rc};">{asset_mode}</div>
</div>
<div style="font-size: 28px; font-weight: 800; margin-bottom: 20px; color: #FFF;">{reactor_output['strat']}</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 25px;">
    <div>
        <div class="strat-subtitle">⏱️ TIMING (DTE)</div>
        <div class="strat-data" style="font-weight: 700; font-size: 18px;">{reactor_output['dte']}</div>
    </div>
    <div>
        <div class="strat-subtitle">🏗️ STRUCTURE</div>
        <div class="strat-data" style="font-weight: 700; font-size: 18px;">{reactor_output['setup']}</div>
    </div>
</div>

<div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; border-left: 4px solid {rc}; margin-bottom: 20px;">
    <div class="strat-subtitle" style="margin-top: 0; color: {rc};">🧠 THE LOGIC</div>
    <div class="strat-data" style="font-size: 14px; color: #E5E7EB; font-style: italic;">"{reactor_output['notes']}"</div>
</div>

<div style="display: flex; gap: 10px;">
    <span class="badge-blue">IV Rank: {iv_rank}</span>
    <span class="badge-blue">Skew: {skew_rank}</span>
    <span class="badge-blue">ADX: {adx_val}</span>
</div>
</div>
""", unsafe_allow_html=True)

    # === TAB 2: WORKFLOW REFERENCE ===
    with tab_workflow:
        st.subheader("📋 The 5-Phase Mission Control Workflow")
        
        w1, w2, w3, w4, w5 = st.columns(5)
        
        with w1:
            st.markdown("""
<div class="metric-card" style="height: 250px;">
<div style="color: #F87171; font-weight: bold; margin-bottom: 10px;">PHASE 1: VETO</div>
<div style="font-size: 12px; color: #AAA;">
1. Check <b>HYG</b> (Credit). Is it crashing (< -0.5%)?
<br>2. Check <b>VIX</b>. Is it spiking (> 5%)?
<br><br>
<span style="color: #F87171;">If YES: STOP. Go to Risk Off.</span>
</div>
</div>
""", unsafe_allow_html=True)
            
        with w2:
            st.markdown("""
<div class="metric-card" style="height: 250px;">
<div style="color: #FBBF24; font-weight: bold; margin-bottom: 10px;">PHASE 2: REGIME</div>
<div style="font-size: 12px; color: #AAA;">
Identify the "Tailwind".
<br>• <b>Goldilocks:</b> Growth + Low Vol
<br>• <b>Liquidity:</b> DXY Down + Crypto Up
<br>• <b>Reflation:</b> Yields + Oil Up
</div>
</div>
""", unsafe_allow_html=True)
            
        with w3:
            st.markdown("""
<div class="metric-card" style="height: 250px;">
<div style="color: #60A5FA; font-weight: bold; margin-bottom: 10px;">PHASE 3: SECTOR</div>
<div style="font-size: 12px; color: #AAA;">
Use the RRG & Sankey charts.
<br>• Find sectors moving from <b>Improving</b> to <b>Leading</b>.
<br>• Confirm capital flow matches Regime (e.g. Risk On = Tech inflows).
</div>
</div>
""", unsafe_allow_html=True)
            
        with w4:
            st.markdown("""
<div class="metric-card" style="height: 250px;">
<div style="color: #A78BFA; font-weight: bold; margin-bottom: 10px;">PHASE 4: TACTICS</div>
<div style="font-size: 12px; color: #AAA;">
Consult the <b>SPX Reactor</b>.
<br>• <b>Vol Check:</b> IV Rank > 50?
<br>• <b>Skew Check:</b> Crash risk?
<br>• <b>Trend Check:</b> ADX > 25?
</div>
</div>
""", unsafe_allow_html=True)
            
        with w5:
            st.markdown("""
<div class="metric-card" style="height: 250px;">
<div style="color: #34D399; font-weight: bold; margin-bottom: 10px;">PHASE 5: EXECUTE</div>
<div style="font-size: 12px; color: #AAA;">
3:00 PM EST Check.
<br>• Confirm Price Action.
<br>• Verify DTE matches plan.
<br>• <b>Enter Trade.</b>
</div>
</div>
""", unsafe_allow_html=True)

    # === TAB 3: MARKET PULSE (Charts) ===
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

    # === TAB 4: MACRO MACHINE ===
    with tab_macro:
        st.subheader("🕸️ The Macro Transmission Mechanism")
        
        col_graph, col_legend = st.columns([3, 1])
        
        with col_graph:
            # THIS IS THE "DOTS" FEATURE REQUESTED
            st.plotly_chart(plot_nexus_graph_dots(market_data, timeframe), use_container_width=True)
            
        with col_legend:
            st.markdown(f"""
<div class="context-box" style="margin-top: 0;">
<div class="context-header">💡 The Plumbing</div>
<div>This graph visualizes the causal chain of the economy using a network model.</div>
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

    # === TAB 5: PLAYBOOK (Expanded) ===
    with tab_playbook:
        st.subheader("📚 Detailed Strategy Rulebook")
        
        with st.expander("🔴 A14 (Crash Protection)", expanded=False):
            st.markdown("""
            ### A14: The "Anti-Fragile" Hedge
            **Concept:** Financing downside protection using OTM puts, creating a "free" crash catcher if filled for a credit.
            
            **1. Setup & Structure (Put Broken Wing Butterfly)**
            * **Long:** 1x ATM Put (e.g., 4000)
            * **Short:** 2x OTM Puts (e.g., 3960 / -40 pts)
            * **Long:** 1x OTM Put (e.g., 3900 / -60 pts) -> *Note the skip strikes*
            
            **2. Entry Protocol**
            * **Time:** Friday Morning (~1 hour after open).
            * **DTE:** 14 Days.
            * **Target Debit:** Net Credit or very small debit.
            
            **3. Management**
            * **Upside (Market Rallies):** Do nothing. Keep the credit.
            * **Downside (Market Crashes):** As price approaches short strikes, the "Tent" expands. 
            * **Exit:** Hard stop at **7 DTE**. Do not hold into gamma week.
            """)
            
        with st.expander("🟣 TIMEZONE (RUT Income)", expanded=False):
            st.markdown("""
            ### TimeZone: High Prob RUT Income
            **Concept:** Harvest theta on the Russell 2000 (RUT) using a hedged structure.
            
            **1. Structure (Combined)**
            * **Leg A (Income):** Put Credit Spread. Sell ~14 Delta / Buy ~5 Delta.
            * **Leg B (Hedge):** Put Calendar. Sell Front Month (15 DTE) / Buy Back Month (45 DTE) at same strike.
            
            **2. Entry Protocol**
            * **Time:** Thursday Afternoon (3:00 PM EST).
            * **DTE:** 15 DTE (Front Month).
            
            **3. Management**
            * **Profit Target:** 5-7% of Margin.
            * **Max Loss:** 5% of Margin.
            * **Hard Stop:** Exit at **7 DTE**.
            * **Adjustment:** If market drops, roll the Calendar Put down or add a Debit Spread.
            """)
            
        with st.expander("🔵 TIMEEDGE (SPX Neutral)", expanded=False):
            st.markdown("""
            ### TimeEdge: Pure Theta Decay
            **Concept:** Exploiting the decay differential between Front and Back months in SPX.
            
            **1. Structure**
            * **Double Calendar:** Sell 15 DTE / Buy 43 DTE (approx).
            * **Strikes:** ATM or slightly OTM.
            
            **2. Entry Protocol**
            * **Time:** Thursday Afternoon (3:30 PM).
            * **DTE:** 15 DTE front month.
            
            **3. Rules**
            * **Management:** No Touch ideally.
            * **Profit Target:** 10%.
            * **Stop Loss:** 10%.
            * **Exit:** Hard stop at 1 DTE (but usually earlier).
            """)

        with st.expander("🌊 FLYAGONAL (Liquidity Drift)", expanded=False):
            st.markdown("""
            ### Flyagonal: The Drift Catcher
            **Concept:** Captures the "melt-up" or slow drift higher.
            
            **1. Structure**
            * **Upside:** Call Broken Wing Butterfly (+10 / -50 / +60 width).
            * **Downside:** Put Diagonal (Sell Front OTM / Buy Back OTM).
            
            **2. Entry Protocol**
            * **Time:** Tuesday or Friday.
            * **DTE:** 7-10 Days.
            
            **3. Management**
            * **Flash Win:** If profit > 4% in 1-2 days, CLOSE immediately.
            * **Scratches:** If trade goes nowhere by 3 DTE, close.
            """)

if __name__ == "__main__":
    main()
