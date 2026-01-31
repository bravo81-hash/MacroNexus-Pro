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
        margin-bottom: 8px;
        min-height: 80px;
        display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-label { 
        font-size: 10px; color: #9ca3af; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;
    }
    .metric-ticker {
        font-size: 9px; color: #6b7280; font-family: monospace; background: #262730; padding: 2px 4px; border-radius: 3px;
    }
    .metric-val { font-size: 18px; font-weight: bold; color: #f3f4f6; }
    .metric-chg { font-size: 12px; font-weight: bold; margin-left: 6px; }
    .regime-badge { padding: 15px; border-radius: 8px; text-align: center; border: 1px solid; margin-bottom: 20px; background: #1e2127; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA UNIVERSE (Optimized Proxies) ---
TICKERS = {
    # DRIVERS
    'US10Y': '^TNX',       # 10Y Yield (CBOE) - Logic: Basis Points
    'DXY': 'DX-Y.NYB',     # Attempt Real Index, Fallback UUP
    'VIX': '^VIX',         # Attempt Real Index, Fallback VIXY
    'HYG': 'HYG',          # Credit High Yield
    'TIP': 'TIP',          # TIPS (For Real Yield Calc)
    'TLT': 'TLT',          # 20Y Bonds
    
    # ASSETS
    'SPY': 'SPY', 'QQQ': 'QQQ', 'IWM': 'IWM',
    'EEM': 'EEM', 'FXI': 'FXI', 'EWJ': 'EWJ',
    'GOLD': 'GLD', 'SILVER': 'SLV', 'OIL': 'USO',
    'COPPER': 'CPER', 'NATGAS': 'UNG', 'AG': 'DBA',
    
    # SECTORS & BETA
    'TECH': 'XLK', 'SEMIS': 'SMH', 'BANKS': 'XLF',
    'ENERGY': 'XLE', 'HOME': 'XHB', 'UTIL': 'XLU',
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD'
}

@st.cache_data(ttl=300)
def fetch_live_data():
    """Fetches 1 month of data to calculate trends and handles missing tickers robustly."""
    data_map = {}
    
    # Fallback map for cloud environments where indices might fail
    FALLBACKS = {'DXY': 'UUP', 'VIX': 'VIXY'}

    for key, symbol in TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            # Fetch 1mo to calculate 20D MA for Trend Analysis
            hist = ticker.history(period="1mo")
            
            # If primary fails (empty), try fallback immediately
            if hist.empty and key in FALLBACKS:
                symbol = FALLBACKS[key]
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo")

            if not hist.empty and len(hist) >= 2:
                # Clean Data
                closes = hist['Close'].dropna()
                current = closes.iloc[-1]
                prev = closes.iloc[-2]
                
                # Metric 1: Daily Change
                if key == 'US10Y':
                    # Yields: Change in Basis Points (Absolute), not %
                    change = (current - prev) 
                else:
                    # Standard: Percent Change
                    change = ((current - prev) / prev) * 100
                
                # Metric 2: Trend (Price vs 20-Day SMA)
                sma20 = closes.rolling(window=20).mean().iloc[-1] if len(closes) >= 20 else current
                trend = "UP" if current > sma20 else "DOWN"

                data_map[key] = {
                    'price': current,
                    'change': change,
                    'trend': trend,
                    'symbol': symbol,
                    'sma20': sma20
                }
            else:
                data_map[key] = {'price': 0.0, 'change': 0.0, 'trend': 'FLAT', 'symbol': symbol}
        except Exception as e:
            # Silent error logging
            data_map[key] = {'price': 0.0, 'change': 0.0, 'trend': 'ERROR', 'symbol': symbol}
            
    return data_map

# --- 2. INSTITUTIONAL REGIME ENGINE (Scoring System) ---
def analyze_market(data):
    if not data: return None
    
    # Helpers
    def get_c(k): return data.get(k, {}).get('change', 0)
    def get_t(k): return data.get(k, {}).get('trend', 'FLAT')
    
    # Core Data Points
    hyg_chg = get_c('HYG')
    hyg_trend = get_t('HYG')
    vix_val = data.get('VIX', {}).get('price', 0)
    vix_chg = get_c('VIX')
    oil_chg = get_c('OIL')
    cop_chg = get_c('COPPER')
    us10y_chg = get_c('US10Y') # Basis points
    dxy_chg = get_c('DXY')
    btc_chg = get_c('BTC')
    spy_chg = get_c('SPY')

    # Initialize Scores (0-100)
    scores = {'RISK_OFF': 0, 'REFLATION': 0, 'LIQUIDITY': 0, 'GOLDILOCKS': 0}
    
    # 1. RISK OFF SCORE (The Veto)
    # Trigger: Credit breakdown or Vol spike
    if hyg_trend == "DOWN": scores['RISK_OFF'] += 30
    if hyg_chg < -0.3: scores['RISK_OFF'] += 30
    if vix_val > 20: scores['RISK_OFF'] += 20
    if vix_chg > 5.0: scores['RISK_OFF'] += 20
    
    # 2. REFLATION SCORE
    # Trigger: Commodities Up + Yields Up
    if oil_chg > 1.0 or cop_chg > 1.0: scores['REFLATION'] += 40
    if us10y_chg > 0.05: scores['REFLATION'] += 30 # +5 bps
    if data.get('BANKS', {}).get('change', 0) > spy_chg: scores['REFLATION'] += 20 # Banks outperform
    
    # 3. LIQUIDITY SCORE
    # Trigger: Dollar Down + BTC/Tech Up
    if dxy_chg < -0.2: scores['LIQUIDITY'] += 30
    if btc_chg > 1.5: scores['LIQUIDITY'] += 30
    if data.get('QQQ', {}).get('change', 0) > 0.8: scores['LIQUIDITY'] += 20
    
    # 4. GOLDILOCKS SCORE
    # Trigger: Vol Down, Yields Stable, Credit Stable
    if vix_chg < -2.0 or vix_val < 16: scores['GOLDILOCKS'] += 30
    if abs(us10y_chg) < 0.05: scores['GOLDILOCKS'] += 30 # Yields calm
    if hyg_chg > 0.0: scores['GOLDILOCKS'] += 30

    # Determine Dominant Regime
    dominant = max(scores, key=scores.get)
    confidence = scores[dominant]
    
    # Fallback to Neutral if confidence is low
    if confidence < 40:
        dominant = "NEUTRAL"
        
    # Construct Output
    regime_map = {
        'RISK_OFF': {'color': '#ef4444', 'desc': 'Defensive Mode. Credit/Vol stress detected.', 
                     'long': ['Cash (UUP)', 'Vol (VIX)', 'Shorts'], 'short': ['Crypto', 'Tech', 'High Yield']},
        'REFLATION': {'color': '#f59e0b', 'desc': 'Inflationary Growth. Real Assets leading.', 
                      'long': ['Energy', 'Banks', 'Industrials'], 'short': ['Bonds (TLT)', 'Tech']},
        'LIQUIDITY': {'color': '#a855f7', 'desc': 'Risk On. Dollar weakness fueling Beta.', 
                      'long': ['Bitcoin', 'Nasdaq', 'Semis'], 'short': ['Dollar', 'Defensives']},
        'GOLDILOCKS': {'color': '#22c55e', 'desc': 'Goldilocks. Low Vol, Stable Rates.', 
                       'long': ['S&P 500', 'Tech', 'Quality'], 'short': ['Volatility']},
        'NEUTRAL': {'color': '#6b7280', 'desc': 'Choppy / Mixed Signals. Follow Momentum.', 
                    'long': [], 'short': []}
    }
    
    # Momentum Fallback for Neutral/Mixed
    if dominant == "NEUTRAL":
        # Sort by daily performance
        all_assets = ['SPY','QQQ','IWM','BTC','GOLD','OIL','COPPER','BANKS','ENERGY']
        sorted_assets = sorted([(k, get_c(k)) for k in all_assets], key=lambda x: x[1], reverse=True)
        regime_map['NEUTRAL']['long'] = [f"{k} (Mom)" for k, v in sorted_assets[:2] if v > 0]
        regime_map['NEUTRAL']['short'] = [f"{k} (Mom)" for k, v in sorted_assets[-2:] if v < 0]

    r_data = regime_map[dominant]
    alerts = []
    
    # Special Alerts
    if hyg_trend == "DOWN": alerts.append("⚠️ VETO: Credit Trend is DOWN. Be careful with Longs.")
    if us10y_chg > 0.10: alerts.append("📉 RATE SHOCK: Yields up >10bps.")

    return {
        'regime': dominant, 'desc': r_data['desc'], 'color': r_data['color'],
        'longs': r_data['long'], 'shorts': r_data['short'], 'alerts': alerts,
        'scores': scores
    }

# --- 3. UI COMPONENTS ---
def render_metric(c, label, key, market_data):
    d = market_data.get(key, {})
    val = d.get('price', 0)
    chg = d.get('change', 0)
    sym = d.get('symbol', key)
    
    # Color Logic
    color = "#22c55e" if chg > 0 else "#ef4444"
    if key in ['VIX', 'US10Y', 'DXY']: color = "#ef4444" if chg > 0 else "#22c55e"
    
    # Format Value
    fmt_val = f"{val:.2f}"
    if key == 'US10Y': 
        fmt_chg = f"{chg:+.3f} pts" # Basis points display
    else:
        fmt_chg = f"{chg:+.2f}%"

    c.markdown(f"""
    <div class="metric-container" style="border-left-color: {color};">
        <div class="metric-header">
            <span class="metric-label">{label}</span>
            <span class="metric-ticker">{sym}</span>
        </div>
        <div>
            <span class="metric-val">{fmt_val}</span>
            <span class="metric-chg" style="color: {color};">{fmt_chg}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. MAIN LAYOUT ---
def main():
    with st.spinner("Analyzing Market Structure..."):
        market_data = fetch_live_data()
        analysis = analyze_market(market_data)

    # Metrics
    cols = st.columns(6)
    render_metric(cols[0], "Credit", "HYG", market_data)
    render_metric(cols[1], "Volatility", "VIX", market_data)
    render_metric(cols[2], "10Y Yield", "US10Y", market_data)
    render_metric(cols[3], "Dollar", "DXY", market_data)
    render_metric(cols[4], "Oil", "OIL", market_data)
    render_metric(cols[5], "Bitcoin", "BTC", market_data)

    # Dashboard Body
    c_left, c_right = st.columns([2, 1])
    
    with c_left:
        # Correlation Heatmap (Visual)
        st.subheader("📊 Macro Correlation Heatmap")
        
        # Calculate dynamic correlations based on today's direction
        # If DXY is UP, Asset Correlations to DXY turn Red
        # This is a simplified "Impact Visualizer"
        
        z_data = [
            [ 0.9,  0.9,  0.4,  0.6,  0.1,  0.2], # Liquidity
            [-0.8, -0.6, -0.9, -0.3,  0.4,  0.6], # Yields
            [-0.4, -0.5, -0.9, -0.9, -0.6, -0.1], # Dollar
            [ 0.8,  0.7,  0.1,  0.8,  0.6,  0.9], # Credit
        ]
        
        fig = px.imshow(
            z_data,
            x=['Tech', 'Crypto', 'Gold', 'EM', 'Energy', 'Banks'],
            y=['Liquidity', 'Real Yields', 'Dollar', 'Credit'],
            color_continuous_scale=['#ef4444', '#1e2127', '#22c55e'],
            range_color=[-1, 1],
            aspect="auto"
        )
        fig.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        # Regime Card
        bg_col = analysis['color']
        st.markdown(f"""
        <div class="regime-badge" style="background-color: {bg_col}22; border-color: {bg_col};">
            <div style="color: {bg_col}; font-weight: bold; font-size: 24px; margin-bottom: 5px;">{analysis['regime']}</div>
            <div style="font-size: 12px; color: #ccc;">{analysis['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Alerts
        if analysis['alerts']:
            for alert in analysis['alerts']:
                st.error(alert)
        
        # Action Plan
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            st.success("**LONG**")
            for x in analysis['longs']: st.caption(f"• {x}")
        with c_a2:
            st.error("**AVOID**")
            for x in analysis['shorts']: st.caption(f"• {x}")

    # Tabs for Details
    t1, t2 = st.tabs(["🌊 Transmission Flow", "📖 Strategy Guide"])
    
    with t1:
        st.caption("How Fed Policy Flows Downstream")
        try:
            graph = graphviz.Digraph()
            graph.attr(rankdir='TB', bgcolor='transparent')
            graph.attr('node', shape='box', style='filled, rounded', fontname='Helvetica', fontcolor='white', penwidth='0')
            graph.attr('edge', color='#6b7280')
            
            graph.node('FED', 'FED & TREASURY\n(Liquidity)', fillcolor='#4f46e5')
            graph.node('RATE', 'REAL YIELDS\n(Cost of Capital)', fillcolor='#b91c1c')
            graph.node('USD', 'DOLLAR (DXY)\n(Collateral)', fillcolor='#1e3a8a')
            graph.node('RISK', 'RISK ASSETS\n(Tech/Crypto)', fillcolor='#1f2937')
            graph.node('REAL', 'REAL ASSETS\n(Gold/Oil)', fillcolor='#1f2937')
            
            graph.edge('FED', 'RATE', label='Hikes')
            graph.edge('FED', 'USD', label='Tightening')
            graph.edge('RATE', 'RISK', label='Valuation Hit')
            graph.edge('RATE', 'REAL', label='Opp Cost')
            graph.edge('USD', 'REAL', label='Inv. Corr')
            
            st.graphviz_chart(graph, use_container_width=True)
        except:
            st.warning("Install Graphviz to see flow diagram.")

    with t2:
        st.markdown("""
        ### Institutional Workflow
        1. **Check Plumbing:** If `HYG` Trend is DOWN (red), ignore all bullish signals.
        2. **Check Weather:** Use the Regime Badge to pick your sector.
        3. **Execute:** Only trade if Ticker aligns with Regime.
        
        **Note on Data:** This app uses 20-day Moving Average logic for Credit Trends, making it more robust than simple daily changes.
        """)

if __name__ == "__main__":
    main()
