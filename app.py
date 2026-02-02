# ============================================================
# MACRONEXUS PRO v1.0 (FINAL)
# Author: MacroNexus
# Purpose: Macro Veto + Directional Bias for 3:00 PM EST Swing Trading
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime
import graphviz

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="MacroNexus Pro v1.0",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #e0e0e0; }
.metric { background:#1e2127; padding:12px; border-radius:6px; border-left:4px solid; }
.go { background:#052e16; border:1px solid #22c55e; padding:16px; border-radius:10px; }
.nogo { background:#3f0d0d; border:1px solid #ef4444; padding:16px; border-radius:10px; }
.neutral { background:#1f2937; border:1px solid #6b7280; padding:16px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA UNIVERSE
# ============================================================
TICKERS = {
    'US10Y': '^TNX',   # 10Y Yield (Yahoo, yield*10)
    'DXY': 'DX-Y.NYB',
    'VIX': '^VIX',
    'HYG': 'HYG',
    'SPY': 'SPY',
    'QQQ': 'QQQ',
    'IWM': 'IWM',
    'XLF': 'XLF',
    'XLE': 'XLE',
    'BTC': 'BTC-USD',
    'TLT': 'TLT',
    'GOLD': 'GLD'
}

FALLBACKS = {'DXY': 'UUP', 'VIX': 'VIXY'}

# ============================================================
# DATA FETCH (ROBUST + SAFE)
# ============================================================
@st.cache_data(ttl=300)
def fetch_live_data():
    data = {}

    for key, symbol in TICKERS.items():
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="3mo")

            if hist.empty and key in FALLBACKS:
                symbol = FALLBACKS[key]
                t = yf.Ticker(symbol)
                hist = t.history(period="3mo")

            hist = hist['Close'].dropna()
            if len(hist) < 22:
                raise ValueError("Insufficient data")

            c, p1, p5, p20 = hist.iloc[-1], hist.iloc[-2], hist.iloc[-6], hist.iloc[-21]

            # ----------------------------
            # ✅ CRITICAL: US10Y UNIT LOGIC
            # ----------------------------
            if key == 'US10Y':
                # Yahoo ^TNX already reports changes in basis points
                d1 = c - p1
                d5 = c - p5
                d20 = c - p20

                # ✅ UNIT SANITY ASSERTION
                if abs(d1) > 20:
                    raise RuntimeError("US10Y unit anomaly detected — aborting")

            else:
                d1 = (c - p1) / p1 * 100
                d5 = (c - p5) / p5 * 100
                d20 = (c - p20) / p20 * 100

            data[key] = {
                'price': c,
                'd1': d1,
                'd5': d5,
                'd20': d20,
                'error': False
            }

        except Exception as e:
            data[key] = {'error': True, 'msg': str(e)}

    return data

# ============================================================
# MACRO LOGIC ENGINE
# ============================================================
def analyze_market(d):
    get = lambda k: d[k]['d1']

    hyg, vix = get('HYG'), get('VIX')
    us10y, dxy = get('US10Y'), get('DXY')
    oil, banks = get('XLE'), get('XLF')
    btc = get('BTC')

    # ----------------------------
    # HARD VETO
    # ----------------------------
    if hyg < -0.5 or vix > 5:
        return {
            'decision': 'NO-GO',
            'reason': 'Credit stress or volatility spike',
            'color': 'nogo'
        }

    # ----------------------------
    # REFLATION
    # ----------------------------
    if (oil > 2 or banks > 1) and us10y > 5:
        return {
            'decision': 'GO',
            'reason': 'Reflation: Energy/Banks + rising yields',
            'color': 'go'
        }

    # ----------------------------
    # LIQUIDITY
    # ----------------------------
    if btc > 3 and dxy < -0.4:
        return {
            'decision': 'GO',
            'reason': 'Liquidity expansion: BTC + weak dollar',
            'color': 'go'
        }

    # ----------------------------
    # GOLDILOCKS
    # ----------------------------
    if vix < 0 and abs(us10y) < 5 and hyg > -0.1:
        return {
            'decision': 'GO',
            'reason': 'Goldilocks: Stable vol, rates, credit',
            'color': 'go'
        }

    # ----------------------------
    # NEUTRAL
    # ----------------------------
    return {
        'decision': 'NO-EDGE',
        'reason': 'No macro tailwind — preserve capital',
        'color': 'neutral'
    }

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.sidebar.warning(
        "⚠️ EXECUTION RULE:\n"
        "This tool is designed for **3:00–4:00 PM EST** only.\n\n"
        "If HYG is crashing intraday (> -1.5%), override to **NO‑GO** manually."
    )

    with st.spinner("Loading MacroNexus Pro v1.0…"):
        data = fetch_live_data()
        verdict = analyze_market(data)

    # ========================================================
    # 🔔 3:00 PM GO / NO-GO PANEL
    # ========================================================
    st.markdown("## ⏰ 3:00 PM EST — FINAL DECISION")

    st.markdown(
        f"""
        <div class="{verdict['color']}">
            <h2>{verdict['decision']}</h2>
            <p><b>Reason:</b> {verdict['reason']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # KEY METRICS
    # ========================================================
    st.markdown("### 📊 Macro Drivers")
    cols = st.columns(6)
    keys = ['HYG', 'VIX', 'US10Y', 'DXY', 'XLE', 'BTC']

    for c, k in zip(cols, keys):
        d = data[k]
        if d.get('error'):
            c.error(k)
        else:
            val = d['price']
            chg = d['d1']
            unit = "bps" if k == 'US10Y' else "%"
            c.metric(k, f"{val:.2f}", f"{chg:+.2f}{unit}")

    # ========================================================
    # TRANSMISSION MAP (OPTIONAL CONTEXT)
    # ========================================================
    with st.expander("🌊 Macro Transmission Map"):
        g = graphviz.Digraph()
        g.attr(rankdir='TB')
        g.node('Rates'); g.node('Dollar'); g.node('Credit')
        g.node('Equities'); g.node('Crypto'); g.node('Energy')
        g.edge('Rates', 'Equities')
        g.edge('Dollar', 'Crypto')
        g.edge('Credit', 'Equities')
        g.edge('Rates', 'Energy')
        st.graphviz_chart(g)

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()
