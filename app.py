import datetime
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ----------------------------
# 0) BASIC CONFIG / CONSTANTS
# ----------------------------

APP_VERSION = "MacroNexus Pro (Matrix Edition v2.5-Stable)"

# Data universe
TICKERS: Dict[str, str] = {
    # Drivers
    "US10Y": "^TNX",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "HYG": "HYG",
    "TLT": "TLT",
    # Commodities
    "GOLD": "GLD",
    "OIL": "USO",
    "COPPER": "CPER",
    # Indices
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "RUT": "^RUT",
    # Sectors
    "TECH": "XLK",
    "SEMIS": "SMH",
    "BANKS": "XLF",
    "ENERGY": "XLE",
    "HOME": "XHB",
    "UTIL": "XLU",
    "STAPLES": "XLP",
    "DISC": "XLY",
    "IND": "XLI",
    "HEALTH": "XLV",
    "MAT": "XLB",
    "COMM": "XLC",
    "RE": "XLRE",
    # Crypto
    "BTC": "BTC-USD",
}

PROXIES: Dict[str, str] = {
    "DXY": "UUP",    # USD Bull ETF (proxy for DXY)
    "VIX": "VIXY",   # VIX futures ETN/ETF proxy (not VIX spot)
    "RUT": "IWM",    # Russell proxy (reasonable)
}

CRITICAL_KEYS = ["HYG", "VIX", "US10Y", "DXY"]
TIMEFRAMES = ["Tactical (Daily)", "Structural (Weekly, WTD)"]


# ----------------------------
# 1) STREAMLIT PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="MacroNexus Pro Terminal",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ----------------------------
# 2) PROFESSIONAL STYLING (CSS)
# ----------------------------

st.markdown(
    """
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

    /* Table Styling */
    .matrix-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; color: #E5E7EB; border: 1px solid #374151; border-radius: 8px; overflow: hidden; }
    .matrix-table th { background-color: #1F2937; text-align: left; padding: 12px 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #9CA3AF; border-bottom: 1px solid #374151; }
    .matrix-table td { padding: 12px 16px; border-bottom: 1px solid #2A2E39; background-color: rgba(30, 34, 45, 0.4); }
    .matrix-table tr:hover td { background-color: rgba(59, 130, 246, 0.05); }
    .strat-name { font-weight: 700; color: #60A5FA; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# 3) DATA MODELS
# ----------------------------

@dataclass
class DataHealth:
    fetched_at_utc: str
    valid_keys: List[str]
    invalid_keys: List[str]
    proxy_used: Dict[str, str]
    errors: Dict[str, str]
    note: Optional[str] = None


# ----------------------------
# 4) DATA ENGINE
# ----------------------------

def _extract_close_series(df_batch: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
    if df_batch is None or df_batch.empty: return None
    try:
        if isinstance(df_batch.columns, pd.MultiIndex):
            if symbol not in df_batch.columns.levels[0]: return None
            s = df_batch[symbol].get("Close", df_batch[symbol].get("Adj Close"))
            return s.dropna() if s is not None else None
        return df_batch["Close"].dropna() if "Close" in df_batch.columns else df_batch["Adj Close"].dropna()
    except Exception: return None

def _weekly_wtd_reference(series: pd.Series) -> Tuple[float, float]:
    curr = float(series.iloc[-1])
    weekly = series.resample("W-FRI").last().dropna()
    if len(weekly) >= 2:
        return curr, float(weekly.iloc[-2])
    prev = float(series.iloc[-6]) if len(series) >= 6 else float(series.iloc[-2])
    return curr, (prev if prev != 0 else curr)

@st.cache_data(ttl=300)
def fetch_market_data() -> Tuple[Dict[str, dict], pd.DataFrame, Dict[str, pd.Series], DataHealth, pd.DataFrame]:
    fetched_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    all_symbols = sorted(set(list(TICKERS.values()) + list(PROXIES.values())))
    try:
        df_batch = yf.download(tickers=all_symbols, period="1y", interval="1d", group_by="ticker", threads=True, progress=False)
    except Exception as e:
        return {}, pd.DataFrame(), {}, DataHealth(fetched_at, [], list(TICKERS.keys()), {}, {"err": str(e)}), pd.DataFrame()

    data, hist_map, full_hist, proxy_map = {}, {}, {}, {}
    for key, sym in TICKERS.items():
        used_sym, source = sym, "primary"
        s = _extract_close_series(df_batch, sym)
        if (s is None or len(s) < 50) and key in PROXIES:
            s, used_sym, source = _extract_close_series(df_batch, PROXIES[key]), PROXIES[key], "proxy"
            proxy_map[key] = used_sym
        
        if s is not None and len(s) >= 2:
            s = s.sort_index()
            hist_map[key] = s; full_hist[key] = s
            curr, prev = float(s.iloc[-1]), float(s.iloc[-2])
            curr_w, prev_w = _weekly_wtd_reference(s)
            
            if key == "US10Y":
                p, cd, cw, f = curr/10.0, (curr-prev)*10.0, (curr_w-prev_w)*10.0, "bps"
            else:
                p, cd, cw, f = curr, ((curr-prev)/prev)*100, ((curr_w-prev_w)/prev_w)*100 if prev_w!=0 else 0, "%"
            
            data[key] = {"price": p, "change": float(cd), "change_w": float(cw), "fmt": f, "valid": True, "source": source, "symbol_used": used_sym}
        else:
            data[key] = {"valid": False}

    health = DataHealth(fetched_at, [k for k,v in data.items() if v.get("valid")], [k for k,v in data.items() if not v.get("valid")], proxy_map, {})
    return data, pd.DataFrame(hist_map), full_hist, health, df_batch


# ----------------------------
# 5) STRATEGY DATABASE (FIXED)
# ----------------------------

STRATEGIES = {
    "GOLDILOCKS": {
        "desc": "Low Vol + Steady Trend. Market climbing wall of worry.",
        "risk": "1.5%", "bias": "Long",
        "index": {"strat": "Directional Diagonal", "dte": "Front 17 / Back 31", "setup": "Buy Back ITM (70D) / Sell Front OTM (30D)", "notes": "Stock replacement. Trend (Delta) + Decay (Theta). Upside is uncapped."},
        "stock": {"strat": "Call Debit Spreads", "dte": "45-60 DTE", "setup": "Buy 60D / Sell 30D (Spread)", "notes": "Focus on Relative Strength Leaders (Tech, Semis). Use pullbacks to EMA21."},
        "longs": "TECH, SEMIS, DISC", "shorts": "VIX, TLT",
    },
    "LIQUIDITY": {
        "desc": "High Liquidity / Dollar Weakness. Drift Up environment.",
        "risk": "1.0%", "bias": "Aggressive Long",
        "index": {"strat": "Flyagonal (Drift)", "dte": "Entry 7-10 DTE", "setup": "Upside: Call BWB (Long +10 / Short 2x +50 / Long +60). Downside: Put Diagonal.", "notes": "Captures the drift. Upside tent funds the downside floor. Target 4% Flash Win."},
        "stock": {"strat": "Risk Reversals", "dte": "60 DTE", "setup": "Sell OTM Put / Buy OTM Call", "notes": "Funding long delta with short volatility. Best for High Beta (Crypto proxies)."},
        "longs": "BTC, SEMIS, QQQ", "shorts": "DXY, CASH",
    },
    "REFLATION": {
        "desc": "Inflation / Rates Rising. Real Assets outperform Tech.",
        "risk": "1.0%", "bias": "Cyclical Long",
        "index": {"strat": "Call Spread (Cyclicals)", "dte": "45 DTE", "setup": "Buy 60D / Sell 30D", "notes": "Focus on Russell 2000 (IWM). Avoid long duration Tech (QQQ) as rates rise."},
        "stock": {"strat": "Cash Secured Puts", "dte": "30-45 DTE", "setup": "Sell 30D Puts on Energy/Banks", "notes": "Energy (XLE) and Banks (XLF) benefit from rising yields. Sell premium to acquire."},
        "longs": "ENERGY, BANKS, IND", "shorts": "TLT, TECH",
    },
    "NEUTRAL": {
        "desc": "Chop / Range Bound. No clear direction.",
        "risk": "Income Size", "bias": "Neutral/Theta",
        "index": {"strat": "TimeEdge (SPX) / TimeZone (RUT)", "dte": "Entry 15 / Exit 7", "setup": "Put Calendar Spread (ATM) or Double Calendar", "notes": "Pure Theta play. Sell 15 DTE / Buy 22+ DTE. Requires VIX < 20."},
        "stock": {"strat": "Iron Condor", "dte": "30-45 DTE", "setup": "Sell 20D Call / Sell 20D Put (Wings 10 wide)", "notes": "Delta neutral income. Best on low beta stocks (KO, PEP) during chop."},
        "longs": "INCOME, CASH", "shorts": "MOMENTUM",
    },
    "RISK OFF": {
        "desc": "High Volatility / Credit Stress. Preservation mode.",
        "risk": "0.5%", "bias": "Short/Hedge",
        "index": {"strat": "A14 Put BWB", "dte": "Entry 14 / Exit 7", "setup": "Long ATM / Short 2x -40 / (Skip) / Long -60", "notes": "Crash Catcher. Zero upside risk. Profit tent expands into the crash. Enter Friday AM."},
        "stock": {"strat": "Put Debit Spreads", "dte": "60 DTE", "setup": "Buy 40D / Sell 15D", "notes": "Directional downside. Selling the 15D put reduces cost and offsets IV crush."},
        "longs": "VIX, DXY", "shorts": "SPY, IWM, HYG",
    },
    "DATA ERROR": {
        "desc": "CRITICAL DATA FEED FAILURE", "risk": "0.0%", "bias": "Flat",
        "index": {"strat": "STAND ASIDE", "dte": "--", "setup": "--", "notes": "Do not trade. Data integrity compromised."},
        "stock": {"strat": "STAND ASIDE", "dte": "--", "setup": "--", "notes": "Do not trade. Data integrity compromised."},
        "longs": "--", "shorts": "--",
    },
}


# ----------------------------
# 6) HELPERS & LOGIC
# ----------------------------

def calculate_adx(df_ohlc: pd.DataFrame, period: int = 14) -> float:
    try:
        df = df_ohlc.copy()
        df['tr'] = np.maximum(df['High']-df['Low'], np.maximum(abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))))
        df['+dm'] = np.where((df['High']-df['High'].shift(1) > df['Low'].shift(1)-df['Low']) & (df['High']-df['High'].shift(1) > 0), df['High']-df['High'].shift(1), 0)
        df['-dm'] = np.where((df['Low'].shift(1)-df['Low'] > df['High']-df['High'].shift(1)) & (df['Low'].shift(1)-df['Low'] > 0), df['Low'].shift(1)-df['Low'], 0)
        def s(x, n):
            out = np.zeros_like(x); out[n] = x[1:n+1].mean()
            for i in range(n+1, len(x)): out[i] = (out[i-1]*(n-1)+x[i])/n
            return out
        trs, dmp, dmm = s(df['tr'].values, period), s(df['+dm'].values, period), s(df['-dm'].values, period)
        dip, dim = 100*(dmp/trs), 100*(dmm/trs)
        dx = 100*abs(dip-dim)/(dip+dim)
        return float(s(np.nan_to_num(dx), period)[-1])
    except: return 20.0

def determine_regime(data, full_hist, timeframe, strict):
    missing = [k for k in CRITICAL_KEYS if not data.get(k,{}).get("valid")]
    if missing: return "DATA ERROR", "NONE", [f"Missing: {missing}"]
    conf = "LOW" if any(data[k].get("source")=="proxy" for k in CRITICAL_KEYS) else "HIGH"
    if conf=="LOW" and strict: return "DATA ERROR", "NONE", ["Proxy in strict mode"]
    
    def g(k): 
        v = data[k].get("change" if timeframe=="Tactical (Daily)" else "change_w", 0)
        return float(v) if np.isfinite(v) else 0
    
    hyg, vix_c, us10y, dxy, btc = g("HYG"), g("VIX"), g("US10Y"), g("DXY"), g("BTC")
    vix_p = data["VIX"].get("price", 20.0)
    
    v_thresh = 5.0
    if "VIX" in full_hist:
        v_thresh = max(5.0, 2.0 * (full_hist["VIX"].pct_change()*100).rolling(20).std().iloc[-1])

    if hyg < -0.5 or vix_c > v_thresh: return "RISK OFF", conf, []
    if dxy < -0.3 and btc > 1.5: return "LIQUIDITY", conf, []
    if (vix_c < 0 or vix_p < 15) and abs(us10y) < 5 and hyg > -0.1: return "GOLDILOCKS", conf, []
    return "NEUTRAL", conf, []


# ----------------------------
# 7) VISUAL WRAPPERS
# ----------------------------

def _sankey_from_values(title, values, color):
    df = pd.DataFrame(list(values.items()), columns=["id", "val"]).sort_values("val", ascending=False)
    if len(df) < 4: return go.Figure().add_annotation(text="Insufficient Data", showarrow=False)
    win, los = df.head(3), df.tail(3)
    labs = list(los["id"]) + list(win["id"])
    src, tgt, vls = [], [], []
    for i in range(3):
        for j in range(3):
            src.append(i); tgt.append(3+j); vls.append(abs(los.iloc[i]["val"]) + abs(win.iloc[j]["val"]))
    fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, label=labs, color=["#ef4444"]*3 + ["#22c55e"]*3), link=dict(source=src, target=tgt, value=vls, color=color))])
    fig.update_layout(title_text=title, font=dict(color="white"), paper_bgcolor="rgba(0,0,0,0)", height=350)
    return fig

def plot_sankey_sectors(data, timeframe):
    keys = ["TECH", "SEMIS", "BANKS", "ENERGY", "HOME", "UTIL", "HEALTH", "MAT", "COMM"]
    vals = {k: data[k].get("change" if timeframe=="Tactical (Daily)" else "change_w", 0) for k in keys if data.get(k,{}).get("valid")}
    return _sankey_from_values(f"Sector Rotation ({timeframe})", vals, "rgba(59, 130, 246, 0.2)")

def plot_sankey_assets(data, timeframe):
    keys = ["SPY", "TLT", "DXY", "GOLD", "BTC", "OIL", "HYG"]
    vals = {k: data[k].get("change" if timeframe=="Tactical (Daily)" else "change_w", 0) for k in keys if data.get(k,{}).get("valid")}
    return _sankey_from_values(f"Asset Rotation ({timeframe})", vals, "rgba(168, 85, 247, 0.2)")

def plot_nexus_graph_dots(data, timeframe):
    nodes = {"US10Y": (0,0), "DXY": (0.8,0.8), "SPY": (-0.8,0.8), "QQQ": (-1.2,0.4), "GOLD": (0.8,-0.8), "HYG": (-0.4,-0.8), "BTC": (-1.5,1.5), "VIX": (0,1.5)}
    edges = [("US10Y", "QQQ"), ("US10Y", "GOLD"), ("HYG", "SPY"), ("DXY", "GOLD"), ("VIX", "SPY")]
    edge_x, edge_y = [], []
    for u, v in edges:
        edge_x.extend([nodes[u][0], nodes[v][0], None]); edge_y.extend([nodes[u][1], nodes[v][1], None])
    nx, ny, nt, nc = [], [], [], []
    for k, p in nodes.items():
        nx.append(p[0]); ny.append(p[1]); nt.append(k)
        v = data.get(k,{}).get("change" if timeframe=="Tactical (Daily)" else "change_w", 0)
        nc.append("#22c55e" if v > 0 else "#ef4444" if v < 0 else "#6b7280")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#4b5563"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=nx, y=ny, mode="markers+text", text=nt, marker=dict(size=40, color=nc, line=dict(width=2, color="white")), textfont=dict(color="white")))
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(b=0,l=0,r=0,t=0))
    return fig

def plot_trend_momentum_quadrant(full_hist, category, timeframe, relative=False):
    keys = ["TECH", "SEMIS", "BANKS", "ENERGY", "HOME"] if category=="SECTORS" else ["SPY", "QQQ", "IWM", "GOLD", "BTC", "TLT", "DXY", "HYG"]
    tw, mw = (20, 5) if timeframe=="Tactical (Daily)" else (100, 25)
    spy_s, items = full_hist.get("SPY"), []
    for k in keys:
        s = full_hist.get(k)
        if s is None or len(s) < (tw+mw+5): continue
        if relative and spy_s is not None:
            ci = s.index.intersection(spy_s.index)
            s = s.loc[ci] / spy_s.loc[ci]
        curr, sma = float(s.iloc[-1]), float(s.rolling(tw).mean().iloc[-1])
        if sma==0: continue
        ts = ((curr/sma)-1)*100; pm = float(s.iloc[-(mw+1)]); ms = (((curr/pm)-1)*100) if pm!=0 else 0
        c = "#22c55e" if ts>0 and ms>0 else "#3b82f6" if ts<0 and ms>0 else "#f59e0b" if ts>0 and ms<0 else "#ef4444"
        items.append({"Symbol": k, "Trend": ts, "Momentum": ms, "Color": c})
    df = pd.DataFrame(items)
    if df.empty: return go.Figure()
    fig = px.scatter(df, x="Trend", y="Momentum", text="Symbol", color="Color", color_discrete_map="identity")
    lm = max(float(df["Trend"].abs().max()), float(df["Momentum"].abs().max())) * 1.1
    fig.update_layout(xaxis=dict(range=[-lm, lm]), yaxis=dict(range=[-lm, lm]), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), showlegend=False, height=450)
    fig.add_hline(y=0, line_dash="dot"); fig.add_vline(x=0, line_dash="dot")
    return fig

def plot_correlation_heatmap(history_df, timeframe):
    if history_df.empty: return go.Figure()
    df = history_df.dropna(how='all').ffill(limit=2)
    if timeframe != "Tactical (Daily)": df = df.resample("W-FRI").last().dropna(how='all')
    corr = df.pct_change().corr()
    subset = ["US10Y", "DXY", "VIX", "HYG", "SPY", "QQQ", "IWM", "BTC", "GOLD", "OIL"]
    cols = [c for c in subset if c in corr.columns]
    fig = px.imshow(corr.loc[cols, cols], text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), height=400)
    return fig


# ----------------------------
# 8) MAIN APP
# ----------------------------

def main():
    st.title(APP_VERSION)
    with st.sidebar:
        st.subheader("Data Controls")
        strict = st.checkbox("Strict mode", value=False)
        st.divider(); st.write("- Weekly: WTD vs last Friday close."); st.info("Drivers (VIX, Rates, DXY) Red if rising (Risk Off).")

    with st.spinner("Connecting to MacroNexus Core..."):
        market_data, history_df, full_hist, health, raw_batch = fetch_market_data()

    if datetime.datetime.utcnow().weekday() >= 5: st.info("📅 Weekend Mode Active: Data reflects Friday's close.")

    # Metric Bar
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def metric_tile(col, label, key):
        d = market_data.get(key, {})
        if not d.get("valid"): col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">--</div></div>', unsafe_allow_html=True); return
        val, chg = d.get("price", 0.0), d.get("change", 0.0)
        color = ("#F43F5E" if chg > 0 else "#10B981") if key in ["US10Y", "DXY", "VIX"] else ("#10B981" if chg > 0 else "#F43F5E")
        vs, cs = (f"{val:.2f}%", f"{chg:+.1f} bps") if key=="US10Y" else (f"{val:.2f}", f"{chg:+.2f}%")
        col.markdown(f'<div class="metric-card" style="border-left: 3px solid {color};"><div class="metric-label">{label}{" (proxy)" if d.get("source")=="proxy" else ""}</div><div class="metric-value">{vs}<span class="metric-delta" style="color:{color};">{cs}</span></div></div>', unsafe_allow_html=True)
    metric_tile(c1, "Credit (HYG)", "HYG"); metric_tile(c2, "Volatility (VIX)", "VIX"); metric_tile(c3, "10Y Yield", "US10Y"); metric_tile(c4, "Dollar (DXY)", "DXY"); metric_tile(c5, "Oil", "OIL"); metric_tile(c6, "Bitcoin", "BTC")

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl2: timeframe = st.selectbox("Analytic View", TIMEFRAMES, index=0, label_visibility="collapsed")
    regime, conf, reasons = determine_regime(market_data, full_hist, timeframe, strict)
    with ctrl1: active_regime = st.selectbox("Force Regime", list(STRATEGIES.keys()), label_visibility="collapsed") if st.checkbox("Manual Override", value=False) else regime
    with ctrl3:
        rc = {"GOLDILOCKS": "#10B981", "LIQUIDITY": "#A855F7", "REFLATION": "#F59E0B", "NEUTRAL": "#6B7280", "RISK OFF": "#EF4444"}.get(active_regime, "#6B7280")
        st.markdown(f'<div style="text-align: right; display: flex; align-items: center; justify-content: flex-end; gap: 15px;"><div style="text-align: right;"><div style="font-size: 10px; color: #8B9BB4;">SYSTEM STATUS</div><div style="font-size: 14px; font-weight: bold; color:{rc};">{"LOW CONF" if conf=="LOW" else "ACTIVE"}</div></div><div class="regime-badge" style="background:{rc}22; color:{rc}; border:1px solid {rc};">{active_regime}</div></div>', unsafe_allow_html=True)

    tabs = st.tabs(["🚀 MISSION CONTROL", "🚦 DECISION MATRIX", "📋 WORKFLOW", "📊 MARKET PULSE", "🕸️ MACRO MACHINE", "📖 STRATEGY PLAYBOOK"])

    with tabs[0]: # Mission Control
        with st.expander("🎛️ SPX Income Reactor Telemetry", expanded=True):
            tc1, tc2, tc3, tc4 = st.columns(4)
            asset_mode = tc1.radio("Asset Class", ["INDEX", "STOCKS"], horizontal=True)
            iv_rank, skew_rank = tc2.slider("IV Rank", 0, 100, 45), tc3.slider("Skew Rank", 0, 100, 50)
            adx_val = calculate_adx(raw_batch["SPY"].dropna()) if not raw_batch.empty else 20.0
            tc4.metric("ADX (Auto, SPY 14)", f"{adx_val:.1f}")
        
        strat = STRATEGIES.get(active_regime, STRATEGIES["NEUTRAL"])
        react = strat["stock"] if asset_mode=="STOCKS" else (STRATEGIES["RISK OFF"]["index"] if active_regime=="RISK OFF" else ({"strat":"Put BWB","dte":"21-30D","setup":"ATM/-40/-60","notes":"High Skew Protection"} if skew_rank>80 else strat["index"]))

        cl, cr = st.columns([1, 2])
        with cl: st.markdown(f'<div class="strat-card"><div class="strat-header"><div class="strat-title" style="color:{rc}">CONTEXT</div><div class="strat-tag" style="background:{rc}22; color:{rc}">{active_regime}</div></div><div class="strat-section"><div class="strat-subtitle">DESCRIPTION</div><div class="strat-data">{strat["desc"]}</div></div><div class="strat-section"><div class="strat-subtitle">RISK SIZE</div><div class="strat-data" style="font-size: 24px; color:{rc}">{strat["risk"]}</div></div><div class="strat-section"><div class="strat-subtitle">BIAS</div><div class="strat-data">{strat["bias"]}</div></div><div style="margin-top:20px; padding-top:15px; border-top:1px solid rgba(255,255,255,0.1);"><div style="margin-bottom:8px;"><span class="badge-green">TARGETS</span> <span style="font-size:13px;">{strat.get("longs")}</span></div><div><span class="badge-red">AVOID</span> <span style="font-size:13px;">{strat.get("shorts")}</span></div></div></div>', unsafe_allow_html=True)
        with cr: st.markdown(f'<div class="strat-card" style="border-color:{rc};"><div class="strat-header"><div class="strat-title" style="color:{rc};">TACTICAL EXECUTION</div></div><div style="font-size:28px; font-weight:800; margin-bottom:20px; color:#FFF;">{react["strat"]}</div><div style="display:grid; grid-template-columns:1fr 1fr; gap:30px; margin-bottom:25px;"><div><div class="strat-subtitle">⏱️ TIMING (DTE)</div><div class="strat-data" style="font-weight:700; font-size:18px;">{react["dte"]}</div></div><div><div class="strat-subtitle">🏗️ STRUCTURE</div><div class="strat-data" style="font-weight:700; font-size:18px;">{react["setup"]}</div></div></div><div style="background:rgba(0,0,0,0.3); padding:15px; border-radius:8px; border-left:4px solid {rc};"><div class="strat-subtitle" style="margin-top:0; color:{rc};">🧠 THE LOGIC</div><div class="strat-data" style="font-size:14px; font-style:italic;">"{react["notes"]}"</div></div></div>', unsafe_allow_html=True)

    with tabs[1]: # Decision Matrix
        st.subheader(f"🚦 Decision Matrix: {active_regime}")
        def badge(t, c): return f'<span class="badge-{c}">{t}</span>'
        if active_regime=="GOLDILOCKS":
            rows = [("Tech/Growth", "Call Debit Spreads", badge("OPEN", "green"), "Low vol + Up trend."), ("Broad Market", "Directional Diagonals", badge("OPEN", "green"), "Stock replacement."), ("Small Caps", "Put Credit Spreads", badge("HOLD", "yellow"), "Participation check."), ("Defensives", "Covered Calls", badge("TRIM", "red"), "Rotate to beta.")]
        elif active_regime=="RISK OFF":
            rows = [("Broad Market", "A14 Crash Catcher", badge("OPEN", "green"), "Panic hedge."), ("Safe Haven", "TLT / Dollar", badge("OPEN", "green"), "Flight to safety."), ("Volatility", "Long VIX Calls", badge("OPEN", "green"), "Consistent upside.")]
        elif active_regime=="LIQUIDITY":
            rows = [("Tech/Growth", "Risk Reversals", badge("OPEN", "green"), "Dollar down = Tech up."), ("Broad Market", "Flyagonal", badge("OPEN", "green"), "Capture drift."), ("Safe Haven", "Gold Longs", badge("OPEN", "green"), "Dollar debasement.")]
        else: rows = [("General", "Standard Tactics", badge("HOLD", "yellow"), "No extreme signal.")]
        html = "".join([f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>" for r in rows])
        st.markdown(f'<table class="matrix-table"><thead><tr><th>Asset</th><th>Strat</th><th>Signal</th><th>Logic</th></tr></thead><tbody>{html}</tbody></table>', unsafe_allow_html=True)

    with tabs[2]: # Workflow
        st.subheader("📋 The 5-Phase Mission Control Workflow")
        w1, w2, w3, w4, w5 = st.columns(5)
        w1.markdown("""<div class="metric-card" style="height: 250px;"><div style="color: #F87171; font-weight: bold; margin-bottom: 10px;">PHASE 1: VETO</div><div style="font-size: 12px; color: #AAA;">1. Check <b>HYG</b> (Credit). Is it crashing (&lt; -0.5%)?<br>2. Check <b>VIX</b>. Is it spiking (&gt; 5%)?<br><br><span style="color: #F87171;">If YES: STOP. Go to Risk Off.</span></div></div>""", unsafe_allow_html=True)
        w2.markdown("""<div class="metric-card" style="height: 250px;"><div style="color: #FBBF24; font-weight: bold; margin-bottom: 10px;">PHASE 2: REGIME</div><div style="font-size: 12px; color: #AAA;">Identify the "Tailwind".<br>• <b>Goldilocks:</b> Growth + Low Vol<br>• <b>Liquidity:</b> DXY Down + Crypto Up<br>• <b>Reflation:</b> Yields + Oil Up</div></div>""", unsafe_allow_html=True)
        w3.markdown("""<div class="metric-card" style="height: 250px;"><div style="color: #60A5FA; font-weight: bold; margin-bottom: 10px;">PHASE 3: SECTOR</div><div style="font-size: 12px; color: #AAA;">Use the Quadrant & Sankey charts.<br>• Find sectors moving from <b>Improving</b> to <b>Leading</b>.<br>• Confirm flows match Regime.</div></div>""", unsafe_allow_html=True)
        w4.markdown("""<div class="metric-card" style="height: 250px;"><div style="color: #A78BFA; font-weight: bold; margin-bottom: 10px;">PHASE 4: TACTICS</div><div style="font-size: 12px; color: #AAA;">Consult the <b>SPX Reactor</b>.<br>• <b>Vol Check:</b> IV Rank &gt; 50?<br>• <b>Skew Check:</b> Crash risk?<br>• <b>Trend Check:</b> ADX &gt; 25?</div></div>""", unsafe_allow_html=True)
        w5.markdown("""<div class="metric-card" style="height: 250px;"><div style="color: #34D399; font-weight: bold; margin-bottom: 10px;">PHASE 5: EXECUTE</div><div style="font-size: 12px; color: #AAA;">3:00 PM EST Check.<br>• Confirm price action.<br>• Verify DTE matches plan.<br>• <b>Enter trade.</b></div></div>""", unsafe_allow_html=True)

    with tabs[3]: # Pulse
        st.subheader("🌊 Capital Flow")
        ps1, ps2 = st.columns([3, 1])
        with ps1: 
            st.plotly_chart(plot_sankey_sectors(market_data, timeframe), use_container_width=True)
            st.plotly_chart(plot_sankey_assets(market_data, timeframe), use_container_width=True)
        with ps2: st.markdown('<div class="context-box"><div class="context-header">How to Read</div>Sankey diagrams visualize rotation from losers (left) to winners (right).<br><br><div class="context-header">Analyst Note</div>If defensives lead, risk appetite is low. Avoid high-beta longs.</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_correlation_heatmap(history_df, timeframe), use_container_width=True)
        is_rrg = st.toggle("RRG Mode (vs SPY)", value=False)
        q1, q2 = st.columns(2)
        with q1: st.plotly_chart(plot_trend_momentum_quadrant(full_hist, "SECTORS", timeframe, is_rrg), use_container_width=True)
        with q2: st.plotly_chart(plot_trend_momentum_quadrant(full_hist, "ASSETS", timeframe, is_rrg), use_container_width=True)

    with tabs[4]: # Macro
        st.subheader("🕸️ The Macro Machine")
        mg1, mg2 = st.columns([3, 1])
        with mg1: st.plotly_chart(plot_nexus_graph_dots(market_data, timeframe), use_container_width=True)
        with mg2: st.markdown('<div class="context-box"><div class="context-header">Key Correlations</div>• <b>Rates ➔ Tech:</b> Inverse (Long duration).<br>• <b>Credit ➔ Equities:</b> Leading indicator (Credit precedes equity moves).</div>', unsafe_allow_html=True)

    with tabs[5]: # Playbook
        st.subheader("📚 Strategy Playbook")
        with st.expander("🔴 A14 (Crash Protection)"):
            st.markdown("""### A14: The "Anti-Fragile" Hedge\n**Concept:** Financing downside protection using OTM puts.\n\n**1. Setup (Put BWB)**\n- Long: 1x ATM Put\n- Short: 2x OTM Puts (-40 pts)\n- Long: 1x OTM Put (-60 pts)\n\n**2. Entry**\n- Friday AM, 14 DTE.\n\n**3. Logic**\nZero upside risk; profit tent expands into crash.""")
        with st.expander("🟣 TIMEZONE (RUT Income)"):
            st.markdown("""### TimeZone: High Prob RUT Income\n- **Logic:** Harvest theta on Russell using a hedged calendar structure.\n- **Entry:** Thursday 3 PM, 15 DTE front month.""")
        with st.expander("🔵 TIMEEDGE (SPX Neutral)"):
            st.markdown("""### TimeEdge: Pure Theta Decay\n- **Logic:** Double calendar (sell 15 DTE / buy 43 DTE).\n- **Exit:** Hard stop at 10% profit/loss.""")
        with st.expander("🌊 FLYAGONAL (Liquidity Drift)"):
            st.markdown("""### Flyagonal: The Drift Catcher\n- **Logic:** Capture melt-up higher using call BWB and put diagonals.""")


if __name__ == "__main__":
    main()
