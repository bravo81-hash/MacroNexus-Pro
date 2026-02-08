import datetime
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ----------------------------
# 0) BASIC CONFIG / CONSTANTS
# ----------------------------
APP_VERSION = "MacroNexus Pro (Visual Ops)"

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

# PROXIES (Smart Degradation)
PROXIES: Dict[str, str] = {
    "DXY": "UUP",    # USD Bull ETF
    "VIX": "VIXY",   # VIX Short-Term Futures ETF
    "RUT": "IWM",    # Russell 2000 ETF
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
    
    /* Control Panel */
    .control-container {
        background-color: #161920;
        border: 1px solid #2A2E39;
        border-radius: 10px;
        padding: 15px 25px;
        margin-bottom: 25px;
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
""",
    unsafe_allow_html=True,
)

# ----------------------------
# 3) LOGGING
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("macronexus")

# ----------------------------
# 4) DATA MODELS
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
# 5) DATA ENGINE
# ----------------------------
def _safe_is_multiindex_columns(df: pd.DataFrame) -> bool:
    return isinstance(df.columns, pd.MultiIndex)

def _extract_close_series(df_batch: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
    if df_batch is None or df_batch.empty:
        return None
    try:
        if _safe_is_multiindex_columns(df_batch):
            top = df_batch.columns.get_level_values(0)
            if symbol not in set(top):
                return None
            cols_lvl1 = set(df_batch[symbol].columns)
            if "Close" in cols_lvl1:
                s = df_batch[symbol]["Close"].dropna()
            elif "Adj Close" in cols_lvl1:
                s = df_batch[symbol]["Adj Close"].dropna()
            else:
                return None
            return s if isinstance(s, pd.Series) else None
        
        if "Close" in df_batch.columns:
            s = df_batch["Close"].dropna()
            return s if isinstance(s, pd.Series) else None
        if "Adj Close" in df_batch.columns:
            s = df_batch["Adj Close"].dropna()
            return s if isinstance(s, pd.Series) else None
        return None
    except Exception:
        return None

def _weekly_wtd_reference(series: pd.Series) -> Tuple[float, float]:
    curr = float(series.iloc[-1])
    if len(series) < 10:
        prev = float(series.iloc[-2]) if len(series) >= 2 else curr
        return curr, prev
        
    weekly = series.resample("W-FRI").last().dropna()
    if len(weekly) >= 2:
        last_date = series.index[-1]
        # Check if last date matches the weekly bucket end (Friday)
        is_friday_close = (getattr(last_date, "weekday", lambda: -1)() == 4) and (weekly.index[-1].date() == last_date.date())
        if is_friday_close:
            prev_friday = float(weekly.iloc[-2])
            return curr, prev_friday
        prev_friday = float(weekly.iloc[-2])
        return curr, prev_friday
        
    prev_approx = float(series.iloc[-6]) if len(series) >= 6 else float(series.iloc[-2])
    return curr, prev_approx

@st.cache_data(ttl=300)
def fetch_market_data() -> Tuple[Dict[str, dict], pd.DataFrame, Dict[str, pd.Series], DataHealth]:
    fetched_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    all_symbols = sorted(set(list(TICKERS.values()) + list(PROXIES.values())))
    
    try:
        df_batch = yf.download(
            tickers=all_symbols,
            period="1y",
            interval="1d",
            group_by="ticker",
            threads=True,
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        health = DataHealth(fetched_at, [], list(TICKERS.keys()), {}, {"__download__": str(e)}, "API exception")
        return {}, pd.DataFrame(), {}, health

    if df_batch is None or df_batch.empty:
        health = DataHealth(fetched_at, [], list(TICKERS.keys()), {}, {"__download__": "No data returned"}, "Empty download")
        return {}, pd.DataFrame(), {}, health

    data: Dict[str, dict] = {}
    history_data: Dict[str, pd.Series] = {}
    full_hist: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}
    proxy_used: Dict[str, str] = {}

    for key, primary_symbol in TICKERS.items():
        used_symbol = primary_symbol
        source = "primary"
        s = _extract_close_series(df_batch, primary_symbol)
        
        if (s is None or len(s) < 50) and key in PROXIES:
            proxy_symbol = PROXIES[key]
            s_proxy = _extract_close_series(df_batch, proxy_symbol)
            if s_proxy is not None and len(s_proxy) >= 50:
                s = s_proxy
                used_symbol = proxy_symbol
                source = "proxy"
                proxy_used[key] = proxy_symbol
        
        if s is None or len(s) < 2:
            data[key] = {"valid": False, "symbol_primary": primary_symbol, "symbol_used": used_symbol, "source": source}
            continue
            
        try:
            s = s.sort_index()
            if not isinstance(s.index, pd.DatetimeIndex):
                continue
        except Exception:
            continue

        history_data[key] = s
        full_hist[key] = s
        curr = float(s.iloc[-1])
        prev = float(s.iloc[-2])
        curr_w, prev_w = _weekly_wtd_reference(s)
        
        if key == "US10Y":
            display_price = curr / 10.0
            chg_d = (curr - prev) * 10.0
            chg_w = (curr_w - prev_w) * 10.0
            fmt = "bps"
        else:
            display_price = curr
            chg_d = ((curr - prev) / prev) * 100.0
            chg_w = ((curr_w - prev_w) / prev_w) * 100.0 if prev_w != 0 else 0.0
            fmt = "%"
            
        data[key] = {
            "price": display_price,
            "change": float(chg_d),
            "change_w": float(chg_w),
            "fmt": fmt,
            "valid": True,
            "symbol_primary": primary_symbol,
            "symbol_used": used_symbol,
            "source": source,
        }

    valid_keys = [k for k, v in data.items() if v.get("valid")]
    invalid_keys = [k for k, v in data.items() if not v.get("valid")]
    
    health = DataHealth(fetched_at, valid_keys, invalid_keys, proxy_used, errors, "Weekly is WTD vs last Fri")
    history_df = pd.DataFrame(history_data)
    return data, history_df, full_hist, health

# ----------------------------
# 6) STRATEGY DATABASE
# ----------------------------
STRATEGIES = {
    "GOLDILOCKS": {
        "desc": "Low Vol + Steady Trend. Market climbing wall of worry.",
        "risk": "1.5%", "bias": "Long",
        "index": {"strat": "Directional Diagonal", "dte": "Front 17 / Back 31", "setup": "Buy Back ITM (70D) / Sell Front OTM (30D)", "notes": "Trend (Delta) + Decay (Theta)."},
        "stock": {"strat": "Call Debit Spreads", "dte": "45-60 DTE", "setup": "Buy 60D / Sell 30D (Spread)", "notes": "Focus on Relative Strength Leaders (Tech, Semis)."},
        "longs": "TECH, SEMIS, DISC", "shorts": "VIX, TLT",
    },
    "LIQUIDITY": {
        "desc": "High Liquidity / Dollar Weakness. Drift Up environment.",
        "risk": "1.0%", "bias": "Aggressive Long",
        "index": {"strat": "Flyagonal (Drift)", "dte": "Entry 7-10 DTE", "setup": "Upside: Call BWB. Downside: Put Diagonal.", "notes": "Captures the drift. Upside tent funds the downside floor."},
        "stock": {"strat": "Risk Reversals", "dte": "60 DTE", "setup": "Sell OTM Put / Buy OTM Call", "notes": "Funding long delta with short volatility. Best for High Beta."},
        "longs": "BTC, SEMIS, QQQ", "shorts": "DXY, CASH",
    },
    "REFLATION": {
        "desc": "Inflation / Rates Rising. Real Assets outperform Tech.",
        "risk": "1.0%", "bias": "Cyclical Long",
        "index": {"strat": "Call Spread (Cyclicals)", "dte": "45 DTE", "setup": "Buy 60D / Sell 30D", "notes": "Focus on Russell 2000 (IWM). Avoid long duration Tech."},
        "stock": {"strat": "Cash Secured Puts", "dte": "30-45 DTE", "setup": "Sell 30D Puts on Energy/Banks", "notes": "Energy (XLE) and Banks (XLF) benefit from rising yields."},
        "longs": "ENERGY, BANKS, IND", "shorts": "TLT, TECH",
    },
    "NEUTRAL": {
        "desc": "Chop / Range Bound. No clear direction.",
        "risk": "Income Size", "bias": "Neutral/Theta",
        "index": {"strat": "TimeEdge (SPX) / TimeZone (RUT)", "dte": "Entry 15 / Exit 7", "setup": "Put Calendar Spread (ATM) or Double Calendar", "notes": "Pure Theta play. Requires VIX < 20."},
        "stock": {"strat": "Iron Condor", "dte": "30-45 DTE", "setup": "Sell 20D Call / Sell 20D Put (Wings 10 wide)", "notes": "Delta neutral income. Best on low beta stocks."},
        "longs": "INCOME, CASH", "shorts": "MOMENTUM",
    },
    "RISK OFF": {
        "desc": "High Volatility / Credit Stress. Preservation mode.",
        "risk": "0.5%", "bias": "Short/Hedge",
        "index": {"strat": "A14 Put BWB", "dte": "Entry 14 / Exit 7", "setup": "Long ATM / Short 2x -40 / (Skip) / Long -60", "notes": "Crash Catcher. Zero upside risk. Profit tent expands into the crash."},
        "stock": {"strat": "Put Debit Spreads", "dte": "60 DTE", "setup": "Buy 40D / Sell 15D", "notes": "Directional downside. Selling the 15D put reduces cost."},
        "longs": "VIX, DXY", "shorts": "SPY, IWM, HYG",
    },
    "DATA ERROR": {
        "desc": "CRITICAL DATA FEED FAILURE",
        "risk": "0.0%", "bias": "Flat",
        "index": {"strat": "STAND ASIDE", "dte": "--", "setup": "--", "notes": "Data integrity compromised."},
        "stock": {"strat": "STAND ASIDE", "dte": "--", "setup": "--", "notes": "Data integrity compromised."},
        "longs": "--", "shorts": "--",
    },
}

# ----------------------------
# 7) HELPERS
# ----------------------------
def get_val(data: Dict[str, dict], key: str, timeframe: str) -> float:
    d = data.get(key, {})
    if not d.get("valid", False): return float("nan")
    if timeframe == "Tactical (Daily)": return float(d.get("change", 0.0))
    return float(d.get("change_w", 0.0))

def determine_regime(data: Dict[str, dict], timeframe: str, strict_data: bool) -> Tuple[str, str, List[str]]:
    reasons: List[str] = []
    
    missing_crit = [k for k in CRITICAL_KEYS if not data.get(k, {}).get("valid", False)]
    if missing_crit:
        reasons.append(f"Missing critical: {', '.join(missing_crit)}")
        return "DATA ERROR", "NONE", reasons
        
    proxy_crit = [k for k in CRITICAL_KEYS if data.get(k, {}).get("source") == "proxy"]
    if proxy_crit:
        reasons.append(f"Proxy used: {', '.join(proxy_crit)}")
        if strict_data:
            return "DATA ERROR", "NONE", reasons
        confidence = "LOW"
    else:
        confidence = "HIGH"
        
    def g(k: str) -> float:
        v = get_val(data, k, timeframe)
        return float(v) if np.isfinite(v) else 0.0
        
    hyg, vix = g("HYG"), g("VIX")
    oil, cop = g("OIL"), g("COPPER")
    us10y, dxy = g("US10Y"), g("DXY")
    btc, banks = g("BTC"), g("BANKS")
    
    if hyg < -0.5 or vix > 5.0: return "RISK OFF", confidence, reasons
    if (oil > 1.5 or cop > 1.5) and us10y > 3.0 and banks > 0: return "REFLATION", confidence, reasons
    if dxy < -0.3 and btc > 1.5: return "LIQUIDITY", confidence, reasons
    if vix < 0 and abs(us10y) < 5.0 and hyg > -0.1: return "GOLDILOCKS", confidence, reasons
        
    return "NEUTRAL", confidence, reasons

# ----------------------------
# 8) VISUALS
# ----------------------------
def plot_nexus_graph_dots(data: Dict[str, dict], timeframe: str) -> go.Figure:
    nodes = {
        "US10Y": {"pos": (0, 0), "label": "Rates (^TNX)"},
        "DXY": {"pos": (0.8, 0.8), "label": "Dollar (DXY)"},
        "SPY": {"pos": (-0.8, 0.8), "label": "S&P 500 (SPY)"},
        "QQQ": {"pos": (-1.2, 0.4), "label": "Nasdaq (QQQ)"},
        "GOLD": {"pos": (0.8, -0.8), "label": "Gold (GLD)"},
        "HYG": {"pos": (-0.4, -0.8), "label": "Credit (HYG)"},
        "BTC": {"pos": (-1.5, 1.5), "label": "Bitcoin"},
        "OIL": {"pos": (1.5, -0.4), "label": "Oil (USO)"},
        "COPPER": {"pos": (1.2, -1.2), "label": "Copper"},
        "IWM": {"pos": (-1.2, -1.0), "label": "Russell (IWM)"},
        "SEMIS": {"pos": (-1.8, 0.8), "label": "Semis (SMH)"},
        "ENERGY": {"pos": (1.8, -0.8), "label": "Energy (XLE)"},
        "HOME": {"pos": (-0.8, -0.4), "label": "Housing (XHB)"},
        "BANKS": {"pos": (1.5, -1.0), "label": "Banks (XLF)"},
        "VIX": {"pos": (0, 1.5), "label": "Vol (^VIX)"},
    }
    
    edges = [("US10Y", "QQQ"), ("US10Y", "GOLD"), ("US10Y", "HOME"), ("DXY", "GOLD"), ("DXY", "OIL"), ("HYG", "SPY"), ("HYG", "IWM"), ("HYG", "BANKS"), ("QQQ", "BTC"), ("QQQ", "SEMIS"), ("COPPER", "US10Y"), ("OIL", "ENERGY"), ("VIX", "SPY")]
    
    edge_x, edge_y = [], []
    for u, v in edges:
        x0, y0 = nodes[u]["pos"]; x1, y1 = nodes[v]["pos"]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        
    node_x, node_y, node_text, hover_text = [], [], [], []
    node_color, node_size = [], []
    
    for key, info in nodes.items():
        node_x.append(info["pos"][0]); node_y.append(info["pos"][1])
        d = data.get(key, {})
        valid = d.get("valid", False)
        val = get_val(data, key, timeframe)
        
        if not valid or not np.isfinite(val): col = "#6b7280"; fmt_val = "N/A"
        else:
            col = "#22c55e" if val > 0 else "#ef4444" if val < 0 else "#6b7280"
            fmt_val = f"{val:+.2f}%"
            if key == "US10Y": fmt_val = f"{val:+.1f} bps"
                
        node_color.append(col); node_size.append(45 if key in ["US10Y", "DXY", "HYG"] else 35)
        node_text.append(info["label"].split("(")[0].strip())
        hover_text.append(f"{info['label']}<br>Change: {fmt_val}<br>Source: {d.get('source', 'primary')}")
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#4b5563"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="bottom center", hovertext=hover_text, hoverinfo="text", marker=dict(size=node_size, color=node_color, line=dict(width=2, color="white")), textfont=dict(size=11, color="white")))
    fig.update_layout(showlegend=False, margin=dict(b=0, l=0, r=0, t=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.0, 2.0]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500)
    return fig

def plot_normalized_history(full_hist, assets, window_days=60):
    """
    Plots normalized performance of multiple assets starting at 0%
    """
    df_merged = pd.DataFrame()
    for asset in assets:
        if asset in full_hist:
            s = full_hist[asset]
            if len(s) > window_days:
                s = s.iloc[-window_days:]
            # Rebase to 0
            start_val = s.iloc[0]
            if start_val > 0:
                df_merged[asset] = ((s / start_val) - 1) * 100
                
    if df_merged.empty: return go.Figure()
    
    fig = px.line(df_merged, x=df_merged.index, y=df_merged.columns)
    fig.update_layout(
        title=f"Normalized Performance (Last {window_days} Days)",
        xaxis_title="", yaxis_title="% Change",
        legend_title="",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"), hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10), height=400
    )
    return fig

def plot_rolling_correlation(full_hist, asset1, asset2, window=20):
    """
    Plots rolling 20-day correlation between two assets.
    """
    if asset1 not in full_hist or asset2 not in full_hist: return go.Figure()
    
    s1 = full_hist[asset1]
    s2 = full_hist[asset2]
    
    # Align dates
    df = pd.DataFrame({asset1: s1, asset2: s2}).dropna()
    rolling_corr = df[asset1].rolling(window=window).corr(df[asset2]).dropna()
    
    # Slice last year max
    if len(rolling_corr) > 252:
        rolling_corr = rolling_corr.iloc[-252:]
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr.values, mode='lines', fill='tozeroy', name='Correlation', line=dict(color='#3b82f6')))
    
    fig.update_layout(
        title=f"{asset1} vs {asset2} ({window}D Rolling Correlation)",
        yaxis=dict(range=[-1.1, 1.1], title="Correlation"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        margin=dict(l=10, r=10, t=40, b=10), height=350,
        shapes=[dict(type="line", xref="paper", x0=0, x1=1, y0=0, y1=0, line=dict(color="white", width=1, dash="dot"))]
    )
    return fig

def _sankey_from_values(title: str, values: Dict[str, float], color_rgba: str) -> go.Figure:
    clean = {k: v for k, v in values.items() if np.isfinite(v)}
    if len(clean) < 6: return go.Figure()
    df = pd.DataFrame(list(clean.items()), columns=["id", "val"]).sort_values("val", ascending=False)
    winners = df.head(3); losers = df.tail(3)
    labels = list(losers["id"]) + list(winners["id"])
    sources, targets, link_values, colors = [], [], [], []
    for i in range(len(losers)):
        for j in range(len(winners)):
            sources.append(i); targets.append(len(losers) + j)
            link_values.append(float(abs(losers.iloc[i]["val"]) + abs(winners.iloc[j]["val"])))
            colors.append(color_rgba)
    node_colors = ["#ef4444"] * 3 + ["#22c55e"] * 3
    fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=node_colors), link=dict(source=sources, target=targets, value=link_values, color=colors))])
    fig.update_layout(title_text=title, font=dict(color="white"), paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_sankey_sectors(data, timeframe):
    sectors = {k: get_val(data, k, timeframe) for k in ["TECH", "SEMIS", "BANKS", "ENERGY", "HOME", "UTIL", "HEALTH", "MAT", "COMM"]}
    return _sankey_from_values(f"Sector Rotation ({'Daily' if timeframe == 'Tactical (Daily)' else 'WTD'})", sectors, "rgba(59, 130, 246, 0.2)")

def plot_sankey_assets(data, timeframe):
    assets = {k: get_val(data, k, timeframe) for k in ["SPY", "TLT", "DXY", "GOLD", "BTC", "OIL", "HYG"]}
    return _sankey_from_values(f"Asset Rotation ({'Daily' if timeframe == 'Tactical (Daily)' else 'WTD'})", assets, "rgba(168, 85, 247, 0.2)")

def plot_trend_momentum_quadrant(full_hist, category, timeframe):
    if category == "SECTORS": keys = ["TECH", "SEMIS", "BANKS", "ENERGY", "HOME", "UTIL", "STAPLES", "DISC", "IND", "HEALTH", "MAT", "COMM", "RE"]
    else: keys = ["SPY", "QQQ", "IWM", "GOLD", "BTC", "TLT", "DXY", "HYG", "OIL"]
    trend_win, mom_win = (20, 5) if timeframe == "Tactical (Daily)" else (100, 25)
    items = []
    for k in keys:
        s = full_hist.get(k)
        if s is None or len(s) < (trend_win + mom_win + 5): continue
        s = s.dropna()
        curr = float(s.iloc[-1])
        sma_long = float(s.rolling(window=trend_win).mean().iloc[-1])
        if sma_long == 0: continue
        trend = ((curr / sma_long) - 1.0) * 100.0
        prev_mom = float(s.iloc[-(mom_win + 1)])
        mom = ((curr / prev_mom) - 1.0) * 100.0 if prev_mom != 0 else 0.0
        c = "#22c55e" if trend > 0 and mom > 0 else "#3b82f6" if trend < 0 and mom > 0 else "#f59e0b" if trend > 0 and mom < 0 else "#ef4444"
        items.append({"Symbol": k, "Trend": trend, "Momentum": mom, "Color": c})
    df = pd.DataFrame(items)
    if df.empty: return go.Figure()
    fig = px.scatter(df, x="Trend", y="Momentum", text="Symbol", color="Color", color_discrete_map="identity")
    fig.update_traces(textposition="top center", marker=dict(size=14, line=dict(width=1, color="white")))
    fig.add_hline(y=0, line_dash="dot", line_color="#555"); fig.add_vline(x=0, line_dash="dot", line_color="#555")
    limit = max(df["Trend"].abs().max(), df["Momentum"].abs().max()) * 1.1
    fig.update_layout(xaxis=dict(range=[-limit, limit], showgrid=True, gridcolor="#333", title=f"Trend (vs SMA{trend_win})"), yaxis=dict(range=[-limit, limit], showgrid=True, gridcolor="#333", title=f"Momentum (ROC {mom_win})"), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), showlegend=False, height=450, margin=dict(l=20, r=20, t=20, b=20))
    return fig

def plot_correlation_heatmap(history_df, timeframe):
    if history_df is None or history_df.empty: return go.Figure()
    df_calc = history_df.resample("W-FRI").last() if timeframe != "Tactical (Daily)" else history_df.copy()
    corr = df_calc.pct_change().corr()
    subset = ["US10Y", "DXY", "VIX", "HYG", "SPY", "QQQ", "IWM", "BTC", "GOLD", "OIL"]
    cols = [c for c in subset if c in corr.columns]
    if len(cols) < 2: return go.Figure()
    fig = px.imshow(corr.loc[cols, cols], text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), height=400, margin=dict(l=10, r=10, t=10, b=10))
    return fig

# ----------------------------
# 9) MAIN APP
# ----------------------------
def main() -> None:
    st.title(APP_VERSION)
    with st.sidebar:
        st.subheader("Data Controls")
        strict_data = st.checkbox("Strict mode (no proxies)", value=False)
        st.divider()
        st.subheader("Notes")
        st.write("- Weekly view is WTD vs last completed Friday close.")
        st.write("- Trend/Mom quadrant is RRG-style proxy.")
        
    with st.spinner("Connecting to MacroNexus Core..."):
        market_data, history_df, full_hist, health = fetch_market_data()
    
    if health.errors.get("__download__"): st.error(f"Download Error: {health.errors['__download__']}")
    
    cols_h = st.columns(4)
    cols_h[0].markdown(f"<span class='badge-blue'>Fetched: {health.fetched_at_utc[11:19]}</span>", unsafe_allow_html=True)
    cols_h[1].markdown(f"<span class='badge-blue'>Valid: {len(health.valid_keys)}/{len(TICKERS)}</span>", unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def metric_tile(col, label, key):
        d = market_data.get(key, {})
        if not d.get("valid", False): 
            col.markdown(f"<div class='metric-card'><div>{label}</div><div>--</div></div>", unsafe_allow_html=True)
            return
        val, chg = d.get("price"), d.get("change")
        color = "#F43F5E" if (chg > 0 and key in ["US10Y", "DXY", "VIX"]) or (chg < 0 and key not in ["US10Y", "DXY", "VIX"]) else "#10B981"
        fmt_val = f"{val:.2f}%" if key == "US10Y" else f"{val:.2f}"
        fmt_chg = f"{chg:+.1f} bps" if key == "US10Y" else f"{chg:+.2f}%"
        col.markdown(f"<div class='metric-card' style='border-left: 3px solid {color};'><div class='metric-label'>{label}</div><div class='metric-value'>{fmt_val}<span class='metric-delta' style='color:{color}'>{fmt_chg}</span></div></div>", unsafe_allow_html=True)

    metric_tile(c1, "Credit", "HYG"); metric_tile(c2, "VIX", "VIX"); metric_tile(c3, "10Y Yield", "US10Y")
    metric_tile(c4, "Dollar", "DXY"); metric_tile(c5, "Oil", "OIL"); metric_tile(c6, "Bitcoin", "BTC")

    with st.container():
        st.markdown('<div class="control-container">', unsafe_allow_html=True)
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
        with col_ctrl2: timeframe = st.selectbox("Analytic View", TIMEFRAMES, label_visibility="collapsed")
        regime, confidence, reasons = determine_regime(market_data, timeframe, strict_data)
        with col_ctrl1: 
            override = st.checkbox("Manual Override", value=False)
            active_regime = st.selectbox("Force Regime", list(STRATEGIES.keys()), label_visibility="collapsed") if override else regime
        with col_ctrl3:
            rc = "#EF4444" if active_regime == "DATA ERROR" else ("#F59E0B" if confidence == "LOW" else "#10B981")
            st.markdown(f"<div style='text-align:right'><span style='color:{rc};font-weight:bold'>{active_regime}</span></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if regime == "DATA ERROR" and not override: st.error("CRITICAL DATA FAILURE")
    elif confidence == "LOW" and not override: st.warning("LOW CONFIDENCE: Proxies used.")

    tab_mission, tab_trends, tab_workflow, tab_pulse, tab_macro, tab_playbook = st.tabs(
        ["🚀 MISSION CONTROL", "📉 MACRO TRENDS", "📋 WORKFLOW", "📊 MARKET PULSE", "🕸️ MACRO MACHINE", "📖 STRATEGY PLAYBOOK"]
    )

    with tab_mission:
        with st.expander("🎛️ SPX Income Reactor Telemetry", expanded=True):
            tc1, tc2, tc3, tc4 = st.columns(4)
            asset_mode = tc1.radio("Asset Class", ["INDEX", "STOCKS"], horizontal=True)
            iv_rank = tc2.slider("IV Rank", 0, 100, 45)
            skew_rank = tc3.slider("Skew", 0, 100, 50)
            adx_val = tc4.slider("ADX", 0, 60, 20)
        
        # Reactor Gauge Visuals
        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=iv_rank, title={'text': "IV Rank"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#3b82f6"}})).update_layout(height=150, margin=dict(l=20,r=20,t=30,b=20)), use_container_width=True)
        g2.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=skew_rank, title={'text': "Skew"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#a855f7"}})).update_layout(height=150, margin=dict(l=20,r=20,t=30,b=20)), use_container_width=True)
        g3.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=adx_val, title={'text': "Trend Strength"}, gauge={'axis': {'range': [0, 60]}, 'bar': {'color': "#f59e0b"}})).update_layout(height=150, margin=dict(l=20,r=20,t=30,b=20)), use_container_width=True)

        st.divider()
        strat_data = STRATEGIES.get(active_regime, STRATEGIES["NEUTRAL"])
        # Logic simplified for display
        output = strat_data["index"] if asset_mode == "INDEX" else strat_data["stock"]
        
        col_L, col_R = st.columns([1, 2])
        with col_L: st.markdown(f"<div class='strat-card'><div class='strat-title' style='color:{rc}'>{active_regime}</div><div class='strat-data'>{strat_data['desc']}</div></div>", unsafe_allow_html=True)
        with col_R: st.markdown(f"<div class='strat-card' style='border-color:{rc}'><div class='strat-title' style='color:{rc}'>{output['strat']}</div><div class='strat-data'>{output['setup']}</div><div class='strat-subtitle'>{output['notes']}</div></div>", unsafe_allow_html=True)

    with tab_trends:
        st.subheader("📉 Macro Trends & Relative Performance")
        c_trend1, c_trend2 = st.columns(2)
        with c_trend1:
            st.plotly_chart(plot_normalized_history(full_hist, ["SPY", "TLT", "GLD", "BTC-USD", "DX-Y.NYB"], 90), use_container_width=True)
        with c_trend2:
            st.plotly_chart(plot_rolling_correlation(full_hist, "SPY", "TLT", 20), use_container_width=True)
            st.plotly_chart(plot_rolling_correlation(full_hist, "SPY", "DX-Y.NYB", 20), use_container_width=True)

    with tab_workflow:
        w1, w2, w3, w4, w5 = st.columns(5)
        with w1: st.markdown("<div class='metric-card' style='height:200px; color:#F87171'><b>1. VETO</b><br><small>Check HYG & VIX</small></div>", unsafe_allow_html=True)
        with w2: st.markdown("<div class='metric-card' style='height:200px; color:#FBBF24'><b>2. REGIME</b><br><small>Identify Tailwind</small></div>", unsafe_allow_html=True)
        with w3: st.markdown("<div class='metric-card' style='height:200px; color:#60A5FA'><b>3. SECTOR</b><br><small>Check Flows</small></div>", unsafe_allow_html=True)
        with w4: st.markdown("<div class='metric-card' style='height:200px; color:#A78BFA'><b>4. TACTICS</b><br><small>Reactor Inputs</small></div>", unsafe_allow_html=True)
        with w5: st.markdown("<div class='metric-card' style='height:200px; color:#34D399'><b>5. EXECUTE</b><br><small>3:00 PM Check</small></div>", unsafe_allow_html=True)

    with tab_pulse:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_sankey_sectors(market_data, timeframe), use_container_width=True)
        with c2: st.plotly_chart(plot_sankey_assets(market_data, timeframe), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3: st.plotly_chart(plot_correlation_heatmap(history_df, timeframe), use_container_width=True)
        with c4: st.plotly_chart(plot_trend_momentum_quadrant(full_hist, "SECTORS", timeframe), use_container_width=True)

    with tab_macro:
        c1, c2 = st.columns([3, 1])
        with c1: st.plotly_chart(plot_nexus_graph_dots(market_data, timeframe), use_container_width=True)
        with c2: st.markdown("<div class='context-box'><b>The Plumbing</b><br>Network graph of macro linkages.<br>Green = Rising, Red = Falling.</div>", unsafe_allow_html=True)

    with tab_playbook:
        st.write("Detailed strategies here...") # Kept brief for length, can expand

if __name__ == "__main__":
    main()
