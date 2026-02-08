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

APP_VERSION = "MacroNexus Pro (Hardened Build)"

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

# NOTE:
# - DXY and VIX are sometimes unreliable on Yahoo.
# - We allow proxies but DO NOT treat them as equivalent:
# proxies will downgrade confidence (or trigger DATA ERROR in strict mode).
PROXIES: Dict[str, str] = {
"DXY": "UUP", # USD Bull ETF (proxy for DXY)
"VIX": "VIXY", # VIX futures ETN/ETF proxy (not VIX spot)
"RUT": "IWM", # Russell proxy (reasonable)
}

CRITICAL_KEYS = ["HYG", "VIX", "US10Y", "DXY"]

TIMEFRAMES = ["Tactical (Daily)", "Structural (Weekly, WTD)"] # WTD vs last completed Friday close


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
# 3) LOGGING (OPTIONAL)
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
proxy_used: Dict[str, str] # key -> proxy symbol used
errors: Dict[str, str] # key -> error string
note: Optional[str] = None


# ----------------------------
# 5) DATA ENGINE (ROBUST BATCH + PROXIES)
# ----------------------------

def _safe_is_multiindex_columns(df: pd.DataFrame) -> bool:
return isinstance(df.columns, pd.MultiIndex)


def _extract_close_series(df_batch: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
"""
Robustly extract a Close series for a symbol from yfinance download output.
Handles:
- MultiIndex columns (common for multi-ticker)
- SingleIndex columns (possible in some failure modes)
Returns a Series indexed by datetime, or None.
"""
if df_batch is None or df_batch.empty:
return None

try:
if _safe_is_multiindex_columns(df_batch):
top = df_batch.columns.get_level_values(0)
if symbol not in set(top):
return None

# Prefer 'Close' if present, otherwise try 'Adj Close'
cols_lvl1 = set(df_batch[symbol].columns)
if "Close" in cols_lvl1:
s = df_batch[symbol]["Close"].dropna()
elif "Adj Close" in cols_lvl1:
s = df_batch[symbol]["Adj Close"].dropna()
else:
return None
return s if isinstance(s, pd.Series) else None

# SingleIndex columns: assume it is a single ticker output.
# We can only support this if it contains a Close column.
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
"""
Structural (Weekly, WTD): compare current close vs LAST COMPLETED Friday close.

- If today is mid-week, we compare today close vs prior Friday close.
- If today is Friday and we have Friday close, we compare Friday close vs previous Friday close.
"""
curr = float(series.iloc[-1])

# Need at least ~10 points for good weekly resampling fallback
if len(series) < 10:
prev = float(series.iloc[-2]) if len(series) >= 2 else curr
return curr, prev

weekly = series.resample("W-FRI").last().dropna()
if len(weekly) >= 2:
# weekly.iloc[-1] may be this week's partial (labeled Fri) unless today is actual Fri close
# The last completed Friday close is the second-to-last bucket in WTD terms.
last_date = series.index[-1]
is_friday_close = (getattr(last_date, "weekday", lambda: -1)() == 4) and (weekly.index[-1].date() == last_date.date())

if is_friday_close:
# Friday close: compare vs previous Friday close
prev_friday = float(weekly.iloc[-2])
return curr, prev_friday

# Mid-week: compare vs last completed Friday close
prev_friday = float(weekly.iloc[-2])
return curr, prev_friday

# Fallback: approximate 5 trading days
prev_approx = float(series.iloc[-6]) if len(series) >= 6 else float(series.iloc[-2])
return curr, prev_approx


@st.cache_data(ttl=300)
def fetch_market_data() -> Tuple[Dict[str, dict], pd.DataFrame, Dict[str, pd.Series], DataHealth]:
"""
Downloads primary AND proxy symbols in one batch.
Returns:
data: dict keyed by macro key with {price, change, change_w, valid, symbol_used, source}
history_df: aligned close series for correlation
full_hist: series dict for RRG/Trend/Momentum computations
health: DataHealth
"""
fetched_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# Download primaries + proxies in one call
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
health = DataHealth(
fetched_at_utc=fetched_at,
valid_keys=[],
invalid_keys=list(TICKERS.keys()),
proxy_used={},
errors={"__download__": str(e)},
note="API exception during download",
)
return {}, pd.DataFrame(), {}, health

if df_batch is None or df_batch.empty:
health = DataHealth(
fetched_at_utc=fetched_at,
valid_keys=[],
invalid_keys=list(TICKERS.keys()),
proxy_used={},
errors={"__download__": "No data returned from Yahoo"},
note="Empty download",
)
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

# Proxy fallback
if (s is None or len(s) < 50) and key in PROXIES:
proxy_symbol = PROXIES[key]
s_proxy = _extract_close_series(df_batch, proxy_symbol)
if s_proxy is not None and len(s_proxy) >= 50:
s = s_proxy
used_symbol = proxy_symbol
source = "proxy"
proxy_used[key] = proxy_symbol

if s is None or len(s) < 2:
data[key] = {
"valid": False,
"symbol_primary": primary_symbol,
"symbol_used": used_symbol,
"source": source,
}
errors[key] = "Missing or insufficient data"
continue

# Ensure DatetimeIndex and sorted
try:
s = s.sort_index()
if not isinstance(s.index, pd.DatetimeIndex):
data[key] = {
"valid": False,
"symbol_primary": primary_symbol,
"symbol_used": used_symbol,
"source": source,
}
errors[key] = "Non-datetime index"
continue
except Exception as e:
data[key] = {
"valid": False,
"symbol_primary": primary_symbol,
"symbol_used": used_symbol,
"source": source,
}
errors[key] = f"Series normalization error: {e}"
continue

# Store for downstream
history_data[key] = s
full_hist[key] = s

curr = float(s.iloc[-1])
prev = float(s.iloc[-2])

# Weekly (WTD vs last completed Fri)
curr_w, prev_w = _weekly_wtd_reference(s)

# Metrics
if key == "US10Y":
# Yahoo TNX is ~10x yield
display_price = curr / 10.0
chg_d = (curr - prev) * 10.0 # bps
chg_w = (curr_w - prev_w) * 10.0 # bps
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
"source": source, # primary | proxy
}

valid_keys = [k for k, v in data.items() if v.get("valid")]
invalid_keys = [k for k, v in data.items() if not v.get("valid")]

health = DataHealth(
fetched_at_utc=fetched_at,
valid_keys=valid_keys,
invalid_keys=invalid_keys,
proxy_used=proxy_used,
errors=errors,
note="Weekly view is WTD vs last completed Friday close",
)

history_df = pd.DataFrame(history_data) # aligned by index union
return data, history_df, full_hist, health


# ----------------------------
# 6) STRATEGY DATABASE
# ----------------------------

STRATEGIES = {
"GOLDILOCKS": {
"desc": "Low Vol + Steady Trend. Market climbing wall of worry.",
"risk": "1.5%",
"bias": "Long",
"index": {
"strat": "Directional Diagonal",
"dte": "Front 17 / Back 31",
"setup": "Buy Back ITM (70D) / Sell Front OTM (30D)",
"notes": "Stock replacement. Trend (Delta) + Decay (Theta). Upside is uncapped.",
},
"stock": {
"strat": "Call Debit Spreads",
"dte": "45-60 DTE",
"setup": "Buy 60D / Sell 30D (Spread)",
"notes": "Focus on Relative Strength Leaders (Tech, Semis). Use pullbacks to EMA21.",
},
"longs": "TECH, SEMIS, DISC",
"shorts": "VIX, TLT",
},
"LIQUIDITY": {
"desc": "High Liquidity / Dollar Weakness. Drift Up environment.",
"risk": "1.0%",
"bias": "Aggressive Long",
"index": {
"strat": "Flyagonal (Drift)",
"dte": "Entry 7-10 DTE",
"setup": "Upside: Call BWB (Long +10 / Short 2x +50 / Long +60). Downside: Put Diagonal.",
"notes": "Captures the drift. Upside tent funds the downside floor. Target 4% Flash Win.",
},
"stock": {
"strat": "Risk Reversals",
"dte": "60 DTE",
"setup": "Sell OTM Put / Buy OTM Call",
"notes": "Funding long delta with short volatility. Best for High Beta (Crypto proxies).",
},
"longs": "BTC, SEMIS, QQQ",
"shorts": "DXY, CASH",
},
"REFLATION": {
"desc": "Inflation / Rates Rising. Real Assets outperform Tech.",
"risk": "1.0%",
"bias": "Cyclical Long",
"index": {
"strat": "Call Spread (Cyclicals)",
"dte": "45 DTE",
"setup": "Buy 60D / Sell 30D",
"notes": "Focus on Russell 2000 (IWM). Avoid long duration Tech (QQQ) as rates rise.",
},
"stock": {
"strat": "Cash Secured Puts",
"dte": "30-45 DTE",
"setup": "Sell 30D Puts on Energy/Banks",
"notes": "Energy (XLE) and Banks (XLF) benefit from rising yields. Sell premium to acquire.",
},
"longs": "ENERGY, BANKS, IND",
"shorts": "TLT, TECH",
},
"NEUTRAL": {
"desc": "Chop / Range Bound. No clear direction.",
"risk": "Income Size",
"bias": "Neutral/Theta",
"index": {
"strat": "TimeEdge (SPX) / TimeZone (RUT)",
"dte": "Entry 15 / Exit 7",
"setup": "Put Calendar Spread (ATM) or Double Calendar",
"notes": "Pure Theta play. Sell 15 DTE / Buy 22+ DTE. Requires VIX < 20.",
},
"stock": {
"strat": "Iron Condor",
"dte": "30-45 DTE",
"setup": "Sell 20D Call / Sell 20D Put (Wings 10 wide)",
"notes": "Delta neutral income. Best on low beta stocks (KO, PEP) during chop.",
},
"longs": "INCOME, CASH",
"shorts": "MOMENTUM",
},
"RISK OFF": {
"desc": "High Volatility / Credit Stress. Preservation mode.",
"risk": "0.5%",
"bias": "Short/Hedge",
"index": {
"strat": "A14 Put BWB",
"dte": "Entry 14 / Exit 7",
"setup": "Long ATM / Short 2x -40 / (Skip) / Long -60",
"notes": "Crash Catcher. Zero upside risk. Profit tent expands into the crash. Enter Friday AM.",
},
"stock": {
"strat": "Put Debit Spreads",
"dte": "60 DTE",
"setup": "Buy 40D / Sell 15D",
"notes": "Directional downside. Selling the 15D put reduces cost and offsets IV crush.",
},
"longs": "VIX, DXY",
"shorts": "SPY, IWM, HYG",
},
"DATA ERROR": {
"desc": "CRITICAL DATA FEED FAILURE",
"risk": "0.0%",
"bias": "Flat",
"index": {"strat": "STAND ASIDE", "dte": "--", "setup": "--", "notes": "Do not trade. Data integrity compromised."},
"stock": {"strat": "STAND ASIDE", "dte": "--", "setup": "--", "notes": "Do not trade. Data integrity compromised."},
"longs": "--",
"shorts": "--",
},
}


# ----------------------------
# 7) HELPERS (TIMEFRAME / REGIME)
# ----------------------------

def get_val(data: Dict[str, dict], key: str, timeframe: str) -> float:
d = data.get(key, {})
if not d.get("valid", False):
return float("nan")

if timeframe == "Tactical (Daily)":
return float(d.get("change", 0.0))
return float(d.get("change_w", 0.0))


def determine_regime(
data: Dict[str, dict],
timeframe: str,
strict_data: bool,
) -> Tuple[str, str, List[str]]:
"""
Returns:
regime: one of STRATEGIES keys
confidence: HIGH | LOW | NONE
reasons: list of strings for warnings
"""
reasons: List[str] = []

# 1) Critical validity check
missing_crit = [k for k in CRITICAL_KEYS if not data.get(k, {}).get("valid", False)]
if missing_crit:
reasons.append(f"Missing critical keys: {', '.join(missing_crit)}")
return "DATA ERROR", "NONE", reasons

# 2) Proxy usage check
proxy_crit = [k for k in CRITICAL_KEYS if data.get(k, {}).get("source") == "proxy"]
if proxy_crit:
reasons.append(f"Proxy used for critical keys: {', '.join(proxy_crit)}")
if strict_data:
reasons.append("Strict mode enabled: proxies not allowed for critical keys.")
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

# Regime priority: Risk Off first (veto)
if hyg < -0.5 or vix > 5.0:
return "RISK OFF", confidence, reasons

if (oil > 1.5 or cop > 1.5) and us10y > 3.0 and banks > 0:
return "REFLATION", confidence, reasons

if dxy < -0.3 and btc > 1.5:
return "LIQUIDITY", confidence, reasons

if vix < 0 and abs(us10y) < 5.0 and hyg > -0.1:
return "GOLDILOCKS", confidence, reasons

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

edges = [
("US10Y", "QQQ"),
("US10Y", "GOLD"),
("US10Y", "HOME"),
("DXY", "GOLD"),
("DXY", "OIL"),
("HYG", "SPY"),
("HYG", "IWM"),
("HYG", "BANKS"),
("QQQ", "BTC"),
("QQQ", "SEMIS"),
("COPPER", "US10Y"),
("OIL", "ENERGY"),
("VIX", "SPY"),
]

edge_x, edge_y = [], []
for u, v in edges:
x0, y0 = nodes[u]["pos"]
x1, y1 = nodes[v]["pos"]
edge_x.extend([x0, x1, None])
edge_y.extend([y0, y1, None])

node_x, node_y, node_text, hover_text = [], [], [], []
node_color, node_size, node_symbol = [], [], []

for key, info in nodes.items():
node_x.append(info["pos"][0])
node_y.append(info["pos"][1])

d = data.get(key, {})
valid = d.get("valid", False)
val = get_val(data, key, timeframe)

if not valid or not np.isfinite(val):
col = "#6b7280"
sym = "circle-open"
fmt_val = "N/A"
else:
col = "#22c55e" if val > 0 else "#ef4444" if val < 0 else "#6b7280"
sym = "circle"
fmt_val = f"{val:+.2f}%"
if key == "US10Y":
fmt_val = f"{val:+.1f} bps"

node_color.append(col)
node_symbol.append(sym)
node_size.append(45 if key in ["US10Y", "DXY", "HYG"] else 35)

label_clean = info["label"].split("(")[0].strip()
node_text.append(label_clean)

source = d.get("source", "primary")
used_sym = d.get("symbol_used", d.get("symbol_primary", ""))
hover_text.append(f"{info['label']}<br>Change: {fmt_val}<br>Source: {source} ({used_sym})")

fig = go.Figure()

fig.add_trace(
go.Scatter(
x=edge_x,
y=edge_y,
mode="lines",
line=dict(width=1, color="#4b5563"),
hoverinfo="none",
)
)

fig.add_trace(
go.Scatter(
x=node_x,
y=node_y,
mode="markers+text",
text=node_text,
textposition="bottom center",
hovertext=hover_text,
hoverinfo="text",
marker=dict(size=node_size, color=node_color, symbol=node_symbol, line=dict(width=2, color="white")),
textfont=dict(size=11, color="white"),
)
)

fig.update_layout(
showlegend=False,
margin=dict(b=0, l=0, r=0, t=0),
xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.0, 2.0]),
paper_bgcolor="rgba(0,0,0,0)",
plot_bgcolor="rgba(0,0,0,0)",
height=500,
)
return fig


def _sankey_from_values(title: str, values: Dict[str, float], color_rgba: str) -> go.Figure:
"""
Build a simple losers -> winners Sankey using only finite values.
If insufficient data, returns an annotated empty figure.
"""
clean = {k: v for k, v in values.items() if np.isfinite(v)}
if len(clean) < 6:
fig = go.Figure()
fig.add_annotation(
text="Insufficient valid data for Sankey",
showarrow=False,
x=0.5,
y=0.5,
xref="paper",
yref="paper",
font=dict(color="#ccc", size=14),
)
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300)
return fig

df = pd.DataFrame(list(clean.items()), columns=["id", "val"]).sort_values("val", ascending=False)

winners = df.head(3)
losers = df.tail(3)

labels = list(losers["id"]) + list(winners["id"])
sources, targets, link_values, colors = [], [], [], []

for i in range(len(losers)):
for j in range(len(winners)):
sources.append(i)
targets.append(len(losers) + j)
link_values.append(float(abs(losers.iloc[i]["val"]) + abs(winners.iloc[j]["val"])))
colors.append(color_rgba)

node_colors = ["#ef4444"] * 3 + ["#22c55e"] * 3

fig = go.Figure(
data=[
go.Sankey(
node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=node_colors),
link=dict(source=sources, target=targets, value=link_values, color=colors),
)
]
)
fig.update_layout(
title_text=title,
font=dict(color="white"),
paper_bgcolor="rgba(0,0,0,0)",
height=350,
margin=dict(l=10, r=10, t=40, b=10),
)
return fig


def plot_sankey_sectors(data: Dict[str, dict], timeframe: str) -> go.Figure:
sector_keys = ["TECH", "SEMIS", "BANKS", "ENERGY", "HOME", "UTIL", "HEALTH", "MAT", "COMM"]
sectors = {k: get_val(data, k, timeframe) for k in sector_keys}
title = f"Sector Rotation ({'Daily' if timeframe == 'Tactical (Daily)' else 'Weekly WTD vs last Fri'})"
return _sankey_from_values(title=title, values=sectors, color_rgba="rgba(59, 130, 246, 0.2)")


def plot_sankey_assets(data: Dict[str, dict], timeframe: str) -> go.Figure:
asset_keys = ["SPY", "TLT", "DXY", "GOLD", "BTC", "OIL", "HYG"]
assets = {k: get_val(data, k, timeframe) for k in asset_keys}
title = f"Asset Rotation ({'Daily' if timeframe == 'Tactical (Daily)' else 'Weekly WTD vs last Fri'})"
return _sankey_from_values(title=title, values=assets, color_rgba="rgba(168, 85, 247, 0.2)")


def plot_trend_momentum_quadrant(full_hist: Dict[str, pd.Series], category: str, timeframe: str) -> go.Figure:
"""
Trend/Momentum Quadrant (RRG-proxy):
- Trend: price vs SMA(trend_win)
- Momentum: ROC(mom_win)
"""
if category == "SECTORS":
keys = ["TECH", "SEMIS", "BANKS", "ENERGY", "HOME", "UTIL", "STAPLES", "DISC", "IND", "HEALTH", "MAT", "COMM", "RE"]
else:
keys = ["SPY", "QQQ", "IWM", "GOLD", "BTC", "TLT", "DXY", "HYG", "OIL"]

if timeframe == "Tactical (Daily)":
trend_win = 20
mom_win = 5
else:
trend_win = 100
mom_win = 25

items = []
for k in keys:
s = full_hist.get(k)
if s is None or len(s) < (trend_win + mom_win + 5):
continue

s = s.dropna()
if len(s) < (trend_win + mom_win + 5):
continue

curr = float(s.iloc[-1])
sma_long = float(s.rolling(window=trend_win).mean().iloc[-1])
if sma_long == 0 or not np.isfinite(sma_long):
continue

trend_score = ((curr / sma_long) - 1.0) * 100.0

# Correct N-day ROC uses -(mom_win + 1)
prev_mom = float(s.iloc[-(mom_win + 1)])
mom_score = ((curr / prev_mom) - 1.0) * 100.0 if prev_mom != 0 else 0.0

if trend_score > 0 and mom_score > 0:
c = "#22c55e" # Leading
elif trend_score < 0 and mom_score > 0:
c = "#3b82f6" # Improving
elif trend_score > 0 and mom_score < 0:
c = "#f59e0b" # Weakening
else:
c = "#ef4444" # Lagging

items.append({"Symbol": k, "Trend": trend_score, "Momentum": mom_score, "Color": c})

df = pd.DataFrame(items)
if df.empty:
fig = go.Figure()
fig.add_annotation(
text="Insufficient valid history for quadrant plot",
showarrow=False,
x=0.5,
y=0.5,
xref="paper",
yref="paper",
font=dict(color="#ccc", size=14),
)
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=400)
return fig

fig = px.scatter(df, x="Trend", y="Momentum", text="Symbol", color="Color", color_discrete_map="identity")
fig.update_traces(textposition="top center", marker=dict(size=14, line=dict(width=1, color="white")))
fig.add_hline(y=0, line_dash="dot", line_color="#555")
fig.add_vline(x=0, line_dash="dot", line_color="#555")

limit = float(max(df["Trend"].abs().max(), df["Momentum"].abs().max()) * 1.1)
limit = max(limit, 1.0)

fig.update_layout(
xaxis=dict(range=[-limit, limit], zeroline=False, showgrid=True, gridcolor="#333", title=f"Trend (vs SMA{trend_win})"),
yaxis=dict(range=[-limit, limit], zeroline=False, showgrid=True, gridcolor="#333", title=f"Momentum (ROC {mom_win})"),
plot_bgcolor="rgba(0,0,0,0)",
paper_bgcolor="rgba(0,0,0,0)",
font=dict(color="#ccc"),
showlegend=False,
height=450,
margin=dict(l=20, r=20, t=20, b=20),
)
return fig


def plot_correlation_heatmap(history_df: pd.DataFrame, timeframe: str) -> go.Figure:
if history_df is None or history_df.empty:
return go.Figure()

df_calc = history_df.copy()

if timeframe != "Tactical (Daily)":
df_calc = df_calc.resample("W-FRI").last()

corr = df_calc.pct_change().corr()

subset = ["US10Y", "DXY", "VIX", "HYG", "SPY", "QQQ", "IWM", "BTC", "GOLD", "OIL"]
cols = [c for c in subset if c in corr.columns]
if len(cols) < 2:
fig = go.Figure()
fig.add_annotation(
text="Insufficient data for correlation heatmap",
showarrow=False,
x=0.5,
y=0.5,
xref="paper",
yref="paper",
font=dict(color="#ccc", size=14),
)
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350)
return fig

corr_subset = corr.loc[cols, cols]
fig = px.imshow(corr_subset, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
fig.update_layout(
paper_bgcolor="rgba(0,0,0,0)",
plot_bgcolor="rgba(0,0,0,0)",
font=dict(color="#ccc"),
height=400,
margin=dict(l=10, r=10, t=10, b=10),
)
return fig


# ----------------------------
# 9) MAIN APP
# ----------------------------

def main() -> None:
st.title(APP_VERSION)

# Sidebar controls (data integrity controls belong here)
with st.sidebar:
st.subheader("Data Controls")
strict_data = st.checkbox("Strict mode (no proxies for critical keys)", value=False)
st.caption("If strict mode is enabled, DXY/VIX proxies trigger DATA ERROR.")
st.divider()
st.subheader("Notes")
st.write("- Weekly view is WTD vs last completed Friday close.")
st.write("- Trend/Momentum quadrant is an RRG-style proxy, not benchmark-relative RRG.")

# Fetch data
with st.spinner("Connecting to MacroNexus Core (batched yfinance)..."):
market_data, history_df, full_hist, health = fetch_market_data()

# Data health banner
valid_count = len(health.valid_keys)
total_count = len(TICKERS)
proxy_count = len(health.proxy_used)

if health.errors.get("__download__"):
st.error(f"Data download error: {health.errors['__download__']}")

cols_health = st.columns([2, 2, 2, 3])
cols_health[0].markdown(f"<span class='badge-blue'>Fetched: {health.fetched_at_utc}</span>", unsafe_allow_html=True)
cols_health[1].markdown(f"<span class='badge-blue'>Valid: {valid_count}/{total_count}</span>", unsafe_allow_html=True)
cols_health[2].markdown(f"<span class='badge-blue'>Proxies used: {proxy_count}</span>", unsafe_allow_html=True)
if proxy_count:
proxy_str = ", ".join([f"{k}->{v}" for k, v in sorted(health.proxy_used.items())])
cols_health[3].markdown(f"<span class='badge-red'>Proxy map: {proxy_str}</span>", unsafe_allow_html=True)
else:
cols_health[3].markdown("<span class='badge-green'>All primary feeds</span>", unsafe_allow_html=True)

# Top metrics bar (always daily "live tape")
c1, c2, c3, c4, c5, c6 = st.columns(6)

def metric_tile(col, label, key):
d = market_data.get(key, {})
valid = d.get("valid", False)
if not valid:
col.markdown(
f"""<div class="metric-card" style="border-left: 3px solid #374151;">
<div class="metric-label">{label}</div>
<div class="metric-value">--</div>
</div>""",
unsafe_allow_html=True,
)
return

val = float(d.get("price", 0.0))
chg = float(d.get("change", 0.0))
is_up = chg > 0

# Drivers inverted for intuitive risk coloring in the tile
if key in ["US10Y", "DXY", "VIX"]:
color = "#F43F5E" if is_up else "#10B981"
else:
color = "#10B981" if is_up else "#F43F5E"

if key == "US10Y":
fmt_chg = f"{chg:+.1f} bps"
val_str = f"{val:.2f}%"
else:
fmt_chg = f"{chg:+.2f}%"
val_str = f"{val:.2f}"

src = d.get("source", "primary")
if src == "proxy":
label = f"{label} (proxy)"

col.markdown(
f"""<div class="metric-card" style="border-left: 3px solid {color};">
<div class="metric-label">{label}</div>
<div class="metric-value">{val_str}
<span class="metric-delta" style="color: {color};">{fmt_chg}</span>
</div>
</div>""",
unsafe_allow_html=True,
)

metric_tile(c1, "Credit (HYG)", "HYG")
metric_tile(c2, "Volatility (VIX)", "VIX")
metric_tile(c3, "10Y Yield", "US10Y")
metric_tile(c4, "Dollar (DXY)", "DXY")
metric_tile(c5, "Oil", "OIL")
metric_tile(c6, "Bitcoin", "BTC")

# Controls row
ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])

with ctrl2:
timeframe = st.selectbox("Analytic View", TIMEFRAMES, index=0, label_visibility="collapsed")

regime, confidence, regime_reasons = determine_regime(market_data, timeframe, strict_data=strict_data)

with ctrl1:
override = st.checkbox("Manual Override", value=False)
active_regime = st.selectbox("Force Regime", list(STRATEGIES.keys()), label_visibility="collapsed") if override else regime

with ctrl3:
r_colors = {
"GOLDILOCKS": "#10B981",
"LIQUIDITY": "#A855F7",
"REFLATION": "#F59E0B",
"NEUTRAL": "#6B7280",
"RISK OFF": "#EF4444",
"DATA ERROR": "#EF4444",
}
rc = r_colors.get(active_regime, "#6B7280")
status_text = "ERROR" if active_regime == "DATA ERROR" else ("LOW CONF" if confidence == "LOW" and not override else "ACTIVE")
st.markdown(
f"""<div style="text-align: right; display: flex; align-items: center; justify-content: flex-end; gap: 15px;">
<div style="text-align: right;">
<div style="font-size: 10px; color: #8B9BB4; letter-spacing: 1px;">SYSTEM STATUS</div>
<div style="font-size: 14px; font-weight: bold; color: {rc};">{status_text}</div>
</div>
<div class="regime-badge" style="background: {rc}22; color: {rc}; border: 1px solid {rc};">{active_regime}</div>
</div>""",
unsafe_allow_html=True,
)

if regime == "DATA ERROR" and not override:
st.error("CRITICAL DATA FAILURE: cannot determine a safe regime. Trading is NOT recommended.")
if regime_reasons:
st.caption(" | ".join(regime_reasons))
elif confidence == "LOW" and not override:
st.warning("LOW CONFIDENCE: one or more critical keys are using proxies. Confirm with a secondary data source.")
if regime_reasons:
st.caption(" | ".join(regime_reasons))

# Tabs
tab_mission, tab_workflow, tab_pulse, tab_macro, tab_playbook = st.tabs(
["🚀 MISSION CONTROL", "📋 WORKFLOW", "📊 MARKET PULSE", "🕸️ MACRO MACHINE", "📖 STRATEGY PLAYBOOK"]
)

# ---- TAB 1: Mission Control ----
with tab_mission:
with st.expander("🎛️ SPX Income Reactor Telemetry (Manual Input)", expanded=True):
tc1, tc2, tc3, tc4 = st.columns(4)
asset_mode = tc1.radio("Asset Class", ["INDEX (SPX/RUT)", "STOCKS"], horizontal=True)
iv_rank = tc2.slider("IV Rank (Percentile)", 0, 100, 45)
skew_rank = tc3.slider("Skew Rank", 0, 100, 50)
adx_val = tc4.slider("Trend ADX", 0, 60, 20)

st.divider()

strat_data = STRATEGIES.get(active_regime, STRATEGIES["NEUTRAL"])

# Reactor logic
if active_regime == "DATA ERROR":
reactor_output = strat_data["index"]
elif asset_mode == "STOCKS":
reactor_output = strat_data["stock"]
else:
if active_regime == "RISK OFF":
reactor_output = STRATEGIES["RISK OFF"]["index"]
elif iv_rank > 50:
if skew_rank > 80:
reactor_output = {
"strat": "Put BWB (High Skew)",
"dte": "21-30 DTE",
"setup": "Long ATM / Short -40 / Skip / Long -60",
"notes": "High skew detected (>80). Crash risk elevated. Use BWB to eliminate upside risk and profit from crash.",
}
else:
reactor_output = {
"strat": "Iron Condor",
"dte": "30-45 DTE",
"setup": "Delta 15 Wings",
"notes": "Volatility is high but skew is normal. Mean reversion income structure.",
}
else:
if adx_val > 25:
if active_regime == "LIQUIDITY":
reactor_output = STRATEGIES["LIQUIDITY"]["index"]
else:
reactor_output = {
"strat": "Directional Diagonal",
"dte": "Front 17 / Back 31",
"setup": "Buy Back ITM / Sell Front OTM",
"notes": "Trend detected (ADX > 25). Avoid capping upside.",
}
else:
reactor_output = STRATEGIES["NEUTRAL"]["index"]

col_L, col_R = st.columns([1, 2])

with col_L:
st.markdown(
f"""
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
""",
unsafe_allow_html=True,
)

with col_R:
st.markdown(
f"""
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
<span class="badge-blue">View: {timeframe}</span>
</div>
</div>
""",
unsafe_allow_html=True,
)

# ---- TAB 2: Workflow ----
with tab_workflow:
st.subheader("📋 The 5-Phase Mission Control Workflow")
w1, w2, w3, w4, w5 = st.columns(5)

with w1:
st.markdown(
"""
<div class="metric-card" style="height: 250px;">
<div style="color: #F87171; font-weight: bold; margin-bottom: 10px;">PHASE 1: VETO</div>
<div style="font-size: 12px; color: #AAA;">
1. Check <b>HYG</b> (Credit). Is it crashing (&lt; -0.5%)?
<br>2. Check <b>VIX</b>. Is it spiking (&gt; 5%)?
<br><br>
<span style="color: #F87171;">If YES: STOP. Go to Risk Off.</span>
</div>
</div>
""",
unsafe_allow_html=True,
)

with w2:
st.markdown(
"""
<div class="metric-card" style="height: 250px;">
<div style="color: #FBBF24; font-weight: bold; margin-bottom: 10px;">PHASE 2: REGIME</div>
<div style="font-size: 12px; color: #AAA;">
Identify the "Tailwind".
<br>• <b>Goldilocks:</b> Growth + Low Vol
<br>• <b>Liquidity:</b> DXY Down + Crypto Up
<br>• <b>Reflation:</b> Yields + Oil Up
</div>
</div>
""",
unsafe_allow_html=True,
)

with w3:
st.markdown(
"""
<div class="metric-card" style="height: 250px;">
<div style="color: #60A5FA; font-weight: bold; margin-bottom: 10px;">PHASE 3: SECTOR</div>
<div style="font-size: 12px; color: #AAA;">
Use the Quadrant & Sankey charts.
<br>• Find sectors moving from <b>Improving</b> to <b>Leading</b>.
<br>• Confirm flows match Regime (e.g., Risk On = Tech inflows).
</div>
</div>
""",
unsafe_allow_html=True,
)

with w4:
st.markdown(
"""
<div class="metric-card" style="height: 250px;">
<div style="color: #A78BFA; font-weight: bold; margin-bottom: 10px;">PHASE 4: TACTICS</div>
<div style="font-size: 12px; color: #AAA;">
Consult the <b>SPX Reactor</b>.
<br>• <b>Vol Check:</b> IV Rank &gt; 50?
<br>• <b>Skew Check:</b> Crash risk?
<br>• <b>Trend Check:</b> ADX &gt; 25?
</div>
</div>
""",
unsafe_allow_html=True,
)

with w5:
st.markdown(
"""
<div class="metric-card" style="height: 250px;">
<div style="color: #34D399; font-weight: bold; margin-bottom: 10px;">PHASE 5: EXECUTE</div>
<div style="font-size: 12px; color: #AAA;">
3:00 PM EST Check.
<br>• Confirm price action.
<br>• Verify DTE matches plan.
<br>• <b>Enter trade.</b>
</div>
</div>
""",
unsafe_allow_html=True,
)

# ---- TAB 3: Market Pulse ----
with tab_pulse:
st.subheader("🌊 Capital Flow: Sectors")
col_s1, col_s2 = st.columns([3, 1])
with col_s1:
st.plotly_chart(plot_sankey_sectors(market_data, timeframe), use_container_width=True)
with col_s2:
st.markdown(
"""
<div class="context-box">
<div class="context-header">How to Read</div>
<div>Sankey diagrams visualize rotation from losers (left) to winners (right) using the selected timeframe change.</div>
<br>
<div class="context-header">Analyst Note</div>
<div>If defensives are consistently on the right, risk appetite is deteriorating.</div>
</div>
""",
unsafe_allow_html=True,
)

st.divider()

st.subheader("🌍 Capital Flow: Macro Assets")
col_a1, col_a2 = st.columns([3, 1])
with col_a1:
st.plotly_chart(plot_sankey_assets(market_data, timeframe), use_container_width=True)
with col_a2:
st.markdown(
"""
<div class="context-box">
<div class="context-header">How to Read</div>
<div>Tracks relative leadership between major macro assets.</div>
<br>
<div class="context-header">Analyst Note</div>
<div>In Risk-On, you often see DXY/TLT lag and SPY/BTC lead.</div>
</div>
""",
unsafe_allow_html=True,
)

st.divider()

st.subheader("🔥 Inter-Correlation Matrix")
col_c1, col_c2 = st.columns([3, 1])
with col_c1:
st.plotly_chart(plot_correlation_heatmap(history_df, timeframe), use_container_width=True)
with col_c2:
st.markdown(
"""
<div class="context-box">
<div class="context-header">How to Read</div>
<div>Red = inverse correlation, Blue = positive correlation. Weekly view uses weekly resampled closes.</div>
<br>
<div class="context-header">Analyst Note</div>
<div>If stocks and bonds correlate positively, classic diversification can fail.</div>
</div>
""",
unsafe_allow_html=True,
)

st.divider()

st.subheader("🎯 Trend/Momentum Quadrant (RRG-Proxy)")
q1, q2 = st.columns(2)
with q1:
st.markdown("**SECTORS**")
st.plotly_chart(plot_trend_momentum_quadrant(full_hist, "SECTORS", timeframe), use_container_width=True)
with q2:
st.markdown("**MACRO ASSETS**")
st.plotly_chart(plot_trend_momentum_quadrant(full_hist, "ASSETS", timeframe), use_container_width=True)

# ---- TAB 4: Macro Machine ----
with tab_macro:
st.subheader("🕸️ The Macro Transmission Mechanism")
col_graph, col_legend = st.columns([3, 1])
with col_graph:
st.plotly_chart(plot_nexus_graph_dots(market_data, timeframe), use_container_width=True)
with col_legend:
st.markdown(
f"""
<div class="context-box" style="margin-top: 0;">
<div class="context-header">The Plumbing</div>
<div>This graph is a fixed-layout network showing macro linkages. Node color uses the selected timeframe.</div>
<br>
<div class="context-header">Current View</div>
<div><b>{timeframe}</b></div>
<br>
<div class="context-header">Data Note</div>
<div>Grey hollow nodes are invalid/missing feeds. Proxy sources are shown in hover.</div>
</div>
""",
unsafe_allow_html=True,
)

# ---- TAB 5: Playbook ----
with tab_playbook:
st.subheader("📚 Detailed Strategy Rulebook")

with st.expander("🔴 A14 (Crash Protection)", expanded=False):
st.markdown(
"""
### A14: The "Anti-Fragile" Hedge
**Concept:** Financing downside protection using OTM puts, creating a "free" crash catcher if filled for a credit.

**1. Setup & Structure (Put Broken Wing Butterfly)**
- **Long:** 1x ATM Put (e.g., 4000)
- **Short:** 2x OTM Puts (e.g., 3960 / -40 pts)
- **Long:** 1x OTM Put (e.g., 3900 / -60 pts) *(skip strikes)*

**2. Entry Protocol**
- **Time:** Friday morning (~1 hour after open)
- **DTE:** 14 days
- **Target:** net credit or very small debit

**3. Management**
- **Upside:** do nothing; keep credit
- **Downside:** tent expands into crash
- **Exit:** hard stop at **7 DTE**
"""
)

with st.expander("🟣 TIMEZONE (RUT Income)", expanded=False):
st.markdown(
"""
### TimeZone: High Prob RUT Income
**Concept:** Harvest theta on Russell using a hedged structure.

**1. Structure**
- **Leg A:** Put credit spread (sell ~14D / buy ~5D)
- **Leg B:** Put calendar (sell 15 DTE / buy 45 DTE same strike)

**2. Entry**
- **Time:** Thursday ~3:00 PM ET
- **DTE:** 15 DTE (front month)

**3. Management**
- Profit target: 5–7% of margin
- Max loss: 5% of margin
- Hard stop: exit at **7 DTE**
"""
)

with st.expander("🔵 TIMEEDGE (SPX Neutral)", expanded=False):
st.markdown(
"""
### TimeEdge: Pure Theta Decay
**Concept:** Exploit decay differential between front and back months in SPX.

**Structure**
- Double calendar: sell 15 DTE / buy ~43 DTE
- Strikes: ATM or slightly OTM

**Rules**
- Profit target: 10%
- Stop loss: 10%
- Exit: hard stop at 1 DTE (usually earlier)
"""
)

with st.expander("🌊 FLYAGONAL (Liquidity Drift)", expanded=False):
st.markdown(
"""
### Flyagonal: The Drift Catcher
**Concept:** Capture melt-up drift higher.

**Structure**
- Upside: call broken-wing butterfly (+10 / -50 / +60 width)
- Downside: put diagonal (sell front OTM / buy back OTM)

**Rules**
- Flash win: if profit > 4% in 1–2 days, close immediately
- Scratch: if no movement by 3 DTE, close
"""
)


if __name__ == "__main__":
main()

