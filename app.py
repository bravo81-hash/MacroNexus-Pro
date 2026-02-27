import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ----------------------------
# 0) SECRETS & CONFIGURATION
# ----------------------------
# INSTRUCTIONS FOR STREAMLIT CLOUD:
# 1. Go to your Streamlit App Settings -> Secrets
# 2. Add the following line:
#    FINNHUB_API_KEY = "d6bfv61r01qnr27kql40d6bfv61r01qnr27kql4g"
#
# INSTRUCTIONS FOR LOCAL DEV:
# Create a folder named `.streamlit` in your project root, and a file `.streamlit/secrets.toml`
# Add this inside secrets.toml:
# FINNHUB_API_KEY = "d6bfv61r01qnr27kql40d6bfv61r01qnr27kql4g"

FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "d6bfv61r01qnr27kql40d6bfv61r01qnr27kql4g")

APP_VERSION = "MacroNexus Pro (Hybrid Data v3)"

# Data universe
TICKERS: Dict[str, str] = {
    # Drivers
    "US10Y": "^TNX",      # YF Fallback (Index)
    "DXY": "DX-Y.NYB",    # YF Fallback (Index)
    "VIX": "^VIX",        # YF Fallback (Index)
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
    "RUT": "^RUT",        # YF Fallback (Index)
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
    "BTC": "BTC-USD",     # Finnhub mapped internally to BINANCE:BTCUSDT
}

PROXIES: Dict[str, str] = {
    "DXY": "UUP",    
    "VIX": "VIXY",   
    "RUT": "IWM",    
}

# Symbols that MUST bypass Finnhub due to free-tier constraints (Indices/Yields)
YF_MANDATORY_SYMBOLS = ["^TNX", "DX-Y.NYB", "^VIX", "^RUT"]

CRITICAL_KEYS = ["HYG", "VIX", "US10Y", "DXY"]
TIMEFRAMES = ["Tactical (Daily)", "Structural (Weekly, WTD)"]

# ----------------------------
# 1) STREAMLIT PAGE CONFIG & CSS
# ----------------------------

st.set_page_config(
    page_title="MacroNexus Pro Terminal",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0B0E11; color: #E6E6E6; font-family: 'Inter', sans-serif; }
    .metric-card {
        background: rgba(30, 34, 45, 0.6); border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .metric-card:hover { border-color: rgba(255, 255, 255, 0.2); transform: translateY(-2px); }
    .metric-label { font-size: 11px; color: #8B9BB4; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }
    .metric-value { font-size: 20px; font-weight: 700; color: #FFFFFF; margin-top: 4px; }
    .metric-delta { font-size: 12px; font-weight: 600; margin-left: 8px; }
    .strat-card {
        background: linear-gradient(180deg, rgba(22, 25, 33, 1) 0%, rgba(15, 17, 22, 1) 100%);
        border: 1px solid #2A2E39; border-radius: 12px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-bottom: 20px; height: 100%; position: relative; overflow: hidden;
    }
    .strat-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 15px; }
    .strat-title { font-size: 20px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px; }
    .strat-tag { font-size: 10px; font-weight: 700; padding: 4px 8px; border-radius: 4px; text-transform: uppercase; }
    .strat-section { margin-bottom: 15px; }
    .strat-subtitle { font-size: 11px; color: #6B7280; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 6px; }
    .strat-data { font-size: 15px; color: #E5E7EB; font-weight: 500; line-height: 1.4; }
    .regime-badge { padding: 6px 16px; border-radius: 6px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; font-size: 18px; display: inline-block; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #2A2E39; }
    .stTabs [data-baseweb="tab"] { height: 45px; border-radius: 6px 6px 0 0; border: none; color: #8B9BB4; font-weight: 600; font-size: 14px; }
    .stTabs [aria-selected="true"] { background-color: #1E222D; color: #FFF; border-bottom: 2px solid #3B82F6; }
    .badge-blue { background: rgba(59, 130, 246, 0.15); color: #60A5FA; padding: 2px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(59, 130, 246, 0.3); font-family: monospace; }
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #34D399; padding: 2px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(16, 185, 129, 0.3); font-family: monospace; }
    .badge-red { background: rgba(239, 68, 68, 0.15); color: #F87171; padding: 2px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(239, 68, 68, 0.3); font-family: monospace; }
    .badge-yellow { background: rgba(245, 158, 11, 0.15); color: #FBBF24; padding: 2px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(245, 158, 11, 0.3); font-family: monospace; }
    .badge-gray { background: rgba(107, 114, 128, 0.15); color: #9CA3AF; padding: 2px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(107, 114, 128, 0.3); font-family: monospace; }
    .context-box { background: rgba(255,255,255,0.03); border-left: 3px solid #3B82F6; padding: 15px; font-size: 13px; color: #9CA3AF; margin-top: 0px; border-radius: 0 6px 6px 0; height: 100%; }
    .context-header { font-weight: 700; color: #E5E7EB; margin-bottom: 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
    .matrix-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; color: #E5E7EB; border: 1px solid #374151; border-radius: 8px; overflow: hidden; }
    .matrix-table th { background-color: #1F2937; text-align: left; padding: 12px 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #9CA3AF; border-bottom: 1px solid #374151; }
    .matrix-table td { padding: 12px 16px; border-bottom: 1px solid #2A2E39; background-color: rgba(30, 34, 45, 0.4); }
    .matrix-table tr:hover td { background-color: rgba(59, 130, 246, 0.05); }
    .strat-name { font-weight: 700; color: #60A5FA; }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("macronexus")

# ----------------------------
# 2) DATA MODELS & PROVIDERS
# ----------------------------

@dataclass
class DataHealth:
    fetched_at_utc: str
    valid_keys: List[str]
    invalid_keys: List[str]
    proxy_used: Dict[str, str]          
    errors: Dict[str, str]              
    api_source: Dict[str, str] = field(default_factory=dict) # NEW: Tracks finnhub vs yfinance_fallback
    note: Optional[str] = None


class FinnhubDataProvider:
    """Handles rate-limited data fetching from Finnhub Free Tier"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1/stock/candle"
        
    def _map_symbol(self, symbol: str) -> str:
        # Finnhub requires specific formatting for crypto
        if symbol == "BTC-USD":
            return "BINANCE:BTCUSDT"
        return symbol
        
    def fetch_series(self, symbol: str, start_ts: int, end_ts: int) -> Tuple[Optional[pd.Series], Optional[str]]:
        """Returns (Series, ErrorMessage). Handles 429 limits gracefully."""
        finnhub_symbol = self._map_symbol(symbol)
        params = {
            "symbol": finnhub_symbol,
            "resolution": "D",
            "from": start_ts,
            "to": end_ts,
            "token": self.api_key
        }
        
        try:
            # Respect Free Plan 60 calls/min limit (small buffer to avoid burst limits)
            time.sleep(0.1) 
            
            res = requests.get(self.base_url, params=params, timeout=10)
            
            if res.status_code == 429:
                return None, "Rate limit exceeded (HTTP 429)"
            if res.status_code != 200:
                return None, f"HTTP {res.status_code}"
                
            data = res.json()
            if data.get("s") != "ok":
                return None, f"Finnhub status: {data.get('s')}"
                
            # Construct standard Pandas Series exactly as yfinance did
            dates = pd.to_datetime(data["t"], unit='s')
            series = pd.Series(data["c"], index=dates)
            
            # Normalize to tz-naive to match yfinance fallback formatting
            series.index = series.index.tz_localize(None)
            series.name = "Close"
            return series, None
            
        except Exception as e:
            return None, str(e)


def _safe_is_multiindex_columns(df: pd.DataFrame) -> bool:
    return isinstance(df.columns, pd.MultiIndex)

def _extract_close_series(df_batch: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
    """Extracts a single ticker's series from a YFinance batch download dataframe"""
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
            # Standardize index to tz-naive
            if isinstance(s.index, pd.DatetimeIndex):
                s.index = s.index.tz_localize(None)
            return s if isinstance(s, pd.Series) else None

        if "Close" in df_batch.columns:
            s = df_batch["Close"].dropna()
        elif "Adj Close" in df_batch.columns:
            s = df_batch["Adj Close"].dropna()
        else:
            return None
            
        if isinstance(s.index, pd.DatetimeIndex):
            s.index = s.index.tz_localize(None)
        return s if isinstance(s, pd.Series) else None
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
        is_friday_close = (getattr(last_date, "weekday", lambda: -1)() == 4) and (weekly.index[-1].date() == last_date.date())

        if is_friday_close:
            prev_friday = float(weekly.iloc[-2])
            return curr, prev_friday
        prev_friday = float(weekly.iloc[-2])
        return curr, prev_friday

    prev_approx = float(series.iloc[-6]) if len(series) >= 6 else float(series.iloc[-2])
    return curr, prev_approx


# ----------------------------
# 3) DATA ENGINE (HYBRID FINNHUB + YFINANCE)
# ----------------------------

@st.cache_data(ttl=300)
def fetch_market_data() -> Tuple[Dict[str, dict], pd.DataFrame, Dict[str, pd.Series], DataHealth]:
    fetched_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    
    # Timing for 1-year history
    end_dt = datetime.datetime.utcnow()
    start_dt = end_dt - datetime.timedelta(days=365)
    end_ts = int(end_dt.timestamp())
    start_ts = int(start_dt.timestamp())
    
    finnhub = FinnhubDataProvider(FINNHUB_API_KEY)
    
    # 1. Gather all required symbols
    all_symbols = list(set(list(TICKERS.values()) + list(PROXIES.values())))
    yf_batch_symbols = [sym for sym in all_symbols if sym in YF_MANDATORY_SYMBOLS]
    
    # 2. Batch download mandatory YFinance symbols (Indices)
    df_yf_batch = pd.DataFrame()
    try:
        if yf_batch_symbols:
            df_yf_batch = yf.download(
                tickers=yf_batch_symbols, period="1y", interval="1d",
                group_by="ticker", threads=True, auto_adjust=False, progress=False
            )
    except Exception as e:
        logger.error(f"YFinance batch error: {e}")

    # 3. Process the Universe Hybridly
    data: Dict[str, dict] = {}
    history_data: Dict[str, pd.Series] = {}
    full_hist: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}
    proxy_used: Dict[str, str] = {}
    api_source: Dict[str, str] = {}

    for key, primary_symbol in TICKERS.items():
        used_symbol = primary_symbol
        source_type = "primary"
        s = None
        current_api = ""

        # Strategy: Fetch logic
        def get_series_hybrid(sym: str) -> Tuple[Optional[pd.Series], str, str]:
            if sym in YF_MANDATORY_SYMBOLS:
                return _extract_close_series(df_yf_batch, sym), "yfinance_fallback", ""
            else:
                s_fh, err = finnhub.fetch_series(sym, start_ts, end_ts)
                if s_fh is not None and not s_fh.empty:
                    return s_fh, "finnhub", ""
                
                # If Finnhub fails (e.g. 429), dynamic emergency fallback to YF
                try:
                    df_emerg = yf.download(tickers=[sym], period="1y", interval="1d", auto_adjust=False, progress=False)
                    s_yf = _extract_close_series(df_emerg, sym)
                    if s_yf is not None:
                        return s_yf, "yfinance_fallback", "Finnhub failed; fell back to YF"
                except Exception:
                    pass
                return None, "failed", err or "No data from both sources"

        # Attempt Primary
        s, current_api, err_msg = get_series_hybrid(primary_symbol)

        # Attempt Proxy if needed
        if (s is None or len(s) < 50) and key in PROXIES:
            proxy_symbol = PROXIES[key]
            s_proxy, proxy_api, proxy_err = get_series_hybrid(proxy_symbol)
            if s_proxy is not None and len(s_proxy) >= 50:
                s = s_proxy
                used_symbol = proxy_symbol
                source_type = "proxy"
                proxy_used[key] = proxy_symbol
                current_api = proxy_api
            else:
                err_msg = f"{err_msg} | Proxy failed: {proxy_err}"

        # Validation
        if s is None or len(s) < 2:
            data[key] = {"valid": False, "symbol_primary": primary_symbol, "symbol_used": used_symbol, "source": source_type}
            errors[key] = err_msg or "Missing data"
            continue

        s = s.sort_index()
        s = s[~s.index.duplicated(keep='last')] # Cleanup duplicates
        
        history_data[key] = s
        full_hist[key] = s
        api_source[key] = current_api

        curr = float(s.iloc[-1])
        prev = float(s.iloc[-2]) if len(s) > 1 else curr
        curr_w, prev_w = _weekly_wtd_reference(s)

        # Zero-division guards added per previous analysis
        if key == "US10Y":
            display_price = curr / 10.0
            chg_d = (curr - prev) * 10.0
            chg_w = (curr_w - prev_w) * 10.0
            fmt = "bps"
        else:
            display_price = curr
            chg_d = ((curr - prev) / prev) * 100.0 if prev != 0 else 0.0
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
            "source": source_type,
        }

    # Compile History DataFrame cleanly
    try:
        history_df = pd.DataFrame(history_data)
        history_df = history_df.ffill() # Forward fill missing alignment gaps
    except Exception as e:
        logger.error(f"DataFrame compilation error: {e}")
        history_df = pd.DataFrame()

    valid_keys = [k for k, v in data.items() if v.get("valid")]
    invalid_keys = [k for k, v in data.items() if not v.get("valid")]

    health = DataHealth(
        fetched_at_utc=fetched_at,
        valid_keys=valid_keys,
        invalid_keys=invalid_keys,
        proxy_used=proxy_used,
        errors=errors,
        api_source=api_source,
        note="Hybrid Feed: Finnhub (Free) + YFinance (Fallback)"
    )

    return data, history_df, full_hist, health


# ----------------------------
# 4) STRATEGY DATABASE & MATRIX
# ----------------------------

STRATEGIES = {
    "GOLDILOCKS": {
        "desc": "Low Vol + Steady Trend. Market climbing wall of worry.",
        "risk": "1.5%", "bias": "Long",
        "index": {"strat": "Directional Diagonal", "dte": "Front 17 / Back 31", "setup": "Buy Back ITM / Sell Front OTM", "notes": "Stock replacement."},
        "stock": {"strat": "Call Debit Spreads", "dte": "45-60 DTE", "setup": "Buy 60D / Sell 30D", "notes": "Focus on Rel Strength Leaders."},
        "longs": "TECH, SEMIS, DISC", "shorts": "VIX, TLT",
    },
    "LIQUIDITY": {
        "desc": "High Liquidity / Dollar Weakness. Drift Up environment.",
        "risk": "1.0%", "bias": "Aggressive Long",
        "index": {"strat": "Flyagonal (Drift)", "dte": "Entry 7-10 DTE", "setup": "Upside BWB / Downside Diag", "notes": "Captures drift."},
        "stock": {"strat": "Risk Reversals", "dte": "60 DTE", "setup": "Sell OTM Put / Buy OTM Call", "notes": "Fund long delta."},
        "longs": "BTC, SEMIS, QQQ", "shorts": "DXY, CASH",
    },
    "REFLATION": {
        "desc": "Inflation / Rates Rising. Real Assets outperform Tech.",
        "risk": "1.0%", "bias": "Cyclical Long",
        "index": {"strat": "Call Spread (Cyclicals)", "dte": "45 DTE", "setup": "Buy 60D / Sell 30D", "notes": "Focus on Russell."},
        "stock": {"strat": "Cash Secured Puts", "dte": "30-45 DTE", "setup": "Sell 30D Puts on Energy/Banks", "notes": "Energy benefits from yields."},
        "longs": "ENERGY, BANKS, IND", "shorts": "TLT, TECH",
    },
    "NEUTRAL": {
        "desc": "Chop / Range Bound. No clear direction.",
        "risk": "Income Size", "bias": "Neutral/Theta",
        "index": {"strat": "TimeEdge (SPX)", "dte": "Entry 15 / Exit 7", "setup": "Put Calendar Spread", "notes": "Pure Theta play."},
        "stock": {"strat": "Iron Condor", "dte": "30-45 DTE", "setup": "Sell 20D Call / Put", "notes": "Delta neutral income."},
        "longs": "INCOME, CASH", "shorts": "MOMENTUM",
    },
    "RISK OFF": {
        "desc": "High Volatility / Credit Stress. Preservation mode.",
        "risk": "0.5%", "bias": "Short/Hedge",
        "index": {"strat": "A14 Put BWB", "dte": "Entry 14 / Exit 7", "setup": "Long ATM / Short 2x -40", "notes": "Crash Catcher."},
        "stock": {"strat": "Put Debit Spreads", "dte": "60 DTE", "setup": "Buy 40D / Sell 15D", "notes": "Directional downside."},
        "longs": "VIX, DXY", "shorts": "SPY, IWM, HYG",
    },
    "DATA ERROR": {
        "desc": "CRITICAL DATA FEED FAILURE", "risk": "0.0%", "bias": "Flat",
        "index": {"strat": "STAND ASIDE", "dte": "--", "setup": "--", "notes": "Integrity compromised."},
        "stock": {"strat": "STAND ASIDE", "dte": "--", "setup": "--", "notes": "Integrity compromised."},
        "longs": "--", "shorts": "--",
    },
}

def generate_decision_matrix_html(regime: str) -> str:
    rows = []
    def badge(t, c):
        cls = f"badge-{c}" if c in ["green", "red", "yellow", "blue"] else "badge-gray"
        return f'<span class="{cls}">{t}</span>'

    if regime == "GOLDILOCKS":
        rows = [
            ("Tech/Growth", "Call Debit Spread", badge("OPEN / ADD", "green"), "Strong trend + Low Vol. Upside uncapped."),
            ("Broad Market", "Directional Diagonals", badge("OPEN", "green"), "Stock replacement strategy works best here."),
            ("Small Caps", "Put Credit Spreads", badge("HOLD", "yellow"), "Confirm participation. If laggy, just hold."),
            ("Safe Haven", "Long Calls", badge("CLOSE", "red"), "Yields stable or rising hurts bonds."),
            ("Volatility", "Long VIX Calls", badge("FORBIDDEN", "red"), "Vol crush is active. Hedges will bleed.")
        ]
    elif regime == "LIQUIDITY":
        rows = [
            ("Tech/Growth", "Risk Reversals", badge("OPEN", "green"), "Dollar down = Tech/Crypto up. Leverage this."),
            ("Broad Market", "Flyagonal (Drift)", badge("OPEN", "green"), "Capture the overnight drift. Primary index play."),
            ("Cyclicals", "Commodity Plays", badge("HOLD", "yellow"), "Oil/Gold benefit from weak dollar."),
            ("Safe Haven", "Gold Longs", badge("OPEN", "green"), "Dollar debasement play.")
        ]
    elif regime == "REFLATION":
        rows = [
            ("Tech/Growth", "Long Duration", badge("CLOSE / REDUCE", "red"), "Rising yields hurt future valuations."),
            ("Broad Market", "Iron Condors", badge("HOLD (Wide)", "yellow"), "Market confused between earnings (Good) vs Rates (Bad)."),
            ("Cyclicals", "Cash Secured Puts", badge("OPEN", "green"), "Energy/Banks are the leaders here."),
            ("Safe Haven", "TLT Longs", badge("FORBIDDEN", "red"), "Don't fight the Fed/Bond Vigilantes.")
        ]
    elif regime == "NEUTRAL":
        rows = [
            ("Tech/Growth", "Directional", badge("CLOSE", "red"), "No trend to pay for theta."),
            ("Broad Market", "TimeEdge / TimeZone", badge("OPEN", "green"), "Theta harvest mode. Exploiting chop."),
            ("Defensives", "Dividend Plays", badge("OPEN", "green"), "Safety outperformed in chop."),
            ("Volatility", "VIX < 15?", badge("AVOID SHORTS", "red"), "Gamma risk too high if VIX wakes up.")
        ]
    elif regime == "RISK OFF":
        rows = [
            ("Tech/Growth", "Long Delta", badge("FORBIDDEN", "red"), "Do not catch falling knives."),
            ("Broad Market", "A14 (Crash Catcher)", badge("OPEN", "green"), "Only way to profit from panic safely."),
            ("Small Caps", "Credit Spreads", badge("CLOSE NOW", "red"), "Credit spreads will blow out."),
            ("Safe Haven", "TLT / Dollar", badge("OPEN", "green"), "Flight to safety trade."),
            ("Volatility", "VIX Calls", badge("OPEN", "green"), "The only asset consistently up.")
        ]
    else:
        rows = [("ALL ASSETS", "ALL STRATEGIES", badge("HALT TRADING", "red"), "Data integrity compromised.")]

    html_rows = ""
    for r in rows:
        html_rows += f"<tr><td>{r[0]}</td><td><span class='strat-name'>{r[1]}</span></td><td>{r[2]}</td><td style='color:#9CA3AF;font-style:italic;'>{r[3]}</td></tr>"
    
    return f"""<table class="matrix-table"><thead><tr><th width="15%">Asset Class</th><th width="25%">Strategy</th><th width="15%">Signal</th><th width="45%">Logic</th></tr></thead><tbody>{html_rows}</tbody></table>"""


# ----------------------------
# 5) LOGIC & HELPERS
# ----------------------------

def get_val(data: Dict[str, dict], key: str, timeframe: str) -> float:
    d = data.get(key, {})
    if not d.get("valid", False): return float("nan")
    return float(d.get("change", 0.0)) if timeframe == "Tactical (Daily)" else float(d.get("change_w", 0.0))

def determine_regime(data: Dict[str, dict], timeframe: str, strict_data: bool) -> Tuple[str, str, List[str]]:
    reasons = []
    missing_crit = [k for k in CRITICAL_KEYS if not data.get(k, {}).get("valid", False)]
    if missing_crit:
        reasons.append(f"Missing critical keys: {', '.join(missing_crit)}")
        return "DATA ERROR", "NONE", reasons

    proxy_crit = [k for k in CRITICAL_KEYS if data.get(k, {}).get("source") == "proxy"]
    if proxy_crit:
        reasons.append(f"Proxy used for critical keys: {', '.join(proxy_crit)}")
        if strict_data: return "DATA ERROR", "NONE", reasons
        confidence = "LOW"
    else:
        confidence = "HIGH"

    def g(k): 
        v = get_val(data, k, timeframe)
        return float(v) if np.isfinite(v) else 0.0

    hyg, vix, oil, cop, us10y, dxy, btc, banks = g("HYG"), g("VIX"), g("OIL"), g("COPPER"), g("US10Y"), g("DXY"), g("BTC"), g("BANKS")

    if hyg < -0.5 or vix > 5.0: return "RISK OFF", confidence, reasons
    if (oil > 1.5 or cop > 1.5) and us10y > 3.0 and banks > 0: return "REFLATION", confidence, reasons
    if dxy < -0.3 and btc > 1.5: return "LIQUIDITY", confidence, reasons
    # FIXED LOGIC: VIX changed from < 0 to < 15 to allow correct programmatic triggering
    if data.get("VIX", {}).get("price", 20.0) < 15.0 and abs(us10y) < 5.0 and hyg > -0.1: return "GOLDILOCKS", confidence, reasons

    return "NEUTRAL", confidence, reasons


# ----------------------------
# 6) VISUALS (UNCHANGED DOWNSTREAM LOGIC)
# ----------------------------

def plot_nexus_graph_dots(data: Dict[str, dict], timeframe: str) -> go.Figure:
    nodes = {
        "US10Y": {"pos": (0, 0), "label": "Rates (^TNX)"}, "DXY": {"pos": (0.8, 0.8), "label": "Dollar (DXY)"},
        "SPY": {"pos": (-0.8, 0.8), "label": "S&P 500"}, "QQQ": {"pos": (-1.2, 0.4), "label": "Nasdaq"},
        "GOLD": {"pos": (0.8, -0.8), "label": "Gold"}, "HYG": {"pos": (-0.4, -0.8), "label": "Credit (HYG)"},
        "BTC": {"pos": (-1.5, 1.5), "label": "Bitcoin"}, "OIL": {"pos": (1.5, -0.4), "label": "Oil"},
        "COPPER": {"pos": (1.2, -1.2), "label": "Copper"}, "IWM": {"pos": (-1.2, -1.0), "label": "Russell"},
        "SEMIS": {"pos": (-1.8, 0.8), "label": "Semis"}, "ENERGY": {"pos": (1.8, -0.8), "label": "Energy"},
        "HOME": {"pos": (-0.8, -0.4), "label": "Housing"}, "BANKS": {"pos": (1.5, -1.0), "label": "Banks"},
        "VIX": {"pos": (0, 1.5), "label": "Vol (^VIX)"},
    }
    edges = [("US10Y", "QQQ"), ("US10Y", "GOLD"), ("US10Y", "HOME"), ("DXY", "GOLD"), ("DXY", "OIL"), 
             ("HYG", "SPY"), ("HYG", "IWM"), ("HYG", "BANKS"), ("QQQ", "BTC"), ("QQQ", "SEMIS"), 
             ("COPPER", "US10Y"), ("OIL", "ENERGY"), ("VIX", "SPY")]

    edge_x, edge_y = [], []
    for u, v in edges:
        edge_x.extend([nodes[u]["pos"][0], nodes[v]["pos"][0], None])
        edge_y.extend([nodes[u]["pos"][1], nodes[v]["pos"][1], None])

    nx, ny, nt, ht, nc, ns, nsym = [], [], [], [], [], [], []
    for k, i in nodes.items():
        nx.append(i["pos"][0]); ny.append(i["pos"][1])
        d = data.get(k, {})
        v = get_val(data, k, timeframe)
        if not d.get("valid", False) or not np.isfinite(v):
            col, sym, fv = "#6b7280", "circle-open", "N/A"
        else:
            col = "#22c55e" if v > 0 else "#ef4444" if v < 0 else "#6b7280"
            sym, fv = "circle", f"{v:+.2f}%" if k != "US10Y" else f"{v:+.1f} bps"
        
        nc.append(col); nsym.append(sym); ns.append(45 if k in ["US10Y", "DXY", "HYG"] else 35)
        nt.append(i["label"].split("(")[0].strip())
        ht.append(f"{i['label']}<br>Change: {fv}<br>Source: {d.get('source', 'primary')} | API: {data.get('api_source', {}).get(k, 'N/A')}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#4b5563"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=nx, y=ny, mode="markers+text", text=nt, textposition="bottom center", hovertext=ht, hoverinfo="text", marker=dict(size=ns, color=nc, symbol=nsym, line=dict(width=2, color="white")), textfont=dict(size=11, color="white")))
    fig.update_layout(showlegend=False, margin=dict(b=0, l=0, r=0, t=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.0, 2.0]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500)
    return fig

def plot_correlation_heatmap(history_df: pd.DataFrame, timeframe: str) -> go.Figure:
    if history_df is None or history_df.empty: return go.Figure()
    df_calc = history_df.copy()
    if timeframe != "Tactical (Daily)": df_calc = df_calc.resample("W-FRI").last()
    corr = df_calc.pct_change().corr()
    cols = [c for c in ["US10Y", "DXY", "VIX", "HYG", "SPY", "QQQ", "IWM", "BTC", "GOLD", "OIL"] if c in corr.columns]
    if len(cols) < 2: return go.Figure()
    fig = px.imshow(corr.loc[cols, cols], text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), height=400, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def plot_trend_momentum_quadrant(full_hist: Dict[str, pd.Series], category: str, timeframe: str) -> go.Figure:
    keys = ["TECH", "SEMIS", "BANKS", "ENERGY", "HOME", "UTIL", "STAPLES", "DISC", "IND", "HEALTH", "MAT", "COMM", "RE"] if category == "SECTORS" else ["SPY", "QQQ", "IWM", "GOLD", "BTC", "TLT", "DXY", "HYG", "OIL"]
    t_win, m_win = (20, 5) if timeframe == "Tactical (Daily)" else (100, 25)
    
    items = []
    for k in keys:
        s = full_hist.get(k)
        if s is None or len(s.dropna()) < (t_win + m_win + 5): continue
        s = s.dropna()
        curr, sma, prev_mom = float(s.iloc[-1]), float(s.rolling(t_win).mean().iloc[-1]), float(s.iloc[-(m_win + 1)])
        if sma == 0 or prev_mom == 0: continue
        
        t_score, m_score = ((curr / sma) - 1.0) * 100.0, ((curr / prev_mom) - 1.0) * 100.0
        c = "#22c55e" if t_score > 0 and m_score > 0 else "#3b82f6" if t_score < 0 and m_score > 0 else "#f59e0b" if t_score > 0 and m_score < 0 else "#ef4444"
        items.append({"Symbol": k, "Trend": t_score, "Momentum": m_score, "Color": c})

    df = pd.DataFrame(items)
    if df.empty: return go.Figure()
    
    fig = px.scatter(df, x="Trend", y="Momentum", text="Symbol", color="Color", color_discrete_map="identity")
    fig.update_traces(textposition="top center", marker=dict(size=14, line=dict(width=1, color="white")))
    fig.add_hline(y=0, line_dash="dot", line_color="#555"); fig.add_vline(x=0, line_dash="dot", line_color="#555")
    fig.update_layout(xaxis=dict(zeroline=False, showgrid=True, gridcolor="#333"), yaxis=dict(zeroline=False, showgrid=True, gridcolor="#333"), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), showlegend=False, height=450, margin=dict(l=20, r=20, t=20, b=20))
    return fig


# ----------------------------
# 7) MAIN APP
# ----------------------------

def main():
    st.title(APP_VERSION)

    with st.sidebar:
        st.subheader("Data Architecture")
        st.caption("Primary: Finnhub.io (Free Tier)")
        st.caption("Fallback: Yahoo Finance (Indices)")
        st.divider()
        strict_data = st.checkbox("Strict mode (no proxies)", value=False)
        st.info("**Color Logic:** Rising VIX, Yields, or Dollar are colored Red (Risk Off).")

    with st.spinner("Connecting to MacroNexus Hybrid Engine..."):
        market_data, history_df, full_hist, health = fetch_market_data()

    # Health Banner updated to show API Sources
    c_fh = len([k for k, v in health.api_source.items() if v == 'finnhub'])
    c_yf = len([k for k, v in health.api_source.items() if v == 'yfinance_fallback'])
    
    c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
    c1.markdown(f"<span class='badge-blue'>Fetched: {health.fetched_at_utc}</span>", unsafe_allow_html=True)
    c2.markdown(f"<span class='badge-green'>Finnhub: {c_fh} | YF: {c_yf}</span>", unsafe_allow_html=True)
    c3.markdown(f"<span class='badge-blue'>Proxies: {len(health.proxy_used)}</span>", unsafe_allow_html=True)
    if health.errors: c4.markdown(f"<span class='badge-red'>Errors: {len(health.errors)}</span>", unsafe_allow_html=True)

    # Top metrics bar
    cols = st.columns(6)
    def m_tile(col, label, key):
        d = market_data.get(key, {})
        if not d.get("valid", False): return col.markdown(f"""<div class="metric-card" style="border-left: 3px solid #374151;"><div class="metric-label">{label}</div><div class="metric-value">--</div></div>""", unsafe_allow_html=True)
        val, chg = float(d.get("price", 0.0)), float(d.get("change", 0.0))
        color = ("#F43F5E" if chg > 0 else "#10B981") if key in ["US10Y", "DXY", "VIX"] else ("#10B981" if chg > 0 else "#F43F5E")
        fmt_chg = f"{chg:+.1f} bps" if key == "US10Y" else f"{chg:+.2f}%"
        val_str = f"{val:.2f}%" if key == "US10Y" else f"{val:.2f}"
        src = f" ({d.get('source')})" if d.get('source') == 'proxy' else ""
        col.markdown(f"""<div class="metric-card" style="border-left: 3px solid {color};"><div class="metric-label">{label}{src}</div><div class="metric-value">{val_str} <span class="metric-delta" style="color: {color};">{fmt_chg}</span></div></div>""", unsafe_allow_html=True)

    for i, (l, k) in enumerate([("Credit (HYG)", "HYG"), ("Volatility (VIX)", "VIX"), ("10Y Yield", "US10Y"), ("Dollar (DXY)", "DXY"), ("Oil", "OIL"), ("Bitcoin", "BTC")]):
        m_tile(cols[i], l, k)

    # Controls row
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl2: timeframe = st.selectbox("Analytic View", TIMEFRAMES, index=0, label_visibility="collapsed")
    regime, confidence, regime_reasons = determine_regime(market_data, timeframe, strict_data)

    with ctrl1: override = st.checkbox("Manual Override"); active_regime = st.selectbox("Force", list(STRATEGIES.keys()), label_visibility="collapsed") if override else regime
    with ctrl3:
        r_colors = {"GOLDILOCKS": "#10B981", "LIQUIDITY": "#A855F7", "REFLATION": "#F59E0B", "NEUTRAL": "#6B7280", "RISK OFF": "#EF4444", "DATA ERROR": "#EF4444"}
        rc = r_colors.get(active_regime, "#6B7280")
        stat = "ERROR" if active_regime == "DATA ERROR" else ("LOW CONF" if confidence == "LOW" and not override else "ACTIVE")
        st.markdown(f"""<div style="text-align: right; display: flex; align-items: center; justify-content: flex-end; gap: 15px;"><div style="text-align: right;"><div style="font-size: 10px; color: #8B9BB4; letter-spacing: 1px;">SYSTEM STATUS</div><div style="font-size: 14px; font-weight: bold; color: {rc};">{stat}</div></div><div class="regime-badge" style="background: {rc}22; color: {rc}; border: 1px solid {rc};">{active_regime}</div></div>""", unsafe_allow_html=True)

    if regime == "DATA ERROR" and not override: st.error("CRITICAL DATA FAILURE. Trading NOT recommended."); st.caption(" | ".join(regime_reasons))

    # Tabs
    t_mis, t_mat, t_pul, t_mac, t_api = st.tabs(["🚀 MISSION", "🚦 MATRIX", "📊 PULSE", "🕸️ MACRO", "🔌 API HEALTH"])

    with t_mis:
        st.info("Mission control UI identical to original logic.")
        # UI logic omitted for brevity as requested to focus on architecture

    with t_mat:
        st.subheader(f"🚦 Matrix: {active_regime}")
        st.markdown(generate_decision_matrix_html(active_regime), unsafe_allow_html=True)

    with t_pul:
        st.subheader("🔥 Inter-Correlation Matrix")
        st.plotly_chart(plot_correlation_heatmap(history_df, timeframe), use_container_width=True)
        st.subheader("🎯 Trend/Momentum Quadrant")
        q1, q2 = st.columns(2)
        with q1: st.plotly_chart(plot_trend_momentum_quadrant(full_hist, "SECTORS", timeframe), use_container_width=True)
        with q2: st.plotly_chart(plot_trend_momentum_quadrant(full_hist, "ASSETS", timeframe), use_container_width=True)

    with t_mac:
        st.subheader("🕸️ Transmission Mechanism")
        st.plotly_chart(plot_nexus_graph_dots(market_data, timeframe), use_container_width=True)

    with t_api:
        st.subheader("🔌 Hybrid Engine Telemetry")
        st.json({
            "Active Data Sources": health.api_source,
            "Proxy Substitutions": health.proxy_used,
            "Engine Errors": health.errors
        })

if __name__ == "__main__":
    main()
