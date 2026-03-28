"""
Oil Structure Analytics Board
==============================
- Fetches OHLC + TAS from 00:00:00 UTC to current time
- Refreshes every 60 seconds
- Four separate charts: Candlestick (5-min), Volume (1-min), Volume Delta (1-min), Delta Diff (1-min)
- Slider: at BOTTOM of candlestick chart, pointer starts at RIGHT (latest), slides LEFT = older
- All 4 charts sync to the same window
- Clean signal markers: ▲ green = long, ▼ red = short, ✱ = trap, ↺ = reversal
- Market Structure Bias from vol/delta/roc analysis
- No synthetic data, no spread-specific language

Run:
    pip install streamlit streamlit-autorefresh plotly pandas requests
    streamlit run monitor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Oil Structure Analytics",
    page_icon="🛢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  ENDPOINTS & AUTH
# ─────────────────────────────────────────────
BASE_OHLC = "OHLC-DATA_API"
BASE_TAS  = "TAS_DATA_API"
DEFAULT_BEARER = "TOKEN_"

MERGE_TOLERANCE_SECONDS = 120

# ─────────────────────────────────────────────
#  SIGNAL ENGINE CONSTANTS
# ─────────────────────────────────────────────
VOLUME_HIGH_MULT     = 1.5      # > 1.5×avg  → HIGH
VOLUME_LOW_MULT      = 0.8      # < 0.8×avg  → LOW
VOLUME_ROLLING_WIN   = 20
DELTA_THRESHOLD_PCT  = 0.15     # delta_threshold  = avg_vol × 0.15
ROC_THRESHOLD_PCT    = 0.10     # roc_threshold    = avg_vol × 0.10
TRAP_PRICE_CHG_MAX   = 0.0003   # 0.03% absolute price move cap for TRAP
DIVERGENCE_LOOKBACK  = 5

# ─────────────────────────────────────────────
#  CHART COLORS
# ─────────────────────────────────────────────
C = {
    "up":      "#26a69a",
    "down":    "#ef5350",
    "accent":  "#42a5f5",
    "delta":   "#ce93d8",
    "roc":     "#ffb74d",
    "neutral": "#90a4ae",
    "bg":      "#0b0f14",
    "panel":   "#0d1117",
    "grid":    "#1c2230",
    "border":  "#22304a",
    "text":    "#cdd6e0",
    "yellow":  "#ffd54f",
    "purple":  "#b39ddb",
    "trap":    "#b39ddb",
    "rev":     "#ffb74d",
}

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #0b0f14;
    color: #cdd6e0;
}
.stApp { background: #0b0f14; }
.block-container { padding: 0.6rem 1.2rem 2rem; }

[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1c2230;
}
[data-testid="stSidebar"] * { font-family: 'JetBrains Mono', monospace !important; }

h1, h2, h3 { font-family: 'JetBrains Mono', monospace !important; }

/* ── Header ── */
.monitor-header {
    display: flex; align-items: baseline; gap: 14px;
    margin-bottom: 0.4rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid #1c2230;
}
.monitor-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem; font-weight: 700;
    color: #e0eaf5; letter-spacing: 0.02em;
}
.monitor-title span { color: #42a5f5; }
.monitor-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: #37474f;
    letter-spacing: 0.08em;
}

/* ── Section labels ── */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: #37474f;
    border-left: 2px solid #1c2230;
    padding-left: 8px; margin: 0.5rem 0 0.2rem;
}

/* ── Metrics row ── */
.metric-row {
    display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 0.5rem;
}
.metric-box {
    background: #0d1117;
    border: 1px solid #1c2230;
    border-radius: 4px;
    padding: 7px 12px;
    min-width: 115px;
    flex: 1;
}
.metric-box .m-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: #37474f;
    margin-bottom: 2px;
}
.metric-box .m-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem; font-weight: 600; color: #e0eaf5;
}

/* ── Market structure badge ── */
.struct-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 5px 12px; border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.struct-BULLISH   { background: rgba(38,166,154,0.12); color: #26a69a; border: 1px solid #26a69a33; }
.struct-BEARISH   { background: rgba(239,83,80,0.12);  color: #ef5350; border: 1px solid #ef535033; }
.struct-TRAP      { background: rgba(179,157,219,0.12);color: #b39ddb; border: 1px solid #b39ddb33; }
.struct-NEUTRAL   { background: rgba(144,164,174,0.08);color: #546e7a; border: 1px solid #37474f33; }
.struct-ABSORBING { background: rgba(255,213,79,0.10); color: #ffd54f; border: 1px solid #ffd54f33; }

/* ── Signal legend ── */
.sig-legend {
    display: flex; gap: 16px; flex-wrap: wrap;
    padding: 6px 12px;
    background: #0d1117;
    border: 1px solid #1c2230;
    border-radius: 4px;
    margin-bottom: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
}
.sig-legend-item { display: flex; align-items: center; gap: 5px; color: #546e7a; }
.sig-legend-sym { font-size: 0.75rem; font-weight: 700; }

/* ── Toast ── */
.toast-wrap {
    position: fixed; bottom: 20px; right: 20px;
    z-index: 9999; width: 290px;
    display: flex; flex-direction: column-reverse; gap: 8px;
    pointer-events: none;
}
.toast {
    background: #0d1117;
    border-radius: 5px; padding: 10px 13px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; box-shadow: 0 8px 32px rgba(0,0,0,0.8);
    border-left: 3px solid #1c2230;
    animation: tIn .25s cubic-bezier(.16,1,.3,1) both, tOut .3s ease 8s forwards;
    pointer-events: auto;
}
.toast.ENTER_LONG     { border-color: #2ecc71; }
.toast.ENTER_SHORT    { border-color: #e74c3c; }
.toast.EXIT_LONG_EARLY, .toast.EXIT_SHORT_EARLY { border-color: #ffb86c; }
.toast.CONFIRM_EXIT   { border-color: #ff9e5e; }
.toast.TRAP           { border-color: #bb86fc; }
.t-head { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
.t-sym  { font-size: 1rem; line-height: 1; }
.t-label { font-weight: 700; font-size: 0.72rem; color: #e0eaf5; flex: 1; letter-spacing:.04em; }
.t-conf  { font-size: 0.6rem; color: #37474f; }
.t-body  { color: #546e7a; font-size: 0.63rem; line-height: 1.55; }
.t-pills { display: flex; gap: 6px; margin-top: 5px; flex-wrap: wrap; }
.t-pill  { font-size: 0.58rem; color: #42a5f5; background: rgba(66,165,245,0.06);
           border-radius: 2px; padding: 1px 5px; border: 1px solid #1c2230; }

@keyframes tIn  { from { opacity:0; transform:translateX(20px); } to { opacity:1; transform:none; } }
@keyframes tOut { from { opacity:1; } to { opacity:0; height:0; padding:0; margin:0; } }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  SIGNAL ENGINE  (enhanced)
# ═════════════════════════════════════════════

@dataclass
class SignalResult:
    signal:       str
    confidence:   float
    reason:       str
    volume_state: str
    delta:        float
    roc_delta:    float
    divergence:   Optional[str]
    vap_coverage: float

    def to_dict(self): return asdict(self)


def _rolling_avg_vol(df: pd.DataFrame) -> float:
    """Rolling 20-bar mean of volume; returns 1.0 if unavailable to avoid /0."""
    vs = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    val = vs.rolling(VOLUME_ROLLING_WIN, min_periods=3).mean().iloc[-1]
    return float(val) if (val and val > 0) else 1.0


def _volume_state(volume: float, avg_vol: float) -> str:
    if avg_vol <= 0:
        return "LOW"
    r = volume / avg_vol
    if r > VOLUME_HIGH_MULT:
        return "HIGH"
    if r >= VOLUME_LOW_MULT:
        return "MEDIUM"
    return "LOW"


def _roc_state(roc_delta: float, roc_threshold: float) -> str:
    if roc_delta >  roc_threshold:  return "+"
    if roc_delta < -roc_threshold:  return "-"
    return "flat"


def _divergence(df: pd.DataFrame) -> Optional[str]:
    """
    Compare price direction vs delta direction over the last DIVERGENCE_LOOKBACK bars.
    Returns 'BEARISH', 'BULLISH', or None.
    Bearish: price up but delta down  → exhaustion / distribution
    Bullish: price down but delta up  → absorption / accumulation
    """
    n = DIVERGENCE_LOOKBACK
    if len(df) < n + 1 or "delta" not in df.columns:
        return None
    window = df.tail(n + 1).dropna(subset=["delta", "close"])
    if len(window) < 2:
        return None
    p_old = float(window["close"].iloc[0])
    p_new = float(window["close"].iloc[-1])
    d_old = float(window["delta"].iloc[0])
    d_new = float(window["delta"].iloc[-1])
    if p_new > p_old and d_new < d_old:
        return "BEARISH"
    if p_new < p_old and d_new > d_old:
        return "BULLISH"
    return None


def _confidence(vol_state: str, delta: float, delta_thr: float,
                roc_st: str, div: Optional[str]) -> float:
    """
    Confidence model (additive, capped at 1.0):
        +0.30  volume == HIGH
        +0.20  |delta| > delta_threshold
        +0.20  roc_state in ['+', '-']  (directional momentum)
        +0.30  divergence present
    """
    score = 0.0
    if vol_state == "HIGH":         score += 0.30
    if abs(delta) > delta_thr:      score += 0.20
    if roc_st in ("+", "-"):        score += 0.20
    if div is not None:             score += 0.30
    return min(score, 1.0)


def _no_signal(coverage: float) -> SignalResult:
    return SignalResult(
        signal="NO_SIGNAL", confidence=0.0,
        reason="No edge condition met.", volume_state="UNKNOWN",
        delta=0.0, roc_delta=0.0, divergence=None, vap_coverage=coverage,
    )


def run_signal_engine(df: pd.DataFrame) -> SignalResult:
    """
    Enhanced intraday signal engine.

    Priority order (first match wins):
        1. TRAP             – high vol, strong delta, no price movement
        2. CONFIRM_EXIT     – delta flip + high vol + divergence confirmed
        3a. EXIT_LONG_EARLY  – weakening long momentum
        3b. EXIT_SHORT_EARLY – weakening short momentum
        4a. ENTER_LONG       – clean high-vol buy setup
        4b. ENTER_SHORT      – clean high-vol sell setup
        DEFAULT: NO_SIGNAL

    Hard constraints:
        - Returns DATA_UNAVAILABLE if vap_coverage == 0 or delta_source == UNAVAILABLE
        - Never generates signals from price alone
        - No lagging indicators (RSI, MACD, etc.)
        - Lightweight: runs in O(n) on last 20 bars only
    """
    # ── guard: need at least 3 bars ───────────────────────────────────────────
    if df is None or df.empty or len(df) < 3:
        return _no_signal(0.0)

    # ── guard: TAS data required ──────────────────────────────────────────────
    coverage     = float(df["vap_coverage"].iloc[-1]) \
                   if "vap_coverage" in df.columns else 0.0
    delta_source = df["delta_source"].iloc[-1] \
                   if "delta_source" in df.columns else "UNAVAILABLE"

    if coverage == 0.0 or delta_source == "UNAVAILABLE":
        return SignalResult(
            signal="DATA_UNAVAILABLE", confidence=0.0,
            reason="TAS data unavailable – all signals suppressed.",
            volume_state="UNKNOWN", delta=0.0, roc_delta=0.0,
            divergence=None, vap_coverage=coverage,
        )

    # ── extract latest and previous rows ─────────────────────────────────────
    latest     = df.iloc[-1]
    prev       = df.iloc[-2]

    volume     = float(latest.get("volume",    0) or 0)
    delta      = float(latest.get("delta",     0) or 0)
    roc_val    = float(latest.get("roc_delta", 0) or 0)
    prev_delta = float(prev.get("delta",       0) or 0)

    close_now  = float(latest.get("close", 0) or 0)
    close_prev = float(prev.get("close",   0) or 0)

    # ── derived thresholds (scale with volume regime) ─────────────────────────
    avg_vol   = _rolling_avg_vol(df)
    delta_thr = avg_vol * DELTA_THRESHOLD_PCT
    roc_thr   = avg_vol * ROC_THRESHOLD_PCT

    # ── feature states ────────────────────────────────────────────────────────
    vol_state = _volume_state(volume, avg_vol)
    roc_st    = _roc_state(roc_val, roc_thr)
    div       = _divergence(df)

    # absolute fractional price change between last two closes
    price_chg = abs(close_now - close_prev) / (abs(close_prev) + 1e-9)

    # delta sign flip: previous and current bar on opposite sides of zero
    delta_flipped = (prev_delta > 0 and delta < 0) or \
                    (prev_delta < 0 and delta > 0)

    # ROC is "strongly opposite" to the prior delta direction
    # (e.g. was trending long → ROC now negative)
    strong_opp_roc = (prev_delta > 0 and roc_st == "-") or \
                     (prev_delta < 0 and roc_st == "+")

    # ── shared confidence score ───────────────────────────────────────────────
    conf = _confidence(vol_state, delta, delta_thr, roc_st, div)

    # ══════════════════════════════════════════════════════════════════════════
    #  PRIORITY 1 — TRAP (absorption)
    #
    #  Conditions:
    #    · volume_state == HIGH         (unusual participation)
    #    · |delta| > delta_threshold    (aggressive directional flow)
    #    · price_change < 0.03%         (price barely reacts)
    #    · roc_state in ['+', '-']      (momentum present but absorbed)
    #
    #  Interpretation: smart money is absorbing aggressive market orders.
    #  The directional flow is real but is being neutralised by a large
    #  counter-party — chasing this bar is dangerous.
    # ══════════════════════════════════════════════════════════════════════════
    if (
        vol_state == "HIGH"
        and abs(delta) > delta_thr
        and price_chg < TRAP_PRICE_CHG_MAX
        and roc_st in ("+", "-")
    ):
        return SignalResult(
            signal="TRAP",
            confidence=min(conf + 0.10, 1.0),
            reason=(
                f"High vol, strong delta ({delta:+.0f}) with price barely "
                f"moving ({price_chg * 100:.4f}%). "
                "Aggressive flow absorbed — avoid chasing this move."
            ),
            volume_state=vol_state,
            delta=delta, roc_delta=roc_val,
            divergence=div, vap_coverage=coverage,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  PRIORITY 2 — CONFIRM EXIT (reversal fully confirmed)
    #
    #  Conditions:
    #    · volume_state == HIGH         (conviction in the reversal)
    #    · delta_flipped                (order flow direction changed)
    #    · strong_opp_roc               (momentum now counter to prior trend)
    #    · divergence is not None       (price/delta already diverging)
    #
    #  Interpretation: all three layers — flow, momentum, divergence — agree
    #  the prior trend has ended.  Full exit is warranted.
    # ══════════════════════════════════════════════════════════════════════════
    if (
        vol_state == "HIGH"
        and delta_flipped
        and strong_opp_roc
        and div is not None
    ):
        return SignalResult(
            signal="CONFIRM_EXIT",
            confidence=conf,
            reason=(
                f"Delta flipped ({prev_delta:+.0f} → {delta:+.0f}), "
                f"HIGH volume, ROC strongly opposite, "
                f"{div} divergence confirmed. Full exit warranted."
            ),
            volume_state=vol_state,
            delta=delta, roc_delta=roc_val,
            divergence=div, vap_coverage=coverage,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  PRIORITY 3a — EXIT LONG EARLY
    #
    #  Conditions:
    #    · delta > 0        (flow still net buy — no flip yet)
    #    · roc_state == '-' (momentum decelerating)
    #    · divergence == 'BEARISH' (price moved up but delta weakening)
    #
    #  Interpretation: longs have not yet been proven wrong by a delta flip,
    #  but early warning signs are stacking.  Trim before the flip.
    # ══════════════════════════════════════════════════════════════════════════
    if delta > 0 and roc_st == "-" and div == "BEARISH":
        return SignalResult(
            signal="EXIT_LONG_EARLY",
            confidence=conf,
            reason=(
                "Delta still positive but ROC has turned negative "
                "with BEARISH price/delta divergence. "
                "Long momentum fading — trim or exit longs early."
            ),
            volume_state=vol_state,
            delta=delta, roc_delta=roc_val,
            divergence=div, vap_coverage=coverage,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  PRIORITY 3b — EXIT SHORT EARLY
    #
    #  Conditions:
    #    · delta < 0        (flow still net sell — no flip yet)
    #    · roc_state == '+' (momentum recovering)
    #    · divergence == 'BULLISH' (price moved down but delta recovering)
    #
    #  Interpretation: shorts have not yet been proven wrong by a delta flip,
    #  but early warning signs are stacking.  Cover before the flip.
    # ══════════════════════════════════════════════════════════════════════════
    if delta < 0 and roc_st == "+" and div == "BULLISH":
        return SignalResult(
            signal="EXIT_SHORT_EARLY",
            confidence=conf,
            reason=(
                "Delta still negative but ROC has recovered "
                "with BULLISH price/delta divergence. "
                "Short momentum fading — cover or reduce shorts early."
            ),
            volume_state=vol_state,
            delta=delta, roc_delta=roc_val,
            divergence=div, vap_coverage=coverage,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  PRIORITY 4a — ENTER LONG
    #
    #  Conditions:
    #    · volume_state == HIGH     (institutional participation)
    #    · delta > delta_threshold  (net buy flow above noise)
    #    · roc_state == '+'         (buy momentum accelerating)
    #    · divergence is None       (no contrary warning)
    #
    #  Interpretation: clean trend initiation — all three confirmations
    #  aligned long with no early-exit warning present.
    # ══════════════════════════════════════════════════════════════════════════
    if (
        vol_state == "HIGH"
        and delta > delta_thr
        and roc_st == "+"
        and div is None
    ):
        return SignalResult(
            signal="ENTER_LONG",
            confidence=conf,
            reason=(
                f"High vol, strong buy delta ({delta:+.0f}), "
                "positive ROC, no divergence. Clean long setup."
            ),
            volume_state=vol_state,
            delta=delta, roc_delta=roc_val,
            divergence=None, vap_coverage=coverage,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  PRIORITY 4b — ENTER SHORT
    #
    #  Conditions:
    #    · volume_state == HIGH      (institutional participation)
    #    · delta < -delta_threshold  (net sell flow above noise)
    #    · roc_state == '-'          (sell momentum accelerating)
    #    · divergence is None        (no contrary warning)
    #
    #  Interpretation: clean trend initiation — all three confirmations
    #  aligned short with no early-exit warning present.
    # ══════════════════════════════════════════════════════════════════════════
    if (
        vol_state == "HIGH"
        and delta < -delta_thr
        and roc_st == "-"
        and div is None
    ):
        return SignalResult(
            signal="ENTER_SHORT",
            confidence=conf,
            reason=(
                f"High vol, strong sell delta ({delta:+.0f}), "
                "negative ROC, no divergence. Clean short setup."
            ),
            volume_state=vol_state,
            delta=delta, roc_delta=roc_val,
            divergence=None, vap_coverage=coverage,
        )

    # ── default ───────────────────────────────────────────────────────────────
    return _no_signal(coverage)


# ─────────────────────────────────────────────
#  SIGNAL META  (clean symbols, no clutter)
# ─────────────────────────────────────────────
#  ▲  = enter long   (green triangle up)
#  ▼  = enter short  (red triangle down)
#  ✱  = trap
#  ↺  = reversal (exit early / confirm exit)
SIGNAL_META = {
    "ENTER_LONG":       {"sym":"▲", "color":"#2ecc71", "label":"LONG",     "marker":"triangle-up",   "size":14},
    "ENTER_SHORT":      {"sym":"▼", "color":"#e74c3c", "label":"SHORT",    "marker":"triangle-down", "size":14},
    "EXIT_LONG_EARLY":  {"sym":"↺", "color":"#ffb86c", "label":"REV",      "marker":"circle-open",   "size":10},
    "EXIT_SHORT_EARLY": {"sym":"↺", "color":"#ffb86c", "label":"REV",      "marker":"circle-open",   "size":10},
    "CONFIRM_EXIT":     {"sym":"↺", "color":"#ff9e5e", "label":"REV",      "marker":"circle-open",   "size":10},
    "TRAP":             {"sym":"✱", "color":"#bb86fc", "label":"TRAP",     "marker":"x",             "size":12},
    "DATA_UNAVAILABLE": {"sym":"○", "color":"#546e7a", "label":"NO DATA",  "marker":"circle",        "size":6},
    "NO_SIGNAL":        {"sym":"·", "color":"#263238", "label":"–",        "marker":"circle",        "size":4},
}


# ─────────────────────────────────────────────
#  MARKET STRUCTURE BIAS
# ─────────────────────────────────────────────

def compute_market_structure(df: pd.DataFrame) -> Tuple[str, str]:
    if df is None or df.empty or len(df) < 5:
        return "NEUTRAL", "Insufficient data."

    coverage = float(df["vap_coverage"].iloc[-1]) if "vap_coverage" in df.columns else 0.0
    if coverage == 0.0:
        return "NEUTRAL", "No TAS data – structure indeterminate."

    tail  = df.tail(20).copy()
    delta = pd.to_numeric(tail.get("delta",     pd.Series(dtype=float)), errors="coerce").fillna(0)
    vol   = pd.to_numeric(tail.get("volume",    pd.Series(dtype=float)), errors="coerce").fillna(0)
    roc   = pd.to_numeric(tail.get("roc_delta", pd.Series(dtype=float)), errors="coerce").fillna(0)
    close = pd.to_numeric(tail.get("close",     pd.Series(dtype=float)), errors="coerce").ffill()

    avg_vol       = float(vol.mean()) if vol.mean() > 0 else 1.0
    cum_delta     = float(delta.sum())
    avg_roc       = float(roc.mean())
    price_chg     = float((close.iloc[-1] - close.iloc[0]) / (abs(close.iloc[0]) + 1e-9))
    high_vol_bars = int((vol > avg_vol * VOLUME_HIGH_MULT).sum())

    if high_vol_bars >= 3 and abs(price_chg) < 0.0005:
        return "ABSORBING", (
            f"Sustained high-vol activity ({high_vol_bars} bars) with minimal price displacement "
            f"({price_chg*100:.3f}%). Large player absorption detected."
        )
    if cum_delta > 0 and avg_roc > 0 and price_chg > 0.0001:
        return "BULLISH", (
            f"Cum delta {cum_delta:+.0f}  ·  avg ROC {avg_roc:+.3f}  ·  "
            f"price +{price_chg*100:.3f}% over last {len(tail)} bars. Buyers in control."
        )
    if cum_delta < 0 and avg_roc < 0 and price_chg < -0.0001:
        return "BEARISH", (
            f"Cum delta {cum_delta:+.0f}  ·  avg ROC {avg_roc:+.3f}  ·  "
            f"price {price_chg*100:.3f}% over last {len(tail)} bars. Sellers in control."
        )
    if (cum_delta > 0 and price_chg < -0.0001) or (cum_delta < 0 and price_chg > 0.0001):
        return "TRAP", (
            f"Divergence: price {'↑' if price_chg > 0 else '↓'}{abs(price_chg)*100:.3f}% "
            f"but cum delta {cum_delta:+.0f}. Potential trap."
        )
    return "NEUTRAL", (
        f"Cum delta {cum_delta:+.0f}  ·  avg ROC {avg_roc:.3f}  ·  "
        f"price {price_chg*100:.3f}%.  No clear directional conviction."
    )


# ─────────────────────────────────────────────
#  API HELPERS
# ─────────────────────────────────────────────

def _hdrs(token: str) -> Dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def fetch_ohlc(instrument: str, start_ts: int, end_ts: int, token: str) -> Tuple[pd.DataFrame, str]:
    params = {
        "instruments": instrument,
        "interval":    "1M",
        "start":       str(start_ts),
        "end":         str(end_ts),
    }
    try:
        r = requests.get(BASE_OHLC, headers=_hdrs(token), params=params, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame(), f"HTTP {r.status_code}: {r.text[:400]}"
        data = r.json()
    except Exception as e:
        return pd.DataFrame(), str(e)

    def _rows_to_df(rows):
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("time").reset_index(drop=True)

    if isinstance(data, dict):
        rows = data.get(instrument, [])
        if not rows:
            for v in data.values():
                if isinstance(v, list):
                    rows = v; break
        return _rows_to_df(rows), ""
    elif isinstance(data, list):
        return _rows_to_df(data), ""
    return pd.DataFrame(), "Unexpected response format."


def fetch_tas(instrument: str, date_str: str, start_time: str, end_time: str,
              token: str) -> Tuple[pd.DataFrame, str]:
    payload = {
        "products": [{
            "id":    instrument,
            "dates": [date_str],
            "start": start_time,
            "end":   end_time,
        }]
    }
    all_trades, url = [], BASE_TAS
    while True:
        try:
            r = requests.post(url, headers=_hdrs(token), json=payload, timeout=30)
            if r.status_code != 200:
                return pd.DataFrame(), f"HTTP {r.status_code}: {r.text[:400]}"
            data = r.json()
        except Exception as e:
            return pd.DataFrame(), str(e)

        trades = data.get("data", [])
        if trades: all_trades.extend(trades)
        url = data.get("next")
        if not url: break

    if not all_trades:
        return pd.DataFrame(), "No TAS data."

    df = pd.DataFrame(all_trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ["price","qty","side"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["price","qty","side"], inplace=True)
    return df.sort_values("timestamp").reset_index(drop=True), ""


def aggregate_tas_1min(tas_df: pd.DataFrame) -> pd.DataFrame:
    if tas_df.empty:
        return pd.DataFrame(columns=["time","buy_volume","sell_volume","delta","total_volume"])
    df = tas_df.copy()
    df["minute"] = df["timestamp"].dt.floor("1min")
    buy  = df[df["side"]==1].groupby("minute")["qty"].sum().rename("buy_volume")
    sell = df[df["side"]==-1].groupby("minute")["qty"].sum().rename("sell_volume")
    agg  = pd.DataFrame(index=buy.index.union(sell.index)).join(buy,how="outer").join(sell,how="outer").fillna(0)
    agg["delta"]        = agg["buy_volume"] - agg["sell_volume"]
    agg["total_volume"] = agg["buy_volume"] + agg["sell_volume"]
    agg.reset_index(inplace=True)
    agg.rename(columns={"minute":"time"}, inplace=True)
    return agg.sort_values("time").reset_index(drop=True)


# ─────────────────────────────────────────────
#  MERGE OHLC + TAS (1-min)
# ─────────────────────────────────────────────

def merge_1min(ohlc_df: pd.DataFrame, tas_agg: pd.DataFrame) -> pd.DataFrame:
    if ohlc_df.empty: return pd.DataFrame()
    df = ohlc_df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    if tas_agg.empty:
        df["delta"]        = np.nan
        df["total_volume"] = np.nan
        df["buy_volume"]   = np.nan
        df["sell_volume"]  = np.nan
        df["vap_coverage"] = 0.0
        df["delta_source"] = "UNAVAILABLE"
        df["roc_delta"]    = 0.0
        df["delta_diff"]   = 0.0
        return df

    ta = tas_agg.copy()
    ta["time"] = pd.to_datetime(ta["time"], utc=True)

    merged = pd.merge_asof(
        df.sort_values("time"),
        ta[["time","buy_volume","sell_volume","delta","total_volume"]].sort_values("time"),
        on="time", direction="nearest",
        tolerance=pd.Timedelta(seconds=MERGE_TOLERANCE_SECONDS),
    )

    matched  = merged["delta"].notna().sum()
    total    = len(merged)
    coverage = matched / total if total > 0 else 0.0
    merged["vap_coverage"] = coverage

    if coverage == 0:
        merged["delta"] = merged["total_volume"] = np.nan
        merged["delta_source"] = "UNAVAILABLE"
    elif coverage < 0.6:
        merged["delta_source"] = "LOW_CONFIDENCE"
        merged["delta"]        = merged["delta"].ffill()
        merged["total_volume"] = merged["total_volume"].ffill()
    else:
        merged["delta_source"] = "HIGH_CONFIDENCE"
        merged["delta"]        = merged["delta"].ffill()
        merged["total_volume"] = merged["total_volume"].ffill()

    eps     = 1e-6
    d_shift = merged["delta"].shift(1)
    denom   = d_shift.abs().replace(0, np.nan).fillna(eps)
    merged["roc_delta"] = ((merged["delta"] - d_shift) / denom).replace([np.inf,-np.inf], 0).fillna(0)
    merged["delta_diff"] = merged["delta"].diff().fillna(0)

    return merged.reset_index(drop=True)


# ─────────────────────────────────────────────
#  RESAMPLE TO 5-MIN
# ─────────────────────────────────────────────

def resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "time" not in df.columns: return pd.DataFrame()
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d.set_index("time").sort_index()
    agg = {"open":"first","high":"max","low":"min","close":"last"}
    if "volume" in d.columns: agg["volume"] = "sum"
    out = d[list(agg)].resample("5min").agg(agg).dropna(how="all").reset_index()
    return out


# ─────────────────────────────────────────────
#  FULL DATA LOAD
# ─────────────────────────────────────────────

def load_full_day(instrument: str, token: str) -> Tuple[pd.DataFrame, str]:
    now_utc   = datetime.now(timezone.utc)
    day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    start_ts = int(day_start.timestamp())
    end_ts   = int(now_utc.timestamp())

    ohlc_df, err = fetch_ohlc(instrument, start_ts, end_ts, token)
    if err or ohlc_df.empty:
        return pd.DataFrame(), err or "No OHLC data returned."

    date_str   = day_start.strftime("%Y-%m-%d")
    start_time = "00:00:00"
    end_time   = now_utc.strftime("%H:%M:%S")

    tas_raw, tas_err = fetch_tas(instrument, date_str, start_time, end_time, token)
    if tas_err:
        st.warning(f"TAS unavailable: {tas_err}. Volume delta disabled.")
        return merge_1min(ohlc_df, pd.DataFrame()), ""

    tas_agg = aggregate_tas_1min(tas_raw)
    merged  = merge_1min(ohlc_df, tas_agg)
    return merged, ""


def load_incremental(instrument: str, last_ts: datetime, token: str) -> Tuple[pd.DataFrame, str]:
    now_utc    = datetime.now(timezone.utc)
    start_ts   = int(last_ts.timestamp()) + 1
    end_ts     = int(now_utc.timestamp())

    if end_ts <= start_ts:
        return pd.DataFrame(), ""

    ohlc_df, err = fetch_ohlc(instrument, start_ts, end_ts, token)
    if err or ohlc_df.empty:
        return pd.DataFrame(), err

    date_str   = last_ts.strftime("%Y-%m-%d")
    start_time = last_ts.strftime("%H:%M:%S")
    end_time   = now_utc.strftime("%H:%M:%S")

    tas_raw, tas_err = fetch_tas(instrument, date_str, start_time, end_time, token)
    if tas_err:
        return merge_1min(ohlc_df, pd.DataFrame()), ""

    tas_agg = aggregate_tas_1min(tas_raw)
    merged  = merge_1min(ohlc_df, tas_agg)
    return merged, ""


# ─────────────────────────────────────────────
#  HISTORICAL SIGNAL SWEEP
# ─────────────────────────────────────────────

def compute_all_signals(df: pd.DataFrame) -> List[Dict]:
    results = []
    for i in range(3, len(df)):
        win = df.iloc[:i+1]
        sig = run_signal_engine(win)
        if sig.signal not in ("NO_SIGNAL","DATA_UNAVAILABLE"):
            results.append({
                **sig.to_dict(),
                "time":  win["time"].iloc[-1],
                "close": float(win["close"].iloc[-1]),
            })
    return results


# ─────────────────────────────────────────────
#  CHART LAYOUT HELPER
# ─────────────────────────────────────────────

def _layout(fig: go.Figure, height: int = 400, show_rangeslider: bool = False):
    """Apply consistent dark layout. No title embedded in chart."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["panel"],
        plot_bgcolor=C["panel"],
        height=height,
        margin=dict(l=55, r=18, t=12, b=36 if not show_rangeslider else 8),
        font=dict(family="JetBrains Mono", color="#546e7a", size=9),
        showlegend=False,
        xaxis=dict(
            gridcolor=C["grid"],
            zerolinecolor=C["grid"],
            showspikes=True,
            spikecolor=C["border"],
            spikemode="across",
            spikesnap="cursor",
            rangeslider=dict(visible=show_rangeslider, bgcolor=C["bg"], thickness=0.06),
        ),
        yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=C["panel"],
            bordercolor=C["border"],
            font=dict(family="JetBrains Mono", size=9, color=C["text"]),
        ),
    )


def _apply_xrange(fig: go.Figure, df: pd.DataFrame, x_start: float, x_end: float):
    t0 = pd.Timestamp(x_start, unit="s", tz="UTC").isoformat()
    t1 = pd.Timestamp(x_end,   unit="s", tz="UTC").isoformat()
    fig.update_xaxes(range=[t0, t1])


# ─────────────────────────────────────────────
#  CHART 1: 5-min Candlestick  (with rangeslider at bottom)
# ─────────────────────────────────────────────

def chart_candles(df_1min: pd.DataFrame, signals: List[Dict],
                  instrument: str, x_start: float, x_end: float) -> go.Figure:
    fig = go.Figure()
    df5 = resample_5min(df_1min)

    if df5.empty:
        fig.add_annotation(text="No OHLC data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=C["neutral"], size=13, family="JetBrains Mono"))
        _layout(fig, 420, show_rangeslider=True)
        return fig

    fig.add_trace(go.Candlestick(
        x=df5["time"],
        open=df5["open"], high=df5["high"],
        low=df5["low"],   close=df5["close"],
        increasing=dict(line=dict(color=C["up"],  width=1.5), fillcolor=C["up"]),
        decreasing=dict(line=dict(color=C["down"],width=1.5), fillcolor=C["down"]),
        name="5‑min",
        whiskerwidth=0.6,
        hoverinfo="x+y",
    ))

    if signals:
        _add_markers_clean(fig, signals, df5)

    _layout(fig, 420, show_rangeslider=True)
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                bgcolor=C["bg"],
                bordercolor=C["border"],
                borderwidth=1,
                thickness=0.07,
            ),
            range=[
                pd.Timestamp(x_start, unit="s", tz="UTC").isoformat(),
                pd.Timestamp(x_end,   unit="s", tz="UTC").isoformat(),
            ],
        )
    )
    return fig


def _add_markers_clean(fig: go.Figure, signals: List[Dict], df5: pd.DataFrame):
    df5i = df5.copy()
    df5i["time"] = pd.to_datetime(df5i["time"], utc=True).dt.floor("5min")
    df5i = df5i.set_index("time").sort_index()

    buckets = {
        "ENTER_LONG":       {"traces":[], "marker":"triangle-up",    "col":"#2ecc71", "size":14, "offset":-0.002},
        "ENTER_SHORT":      {"traces":[], "marker":"triangle-down",  "col":"#e74c3c", "size":14, "offset": 0.002},
        "TRAP":             {"traces":[], "marker":"x",              "col":"#bb86fc", "size":12, "offset": 0.0025},
        "EXIT_LONG_EARLY":  {"traces":[], "marker":"circle-open",    "col":"#ffb86c", "size":10, "offset": 0.0025},
        "EXIT_SHORT_EARLY": {"traces":[], "marker":"circle-open",    "col":"#ffb86c", "size":10, "offset":-0.002},
        "CONFIRM_EXIT":     {"traces":[], "marker":"circle-open",    "col":"#ff9e5e", "size":10, "offset": 0.0025},
    }

    seen = {}
    for s in signals:
        sig = s.get("signal","")
        if sig not in buckets: continue
        t5 = pd.to_datetime(s["time"], utc=True).floor("5min")
        key = (sig, t5)
        if key in seen: continue
        seen[key] = s

    for s in seen.values():
        sig = s.get("signal","")
        b   = buckets[sig]
        t5  = pd.to_datetime(s["time"], utc=True).floor("5min")
        if t5 in df5i.index:
            ref_price = (float(df5i.loc[t5,"low"])  if b["offset"] < 0
                        else float(df5i.loc[t5,"high"]))
        else:
            ref_price = float(s.get("close",0) or 0)
        y = ref_price * (1 + b["offset"])
        b["traces"].append((t5, y, f"{SIGNAL_META[sig]['label']}  conf {s['confidence']*100:.0f}%"))

    for sig, b in buckets.items():
        if not b["traces"]: continue
        xs, ys, labels = zip(*b["traces"])
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(ys),
            mode="markers",
            marker=dict(
                symbol=b["marker"],
                color=b["col"],
                size=b["size"],
                line=dict(color="#f0f0f0", width=2),
                opacity=0.95,
            ),
            hovertemplate="<b>%{customdata}</b><br>%{x|%H:%M}<extra></extra>",
            customdata=labels,
            name=SIGNAL_META[sig]["label"],
        ))


# ─────────────────────────────────────────────
#  CHART 2: Volume (OHLC)
# ─────────────────────────────────────────────

def chart_volume(df: pd.DataFrame, x_start: float, x_end: float) -> go.Figure:
    fig = go.Figure()
    if df.empty or "volume" not in df.columns:
        fig.add_annotation(text="No volume data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=C["neutral"],size=11))
        _layout(fig, 240)
        return fig

    vol    = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    colors = [C["up"] if float(r.close or 0) >= float(r.open or 0) else C["down"]
              for _, r in df.iterrows()]

    fig.add_trace(go.Bar(x=df["time"], y=vol,
                         marker_color=colors, opacity=0.75,
                         hovertemplate="%{y:,.0f}<extra>vol</extra>"))
    fig.add_trace(go.Scatter(x=df["time"], y=vol.rolling(5,min_periods=1).mean(),
                             line=dict(color=C["accent"], width=1.5, dash="dot"),
                             hoverinfo="skip"))
    _layout(fig, 240)
    _apply_xrange(fig, df, x_start, x_end)
    return fig


# ─────────────────────────────────────────────
#  CHART 3: Volume Delta (TAS)
# ─────────────────────────────────────────────

def chart_volume_delta(df: pd.DataFrame, x_start: float, x_end: float) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    coverage     = float(df["vap_coverage"].iloc[-1]) if "vap_coverage" in df.columns else 0.0
    delta_source = df["delta_source"].iloc[-1]         if "delta_source" in df.columns else "UNAVAILABLE"

    if df.empty or coverage == 0 or delta_source == "UNAVAILABLE":
        fig.add_annotation(text="TAS data unavailable", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=C["neutral"],size=11))
        _layout(fig, 240)
        return fig

    delta  = df["delta"].fillna(0)
    colors = [C["up"] if v >= 0 else C["down"] for v in delta]
    cum    = delta.cumsum()
    cum_c  = C["up"] if (len(cum)>0 and cum.iloc[-1]>=0) else C["down"]

    fig.add_trace(go.Bar(x=df["time"], y=delta,
                         marker_color=colors, opacity=0.8,
                         hovertemplate="%{y:+,.0f}<extra>Δ</extra>"),
                  secondary_y=False)
    fig.add_hline(y=0, line_color=C["grid"], line_width=1)
    fig.add_trace(go.Scatter(x=df["time"], y=cum,
                             line=dict(color=cum_c, width=1.8, dash="dot"),
                             hovertemplate="%{y:+,.0f}<extra>cum Δ</extra>"),
                  secondary_y=True)

    _layout(fig, 240)
    fig.update_yaxes(title_text="Δ", gridcolor=C["grid"],  secondary_y=False,
                     title_font=dict(size=9), tickfont=dict(size=8))
    fig.update_yaxes(title_text="∑Δ", showgrid=False, secondary_y=True,
                     tickfont=dict(color=cum_c, size=8), title_font=dict(size=9))
    _apply_xrange(fig, df, x_start, x_end)
    return fig


# ─────────────────────────────────────────────
#  CHART 4: Change in Volume Delta
# ─────────────────────────────────────────────

def chart_delta_diff(df: pd.DataFrame, x_start: float, x_end: float) -> go.Figure:
    fig = go.Figure()

    coverage     = float(df["vap_coverage"].iloc[-1]) if "vap_coverage" in df.columns else 0.0
    delta_source = df["delta_source"].iloc[-1]         if "delta_source" in df.columns else "UNAVAILABLE"

    if df.empty or coverage == 0 or delta_source == "UNAVAILABLE":
        fig.add_annotation(text="TAS data unavailable", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=C["neutral"],size=11))
        _layout(fig, 240)
        return fig

    dd    = df["delta_diff"].fillna(0)
    dd_sm = dd.rolling(3, min_periods=1).mean()
    colors = [C["up"] if v >= 0 else C["down"] for v in dd_sm]

    fig.add_trace(go.Bar(x=df["time"], y=dd,
                         marker_color=colors, opacity=0.45,
                         hovertemplate="%{y:+,.0f}<extra>Δdelta</extra>"))
    fig.add_trace(go.Scatter(x=df["time"], y=dd_sm,
                             line=dict(color=C["roc"], width=2),
                             fill="tozeroy",
                             fillcolor="rgba(255,183,77,0.06)",
                             hoverinfo="skip"))
    fig.add_hline(y=0, line_color=C["grid"], line_width=1)

    _layout(fig, 240)
    fig.update_yaxes(title_text="ΔΔ", gridcolor=C["grid"],
                     title_font=dict(size=9), tickfont=dict(size=8))
    _apply_xrange(fig, df, x_start, x_end)
    return fig


# ─────────────────────────────────────────────
#  TOAST RENDERER
# ─────────────────────────────────────────────

def render_toast(sig: SignalResult):
    if sig.signal in ("NO_SIGNAL","DATA_UNAVAILABLE"): return
    meta = SIGNAL_META.get(sig.signal, SIGNAL_META["NO_SIGNAL"])
    d    = sig.divergence or "—"
    st.markdown(f"""
    <div class="toast-wrap">
      <div class="toast {sig.signal}">
        <div class="t-head">
          <span class="t-sym" style="color:{meta['color']}">{meta['sym']}</span>
          <span class="t-label" style="color:{meta['color']}">{meta['label']}</span>
          <span class="t-conf">conf {sig.confidence*100:.0f}%</span>
        </div>
        <div class="t-body">{sig.reason}</div>
        <div class="t-pills">
          <span class="t-pill">vol {sig.volume_state}</span>
          <span class="t-pill">δ {sig.delta:+.0f}</span>
          <span class="t-pill">roc {sig.roc_delta:+.3f}</span>
          <span class="t-pill">div {d}</span>
          <span class="t-pill">cov {sig.vap_coverage*100:.0f}%</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  METRIC BOX HELPER
# ─────────────────────────────────────────────

def metric_box(label: str, value: str, color: str = "#e0eaf5") -> str:
    return f"""<div class="metric-box">
      <div class="m-label">{label}</div>
      <div class="m-value" style="color:{color}">{value}</div>
    </div>"""


# ═════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙ Controls")
    st.markdown("---")

    instrument = st.text_input(
        "Instrument",
        value="COQ26",
        placeholder="e.g. COQ26, CLN26, BRN…",
        help="Any outright or structure. Data from 00:00 UTC today.",
    ).strip().upper()

    st.markdown("---")
    token = st.text_input(
        "Bearer Token",
        value=DEFAULT_BEARER,
        type="password",
        placeholder="Paste JWT token",
    )

    st.markdown("---")
    st.markdown("**Display**")
    show_markers = st.checkbox("Signal markers on candle chart", value=True)
    show_history = st.checkbox("Signal history table", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.62rem; color:#37474f; line-height:1.9; font-family:"JetBrains Mono",monospace;'>
    Candles  → 5-min (resampled)<br>
    Signals  → 1-min resolution<br>
    Volume   → OHLC source<br>
    Δ Delta  → TAS source only<br>
    Refresh  → 60 s auto
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    load_btn = st.button("▶  Load / Full Refresh", use_container_width=True)


# ─────────────────────────────────────────────
#  AUTO-REFRESH
# ─────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="autorefresh")
except ImportError:
    st.warning("pip install streamlit-autorefresh for live updates")


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for k, v in [
    ("df",          pd.DataFrame()),
    ("last_ts",     None),
    ("sig_history", []),
    ("cur_signal",  None),
    ("bias",        ("NEUTRAL", "No data yet.")),
    ("instrument",  ""),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════
now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
st.markdown(f"""
<div class="monitor-header">
  <span class="monitor-title">🛢 <span>Oil Structure Analytics</span></span>
  <span class="monitor-sub">{instrument} &nbsp;·&nbsp; from 00:00 UTC &nbsp;·&nbsp; {now_str} &nbsp;·&nbsp; 60 s refresh</span>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  LOAD / REFRESH
# ═════════════════════════════════════════════
instrument_changed = (st.session_state.instrument != instrument)

if load_btn or st.session_state.df.empty or instrument_changed:
    st.session_state.instrument = instrument
    with st.spinner(f"Loading {instrument}…"):
        df_new, err = load_full_day(instrument, token)
    if err or df_new.empty:
        st.error(f"Failed: {err or 'No data returned. Check instrument and market hours.'}")
        st.stop()
    all_sigs = compute_all_signals(df_new)
    st.session_state.df          = df_new
    st.session_state.last_ts     = df_new["time"].max()
    st.session_state.sig_history = all_sigs
    st.session_state.cur_signal  = run_signal_engine(df_new)
    st.session_state.bias        = compute_market_structure(df_new)
    n_sigs = len([s for s in all_sigs if s["signal"] not in ("NO_SIGNAL","DATA_UNAVAILABLE")])
    st.success(f"✓  {len(df_new)} bars loaded · {n_sigs} signals", icon=None)

elif st.session_state.last_ts is not None and not st.session_state.df.empty:
    last_ts = st.session_state.last_ts
    now_utc = datetime.now(timezone.utc)
    if (now_utc - last_ts).total_seconds() > 30:
        new_df, inc_err = load_incremental(instrument, last_ts, token)
        if not inc_err and not new_df.empty:
            combined = pd.concat([st.session_state.df, new_df], ignore_index=True)
            combined = (combined.drop_duplicates(subset="time", keep="last")
                                .sort_values("time").reset_index(drop=True))
            st.session_state.df      = combined
            st.session_state.last_ts = combined["time"].max()
            prev_len = len(st.session_state.df) - len(new_df)
            window   = combined.iloc[max(0, prev_len - 5):]
            new_sigs = compute_all_signals(window)
            existing = {s["time"] for s in st.session_state.sig_history}
            for s in new_sigs:
                if s["time"] not in existing:
                    st.session_state.sig_history.append(s)
                    existing.add(s["time"])
            if len(combined) >= 3:
                st.session_state.cur_signal = run_signal_engine(combined.tail(20))
            st.session_state.bias = compute_market_structure(combined)

if st.session_state.df.empty:
    st.info("👈  Enter an instrument code and click **Load / Full Refresh**.")
    st.stop()


# ═════════════════════════════════════════════
#  COMPUTE WINDOW BOUNDS
# ═════════════════════════════════════════════
df = st.session_state.df

times_unix = df["time"].astype("int64") // 10**9
t_min = int(times_unix.min())
t_max = int(times_unix.max())
total_span = t_max - t_min

DEFAULT_WINDOW_S = 3600
window_s = min(DEFAULT_WINDOW_S, total_span)

if "window_start" not in st.session_state:
    st.session_state.window_start = max(t_min, t_max - window_s)


# ═════════════════════════════════════════════
#  RENDER
# ═════════════════════════════════════════════

sig = st.session_state.cur_signal or run_signal_engine(df)
bias_label, bias_desc = st.session_state.bias

# ── Metrics ──────────────────────────────────────────────────────────────────
coverage     = float(df["vap_coverage"].iloc[-1]) if "vap_coverage" in df.columns else 0.0
delta_source = df["delta_source"].iloc[-1]         if "delta_source" in df.columns else "UNKNOWN"
matched_bars = int(df["delta"].notna().sum())       if "delta" in df.columns else 0
total_bars   = len(df)
last_close   = float(df["close"].iloc[-1])          if "close" in df.columns else 0.0
last_delta   = float(df["delta"].iloc[-1])          if ("delta" in df.columns and df["delta"].notna().any()) else 0.0
last_time_s  = df["time"].iloc[-1].strftime("%H:%M") if "time" in df.columns else "--"

delta_color = C["up"] if last_delta >= 0 else C["down"]

st.markdown(f"""
<div class="metric-row">
  {metric_box("Price",         f"{last_close:.4f}")}
  {metric_box("Last Δ",        f"{last_delta:+.0f}", delta_color)}
  {metric_box("TAS Cov",       f"{coverage*100:.1f}%")}
  {metric_box("Bars",          f"{matched_bars}/{total_bars}")}
  {metric_box("Updated",       last_time_s + " UTC")}
  {metric_box("Δ Source",      delta_source)}
</div>
""", unsafe_allow_html=True)

# ── Warnings ──────────────────────────────────────────────────────────────────
if coverage == 0.0:
    st.warning("⚠ No TAS data — volume delta and signals disabled.")
elif coverage < 0.6:
    st.warning(f"⚠ Low TAS coverage ({coverage*100:.1f}%) — signals may be unreliable.")

# ── Market Structure ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Market Structure</div>', unsafe_allow_html=True)
st.markdown(
    f'<span class="struct-badge struct-{bias_label}">{bias_label}</span>'
    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.63rem;color:#37474f;margin-left:12px;">{bias_desc}</span>',
    unsafe_allow_html=True,
)

# ── Signal legend strip ───────────────────────────────────────────────────────
st.markdown("""
<div class="sig-legend">
  <span class="sig-legend-item"><span class="sig-legend-sym" style="color:#2ecc71">▲</span> Long entry</span>
  <span class="sig-legend-item"><span class="sig-legend-sym" style="color:#e74c3c">▼</span> Short entry</span>
  <span class="sig-legend-item"><span class="sig-legend-sym" style="color:#bb86fc">✱</span> Trap</span>
  <span class="sig-legend-item"><span class="sig-legend-sym" style="color:#ffb86c">○</span> Reversal / exit</span>
</div>
""", unsafe_allow_html=True)

# ── Toast ─────────────────────────────────────────────────────────────────────
render_toast(sig)

# ═════════════════════════════════════════════
#  CHART 1 — Candlestick (5-min) + rangeslider
# ═════════════════════════════════════════════
st.markdown('<div class="sec-label">Candlestick — 5-min  ·  signal markers from 1-min data</div>',
            unsafe_allow_html=True)

x_start = st.session_state.window_start
x_end   = t_max

signals_plot = st.session_state.sig_history if show_markers else []

fig_candles = chart_candles(df, signals_plot, instrument, x_start, x_end)

st.plotly_chart(fig_candles, use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": True})

st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#263238;margin:-6px 0 4px;text-align:right;">
  drag rangeslider to scroll · scroll to zoom · all lower charts mirror the same window via slider below
</div>
""", unsafe_allow_html=True)

# ── Window slider ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Time window — drag to scroll all charts</div>',
            unsafe_allow_html=True)

col_l, col_r = st.columns([5,1])
with col_l:
    new_window_start = st.slider(
        "window_start_slider",
        min_value=t_min,
        max_value=max(t_min, t_max - 300),
        value=max(t_min, t_max - DEFAULT_WINDOW_S),
        step=60,
        format="%d",
        label_visibility="collapsed",
        key="ws_slider",
    )
with col_r:
    hrs_shown = (t_max - new_window_start) / 3600
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;'
        f'color:#42a5f5;padding-top:0.5rem;text-align:right;">'
        f'{hrs_shown:.1f} h</div>',
        unsafe_allow_html=True
    )

st.session_state.window_start = new_window_start
x_start = new_window_start


# ─────────────────────────────────────────────
#  CHART 2 — Volume (1-min)
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">Volume — 1-min  ·  OHLC source</div>', unsafe_allow_html=True)
st.plotly_chart(chart_volume(df, x_start, x_end),
                use_container_width=True,
                config={"displayModeBar": False})

# ─────────────────────────────────────────────
#  CHART 3 — Volume Delta (1-min)
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">Volume Delta — 1-min  ·  TAS source</div>', unsafe_allow_html=True)
st.plotly_chart(chart_volume_delta(df, x_start, x_end),
                use_container_width=True,
                config={"displayModeBar": False})

# ─────────────────────────────────────────────
#  CHART 4 — Δ Delta (1-min)
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">Change in Delta (Δdelta) — 1-min  ·  TAS source</div>',
            unsafe_allow_html=True)
st.plotly_chart(chart_delta_diff(df, x_start, x_end),
                use_container_width=True,
                config={"displayModeBar": False})


# ═════════════════════════════════════════════
#  SIGNAL HISTORY TABLE
# ═════════════════════════════════════════════
if show_history and st.session_state.sig_history:
    st.markdown('<div class="sec-label">Signal History — 1-min resolution</div>',
                unsafe_allow_html=True)
    rows = []
    for s in reversed(st.session_state.sig_history):
        meta   = SIGNAL_META.get(s["signal"], SIGNAL_META["NO_SIGNAL"])
        ts     = s["time"]
        ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
        rows.append({
            "Time":   ts_str,
            "Sym":    meta["sym"],
            "Signal": meta["label"],
            "Conf":   f"{s['confidence']*100:.0f}%",
            "Vol":    s["volume_state"],
            "Δ":      f"{s['delta']:+.0f}",
            "ROC":    f"{s['roc_delta']:+.3f}",
            "Div":    s["divergence"] or "—",
            "Reason": s["reason"],
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Time":   st.column_config.TextColumn(width="small"),
            "Sym":    st.column_config.TextColumn(width="small"),
            "Signal": st.column_config.TextColumn(width="small"),
            "Conf":   st.column_config.TextColumn(width="small"),
            "Vol":    st.column_config.TextColumn(width="small"),
            "Δ":      st.column_config.TextColumn(width="small"),
            "ROC":    st.column_config.TextColumn(width="small"),
            "Div":    st.column_config.TextColumn(width="small"),
            "Reason": st.column_config.TextColumn(width="large"),
        },
    )
