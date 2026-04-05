import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cloud Storage Growth Model",
    page_icon="☁️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.main { background: #0d1117; }

.stApp { background: #0d1117; color: #e6edf3; }

h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; color: #e6edf3 !important; }

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-value.alert { color: #f85149; }
.metric-value.ok    { color: #3fb950; }

.section-header {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    border-bottom: 1px solid #30363d;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

div[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}

div[data-testid="stSidebar"] label {
    color: #c9d1d9 !important;
    font-size: 13px !important;
}

.stSlider > div > div { background: #30363d; }

.stCheckbox label { color: #c9d1d9 !important; }

.title-block {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
}
.title-main {
    font-size: 26px;
    font-weight: 700;
    color: #e6edf3;
    font-family: 'IBM Plex Sans', sans-serif;
    margin: 0 0 4px 0;
}
.title-sub {
    font-size: 13px;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    margin: 0;
}

.alert-box {
    background: #2d1b1b;
    border: 1px solid #f85149;
    border-radius: 8px;
    padding: 14px 18px;
    color: #f85149;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    margin-top: 8px;
}
.ok-box {
    background: #1b2d1b;
    border: 1px solid #3fb950;
    border-radius: 8px;
    padding: 14px 18px;
    color: #3fb950;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Core model functions ──────────────────────────────────────────────────────
def logistic_growth(S0: float, r: float, K: float, days: int) -> list:
    """
    Simulates cloud storage growth using the logistic model.
    Governing equation: dS/dt = r * S(t) * (1 - S(t) / K)
    Discretised via Euler's method: S(t+1) = S(t) + r * S(t) * (1 - S(t) / K)

    Parameters
    ----------
    S0   : Initial storage used (GB)
    r    : Daily growth rate (fraction per day)
    K    : Maximum storage capacity (GB)
    days : Number of days to simulate
    """
    storage = [S0]
    for _ in range(days):
        S = storage[-1]
        dS = r * S * (1.0 - S / K)
        storage.append(S + dS)
    return storage


def find_threshold_day(storage: list, K: float, pct: float = 0.80) -> int | None:
    """Return the first day storage crosses pct * K, or None."""
    target = pct * K
    for day, s in enumerate(storage):
        if s >= target:
            return day
    return None


# ── Sidebar — user inputs ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ☁️ Cloud Storage Model")
    st.markdown("---")

    st.markdown('<p class="section-header">Parameters</p>', unsafe_allow_html=True)

    S0 = st.number_input(
        "Initial Storage S₀ (GB)",
        min_value=1.0, max_value=10000.0, value=50.0, step=10.0,
        help="How much storage is already in use on Day 0."
    )
    r = st.number_input(
        "Daily Growth Rate r",
        min_value=0.0001, max_value=0.05, value=0.002, step=0.0005, format="%.4f",
        help="Fraction of remaining capacity added each day. 0.002 = 0.2%/day."
    )
    K = st.number_input(
        "Max Capacity K (GB)",
        min_value=100.0, max_value=100000.0, value=5000.0, step=100.0,
        help="Hard upper limit of provisioned cloud storage."
    )
    days = st.slider(
        "Simulation Duration (days)",
        min_value=30, max_value=1825, value=730, step=30,
        help="1825 days = 5 years"
    )

    st.markdown("---")
    st.markdown('<p class="section-header">Compare Scenarios</p>', unsafe_allow_html=True)

    show_high   = st.checkbox("High Upload  (r = 0.005)", value=True)
    show_low    = st.checkbox("Low Capacity (K = 2000 GB)", value=False)
    show_agg    = st.checkbox("Aggressive   (r = 0.008, K = 10 000 GB)", value=False)
    show_cons   = st.checkbox("Conservative (r = 0.0005)", value=False)

    st.markdown("---")
    st.markdown('<p class="section-header">Threshold</p>', unsafe_allow_html=True)
    alert_pct = st.slider("Alert Threshold (%)", 50, 95, 80, 5,
                          help="Day the model flags for expansion planning.")


# ── Run main simulation ───────────────────────────────────────────────────────
storage      = logistic_growth(S0, r, K, days)
alert_day    = find_threshold_day(storage, K, alert_pct / 100)
critical_day = find_threshold_day(storage, K, 0.95)
day365_val   = storage[min(365, days)]
final_val    = storage[-1]
usage_pct    = final_val / K * 100


# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="title-block">
  <p class="title-main">☁️ Corporate Cloud Storage Growth Model</p>
  <p class="title-sub">Logistic Growth Simulation &nbsp;|&nbsp;
     S(t+1) = S(t) + r·S(t)·(1 − S(t)/K) &nbsp;|&nbsp;
     Euler Discretisation</p>
</div>
""", unsafe_allow_html=True)


# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Storage at Day 365</div>
      <div class="metric-value">{day365_val:,.0f} <span style="font-size:14px">GB</span></div>
    </div>""", unsafe_allow_html=True)

with c2:
    alert_cls = "alert" if alert_day else "ok"
    alert_txt = f"Day {alert_day}" if alert_day else "Not reached"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{alert_pct}% Alert Day</div>
      <div class="metric-value {alert_cls}">{alert_txt}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    crit_cls = "alert" if critical_day else "ok"
    crit_txt = f"Day {critical_day}" if critical_day else "Not reached"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">95% Critical Day</div>
      <div class="metric-value {crit_cls}">{crit_txt}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    fin_cls = "alert" if usage_pct > 80 else "ok"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Final Usage (Day {days})</div>
      <div class="metric-value {fin_cls}">{final_val:,.0f} <span style="font-size:14px">GB</span></div>
    </div>""", unsafe_allow_html=True)


# ── Alert banner ──────────────────────────────────────────────────────────────
if alert_day:
    st.markdown(f"""
    <div class="alert-box">
      ⚠️  Expansion alert: storage reaches {alert_pct}% of capacity (
      {alert_pct/100*K:,.0f} GB) on <strong>Day {alert_day}</strong>.
      Plan procurement at least 3–6 months in advance.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="ok-box">
      ✅  Storage stays below {alert_pct}% capacity for the entire {days}-day window.
      No expansion required within this period.
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Main chart ────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
CARD_BG  = "#161b22"
BORDER   = "#30363d"
BLUE     = "#58a6ff"
RED      = "#f85149"
ORANGE   = "#e3b341"
GREEN    = "#3fb950"
PURPLE   = "#bc8cff"
GREY     = "#8b949e"

t = np.arange(days + 1)

fig, ax = plt.subplots(figsize=(13, 5.2))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)

# Baseline
ax.plot(t, storage, color=BLUE, linewidth=2.5,
        label=f"Baseline  (r={r}, K={K:,.0f} GB)", zorder=5)

# Optional scenarios
if show_high:
    s_hi = logistic_growth(S0, 0.005, K, days)
    ax.plot(t, s_hi, color=RED, linewidth=1.8, linestyle="-",
            label="High Upload  (r=0.005)", zorder=4)

if show_low:
    s_lo = logistic_growth(S0, r, 2000, days)
    ax.plot(t, s_lo, color=ORANGE, linewidth=1.8, linestyle="-",
            label="Low Capacity  (K=2000 GB)", zorder=4)

if show_agg:
    s_ag = logistic_growth(S0, 0.008, 10000, days)
    ax.plot(t, s_ag, color=GREEN, linewidth=1.8, linestyle="-",
            label="Aggressive  (r=0.008, K=10 000 GB)", zorder=4)

if show_cons:
    s_co = logistic_growth(S0, 0.0005, K, days)
    ax.plot(t, s_co, color=PURPLE, linewidth=1.8, linestyle="-",
            label="Conservative  (r=0.0005)", zorder=4)

# Threshold lines
ax.axhline(alert_pct / 100 * K, color=ORANGE, linewidth=1.2, linestyle="--",
           label=f"{alert_pct}% Alert ({alert_pct/100*K:,.0f} GB)", zorder=3)
ax.axhline(K, color=RED, linewidth=1.2, linestyle="--",
           label=f"Max Capacity ({K:,.0f} GB)", zorder=3)

# Alert day marker
if alert_day and alert_day <= days:
    ax.axvline(alert_day, color=ORANGE, linewidth=1, linestyle=":",
               alpha=0.7, zorder=3)
    ax.annotate(f"Day {alert_day}",
                xy=(alert_day, alert_pct / 100 * K),
                xytext=(alert_day + days * 0.02, alert_pct / 100 * K * 0.92),
                color=ORANGE, fontsize=9,
                fontfamily="monospace",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))

# Styling
for spine in ax.spines.values():
    spine.set_color(BORDER)
ax.tick_params(colors=GREY, labelsize=9)
ax.xaxis.label.set_color(GREY)
ax.yaxis.label.set_color(GREY)
ax.set_xlabel("Days", fontsize=10)
ax.set_ylabel("Storage Used (GB)", fontsize=10)
ax.set_title("Storage Growth Over Time  —  Logistic Model",
             fontsize=13, color="#e6edf3", pad=14, fontweight="600")
ax.grid(True, color=BORDER, linewidth=0.6, linestyle="--", alpha=0.6)
ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor="#c9d1d9",
          fontsize=9, loc="upper left")
ax.set_xlim(0, days)

plt.tight_layout()
st.pyplot(fig)
plt.close()


# ── Simulation data table ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">Simulation Snapshot</p>', unsafe_allow_html=True)

import pandas as pd

checkpoints = [0, 30, 90, 180, 365, 730, 1095, 1460, 1825]
checkpoints = [d for d in checkpoints if d <= days]
if days not in checkpoints:
    checkpoints.append(days)

rows = []
for d in checkpoints:
    s = storage[d]
    rows.append({
        "Day": d,
        "Storage Used (GB)": f"{s:,.1f}",
        "% of Capacity": f"{s/K*100:.1f}%",
        "Remaining (GB)": f"{K-s:,.1f}",
        "Status": "🔴 Critical" if s >= 0.95*K else
                  ("🟡 Warning" if s >= alert_pct/100*K else "🟢 OK"),
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#8b949e; font-size:12px; font-family:'IBM Plex Mono',monospace;">
  Cloud Storage Growth Model &nbsp;·&nbsp;
  Mathematical Modelling for Computer Applications &nbsp;·&nbsp;
  Antara Nilesh Shelar | Roll No. 78 | SY BSC IT &nbsp;·&nbsp;
  Smt. CHM (Autonomous) College 2025–26
</p>
""", unsafe_allow_html=True)
