# figures for the draft. usage: python figures.py results_5 results_all results_pit --out figs/
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"ridge": "#2a78d6", "lasso": "#eb6834", "enet": "#1baf7a", "hm": "#52514e"}
NAME = {"ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic net", "hm": "Hist. mean"}
LABEL = {"results_5": "5 signals", "results_all": "152 signals", "results_pit": "Published signals only"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "font.family": "serif"})

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--out", default="figs/")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dirs = [d.rstrip("/") for d in a.dirs]

# fig 1: cumulative squared-error advantage over the historical mean (rises = model beats hist. mean)
fig, axs = plt.subplots(1, len(dirs), figsize=(6.75, 2.1), sharey=False)
axs = np.atleast_1d(axs)
for ax, d in zip(axs, dirs):
    p = pd.read_csv(d + "/val_predictions.csv")
    p["month"] = pd.PeriodIndex(p.month, freq="M").to_timestamp()
    ms = [m for m in ["ridge", "lasso", "enet"] if m in p]
    for m in ms:
        g = p.assign(v=(p.r - p.hm)**2 - (p.r - p[m])**2).groupby("month").v.sum().cumsum()
        g = g / ((p.r - p.hm)**2).sum() * 100          # in % of total hist.-mean SSE, so the end value = R2_OS
        ax.plot(g.index, g.values, color=COL[m], lw=1.4, label=NAME[m])
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_title(LABEL.get(os.path.basename(d), d), fontsize=8)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_xlim(pd.Timestamp("1989-06"), pd.Timestamp("2010-06"))
axs[0].set_ylabel("Cumulative R$^2_{OS}$ (%)")
axs[0].legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout()
fig.savefig(a.out + "/cum_r2.pdf"); fig.savefig(a.out + "/cum_r2.png", dpi=200)

# fig 2: R2_OS with 95% block-bootstrap intervals
fig, ax = plt.subplots(figsize=(4.2, 2.3))
y, ticks = 0, []
for d in dirs:
    r = json.load(open(d + "/results.json"))
    for m in ["ridge", "lasso", "enet"]:
        if "r2os_" + m not in r:
            continue
        e, lo, hi = [100 * v for v in r["r2os_" + m]]
        ax.plot([lo, hi], [y, y], color=COL[m], lw=1.6, solid_capstyle="round")
        ax.plot(e, y, "o", ms=4.5, color=COL[m], mec="#fcfcfb", mew=1)
        ticks.append((y, f"{LABEL.get(os.path.basename(d), d)}: {NAME[m]}"))
        y -= 1
    y -= 0.6
ax.axvline(0, color="#8a8984", lw=0.8)
ax.set_yticks([t[0] for t in ticks]); ax.set_yticklabels([t[1] for t in ticks], fontsize=6.5)
ax.set_xlabel("Out-of-sample R$^2$ vs. historical mean (%)", fontsize=7)
ax.grid(axis="x", color="#e6e5e0", lw=0.6)
fig.tight_layout()
fig.savefig(a.out + "/r2_ci.pdf"); fig.savefig(a.out + "/r2_ci.png", dpi=200)
print("ok")
