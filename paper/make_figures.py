"""Paper figures (vector PDF, true size). Spec: .claude/skills/pesgm-dc-forecasting-paper/references/figures-tables.md

    python paper/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from aidc_gridbench.data.windows import TASKS
from aidc_gridbench.scenarios.generate import sample

OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
SINGLE, DOUBLE = 3.5, 7.16

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.linewidth": 0.3, "grid.color": "#d9d8d4",
    "lines.linewidth": 1.0, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def panel(ax, s):
    ax.text(0.01, 0.97, s, transform=ax.transAxes, va="top", ha="left", fontsize=8, color=INK)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1 framework
EX_ORIGIN = "2021-08-02 11:00:00+00:00"  # same origin as Fig. 4


def _example_data():
    """Real quantities at the Fig. 4 example origin (1-h/24-h task, test period)."""
    tname = "mit-1h-24h"; task = TASKS[tname]; taus = np.asarray(task.taus); H = task.horizon
    tr = np.load(f"results/calibration/{tname}_truth.npz")
    i = int(np.flatnonzero(pd.to_datetime(tr["origins"], utc=True) == pd.Timestamp(EX_ORIGIN))[0])
    st = np.load(f"results/calibration_tcn/seed0/{tname}_qtcn_stream.npz")
    nv, t = int(st["n_val"]), st["t"]
    q_tcn = st["q"][nv:].astype(float)
    resid = st["y"].astype(float) - st["q"][..., taus.searchsorted(0.5)].astype(float)
    avail = np.searchsorted(t + H, t, side="right")[nv:]
    scen = sample(q_tcn[i:i + 1], taus, 200, "schaake", resid_hist=resid, resid_avail=avail[i:i + 1], seed=1)[0]
    q_raw = np.load(f"results/calibration/{tname}_qlgbm_rel_raw.npz")["q"][i].astype(float)
    q_cal = np.load(f"results/calibration/{tname}_qlgbm_rel_online_regime.npz")["q"][i].astype(float)
    # recent history before the origin for the data panel
    s = pd.read_parquet("data/processed/mit_cluster_1min.parquet").resample("1h").mean()
    o = pd.Timestamp(EX_ORIGIN)
    hist = s[(s.index > o - pd.Timedelta(days=4)) & (s.index <= o)]
    return dict(taus=taus, H=H, y=tr["y"][i].astype(float), q_tcn=q_tcn[i], q_raw=q_raw, q_cal=q_cal,
                scen=scen, hist=hist)


def _mini(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.5); ax.spines[sp].set_color(GRAY)


def fig1():
    d = _example_data(); taus = d["taus"]; hrs = np.arange(1, d["H"] + 1)
    k = {tau: j for j, tau in enumerate(taus)}
    fig = plt.figure(figsize=(DOUBLE, 2.45))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")

    # geometry (figure fractions)
    xd, wd = 0.012, 0.105                       # data column
    xs = [0.165, 0.375, 0.585, 0.795]; ws = 0.185  # stage panels
    yp0, yp1 = 0.30, 0.97                       # panel box
    ya0, ya1 = 0.44, 0.78                       # mini-plot axes
    stage_titles = ["Probabilistic forecasting", "Uncertainty calibration",
                    "Temporal dependence", "Reserve decision"]
    plot_labels = ["TCN quantiles", "LightGBM 90% interval", "Schaake scenarios", "horizon-wide deviation"]
    methods = ["persistence, LightGBM, TCN\n(quantile outputs)", "online conformal correction,\nregime-conditioned",
               "independent, copula,\nbootstrap, Schaake shuffle", "newsvendor rule,\nper-step and horizon-wide"]
    metrics = ["CRPS, MAE", "coverage per regime,\nWinkler score", "energy / variogram score,\nKS of peak and ramp",
               "realised cost,\nshortfall probability"]
    passed = [r"$\hat q^{\,\tau}_{t,h}$", r"$\tilde q^{\,\tau}_{t,h}$", r"$P^{(s)}_{t+h}$"]

    def box(x, w, y0, y1, fc, ec, ls="-"):
        bg.add_patch(FancyBboxPatch((x, y0), w, y1 - y0, boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=fc, ec=ec, lw=0.6, ls=ls, transform=bg.transAxes))

    def arrow(x0, x1, y, label=None):
        bg.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                     color=INK, transform=bg.transAxes))
        if label:
            bg.text((x0 + x1) / 2, y + 0.035, label, ha="center", va="bottom", fontsize=7.5, color=INK)

    # data column
    box(xd, wd, yp0, yp1, "white", GRAY)
    bg.text(xd + wd / 2, 0.925, "Measured data", ha="center", va="center", fontsize=7.5, weight="bold", color=INK)
    a = fig.add_axes([xd + 0.012, 0.62, wd - 0.024, 0.2]); _mini(a)
    a.plot(d["hist"].index, d["hist"].power_w / 1e3, color=INK, lw=0.7)
    a.text(0.0, 1.02, "GPU power", transform=a.transAxes, fontsize=6.5, color=MUTED, va="bottom")
    a = fig.add_axes([xd + 0.012, 0.38, wd - 0.024, 0.16]); _mini(a)
    a.step(d["hist"].index, d["hist"].gpus_alloc, color=BLUE, lw=0.7, where="post")
    a.text(0.0, 1.02, "scheduler state", transform=a.transAxes, fontsize=6.5, color=MUTED, va="bottom")
    bg.text(xd + wd / 2, 0.325, "MIT Supercloud", ha="center", va="center", fontsize=6.5, color=MUTED, style="italic")
    arrow(xd + wd + 0.006, xs[0] - 0.006, 0.61)

    axs = []
    for j, x in enumerate(xs):
        box(x, ws, yp0, yp1, "#f6f5f2", GRAY)
        bg.text(x + 0.017, 0.925, str(j + 1), ha="center", va="center", fontsize=6.5, color="white", weight="bold",
                bbox=dict(boxstyle="circle,pad=0.18", fc=INK, ec="none"))
        bg.text(x + 0.034, 0.925, stage_titles[j], ha="left", va="center", fontsize=7.2, weight="bold", color=INK)
        bg.text(x + ws / 2, 0.36, methods[j], ha="center", va="center", fontsize=6.5, color=INK, linespacing=1.15)
        axs.append(fig.add_axes([x + 0.014, ya0, ws - 0.028, ya1 - ya0])); _mini(axs[-1])
        axs[-1].text(0.0, 1.03, plot_labels[j], transform=axs[-1].transAxes, fontsize=6.5, color=MUTED, va="bottom")
        if j < 3:
            arrow(x + ws + 0.004, xs[j + 1] - 0.004, 0.61, passed[j])
    arrow(xs[3] + ws + 0.004, 0.993, 0.61)
    bg.text(0.996, 0.66, r"$R_t$", ha="right", va="bottom", fontsize=7.5, color=INK)

    # 1: quantile fan (TCN)
    a = axs[0]; q = d["q_tcn"]
    a.fill_between(hrs, q[:, k[0.05]], q[:, k[0.95]], color=ORANGE, alpha=0.18, lw=0)
    a.fill_between(hrs, q[:, k[0.25]], q[:, k[0.75]], color=ORANGE, alpha=0.38, lw=0)
    a.plot(hrs, q[:, k[0.5]], color=ORANGE, lw=0.9)
    # 2: interval before / after calibration (LightGBM) with observations
    a = axs[1]
    a.fill_between(hrs, d["q_cal"][:, k[0.05]], d["q_cal"][:, k[0.95]], color=BLUE, alpha=0.2, lw=0)
    for qq, st_ in ((d["q_raw"], dict(color=GRAY, lw=0.7, ls=(0, (3, 2)))), (d["q_cal"], dict(color=BLUE, lw=0.8))):
        a.plot(hrs, qq[:, k[0.05]], **st_); a.plot(hrs, qq[:, k[0.95]], **st_)
    a.plot(hrs, d["y"], ls="none", marker="o", ms=1.8, color=INK)
    lo_, hi_ = a.get_ylim(); a.set_ylim(lo_ - 0.06 * (hi_ - lo_), hi_ + 0.04 * (hi_ - lo_))
    a.text(0.03, 0.06, "-- raw", transform=a.transAxes, fontsize=6, color=GRAY)
    a.text(0.03, 0.17, "\u2014 calibrated", transform=a.transAxes, fontsize=6, color=BLUE)
    # 3: Schaake scenarios
    a = axs[2]
    for s in range(15):
        a.plot(hrs, d["scen"][s], color=BLUE, lw=0.45, alpha=0.55)
    a.plot(hrs, d["y"], color=INK, lw=0.9)
    # 4: distribution of horizon-wide deviation and reserve threshold
    a = axs[3]
    med = d["q_tcn"][:, k[0.5]]
    D = (d["scen"] - med[None]).max(axis=1)
    R = np.quantile(D, 0.9)
    bins = np.linspace(D.min(), D.max(), 22)
    n, b, patches = a.hist(D, bins=bins, color=GRAY, alpha=0.55, lw=0)
    for p, left in zip(patches, b[:-1]):
        if left >= R:
            p.set_facecolor(ORANGE); p.set_alpha(0.8)
    a.axvline(R, color=INK, lw=0.9)
    a.text(R, a.get_ylim()[1] * 0.97, r"$R_t$ ", ha="right", va="top", fontsize=7, color=INK)
    a.text(0.98, 0.84, "shortfall\nregion", transform=a.transAxes, ha="right", va="top", fontsize=6, color=ORANGE)

    # stage-wise evaluation band
    box(xs[0] - 0.005, xs[3] + ws - xs[0] + 0.01, 0.02, 0.235, "#ecebe7", GRAY, ls=(0, (3, 2)))
    bg.text(xd + wd / 2, 0.128, "Stage-wise\nevaluation", ha="center", va="center", fontsize=7, weight="bold",
            color=INK, linespacing=1.1)
    for j, x in enumerate(xs):
        bg.text(x + ws / 2, 0.128, metrics[j], ha="center", va="center", fontsize=6.5, color=INK,
                style="italic", linespacing=1.15)
        bg.plot([x + ws / 2, x + ws / 2], [yp0, 0.235], color=GRAY, lw=0.5, ls=(0, (1, 1.5)), transform=bg.transAxes)
    save(fig, "fig1_framework")


# ---------------------------------------------------------------- Fig. 2 data overview
def fig2():
    s = pd.read_parquet("data/processed/mit_cluster_1min.parquet")
    h = s.resample("1h").mean()
    h = h[h.index >= "2021-03-01"]
    fig, ax = plt.subplots(2, 1, figsize=(SINGLE, 2.3), sharex=True, gridspec_kw={"hspace": 0.12})
    task = TASKS["mit-1h-24h"]
    for a in ax:
        for (lo, hi), c in ((task.train, "#ffffff"), (task.val, "#efeeea"), (task.test, "#e4e3de")):
            a.axvspan(pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC") + pd.Timedelta(days=1), color=c, lw=0, zorder=0)
    ax[0].plot(h.index, h.power_w / 1e3, color=INK, lw=0.6)
    ax[0].set_ylabel("GPU power (kW)")
    ax[1].plot(h.index, h.gpus_alloc, color=BLUE, lw=0.6)
    ax[1].set_ylabel("Allocated GPUs")
    ax[0].set_ylim(10, 58); ax[1].set_ylim(0, 470)
    for lab, (lo, hi) in (("train", task.train), ("val", task.val), ("test", task.test)):
        mid = pd.Timestamp(lo, tz="UTC") + (pd.Timestamp(hi, tz="UTC") - pd.Timestamp(lo, tz="UTC")) / 2
        ax[0].text(mid, 55, lab, ha="center", va="top", fontsize=7, color=MUTED)
    # one maintenance outage and the capacity increase
    out = h.gpus_alloc[(h.index > "2021-06-10") & (h.index < "2021-06-25")].idxmin()
    ax[1].annotate("maintenance", xy=(out, 10), xytext=(out + pd.Timedelta(days=9), 90), fontsize=6.5,
                   color=MUTED, arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    ax[1].annotate("capacity increase", xy=(pd.Timestamp("2021-05-11", tz="UTC"), 300),
                   xytext=(pd.Timestamp("2021-03-08", tz="UTC"), 400), fontsize=6.5, color=MUTED,
                   arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    panel(ax[0], "(a)"); panel(ax[1], "(b)")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.1)
    save(fig, "fig2_data")


# ---------------------------------------------------------------- Fig. 3 coverage by regime
def fig3():
    t = "mit-1h-24h"
    lg = json.load(open(f"results/calibration/{t}.json"))["qlgbm_rel"]
    tc = [json.load(open(p))["qtcn"]["raw"]["cov90_by_regime"]
          for p in sorted(Path("results/calibration_tcn").glob(f"seed*/{t}.json"))]
    regimes = ["0", "1", "2", "3"]; names = ["low", "calm", "active", "volatile"]
    series = [
        ("LightGBM", [lg["raw"]["cov90_by_regime"][r] for r in regimes], dict(marker="o", mfc="white", color=GRAY)),
        ("LightGBM + online calibration", [lg["online_regime"]["cov90_by_regime"][r] for r in regimes],
         dict(marker="o", color=BLUE)),
        ("TCN", [np.mean([d[r] for d in tc]) for r in regimes], dict(marker="s", color=ORANGE)),
    ]
    fig, ax = plt.subplots(figsize=(SINGLE, 1.9))
    x = np.arange(4)
    for k, (lab, v, st) in enumerate(series):
        ax.plot(x + (k - 1) * 0.16, 100 * np.array(v), ls="none", ms=5, mew=0.9, label=lab, **st)
    ax.axhline(90, ls=(0, (4, 3)), lw=0.7, color=INK)
    ax.text(3.45, 90.8, "nominal 90%", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(x, names); ax.set_xlim(-0.5, 3.5); ax.set_ylim(40, 100)
    ax.set_ylabel("Coverage (%)"); ax.set_xlabel("Workload regime at forecast origin")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.2, columnspacing=0.8)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.2)
    save(fig, "fig3_regime_coverage")


# ---------------------------------------------------------------- Fig. 4 scenario structure
def fig4():
    tname = "mit-1h-24h"; task = TASKS[tname]; taus = np.asarray(task.taus); H = task.horizon
    d = Path("results/calibration_tcn/seed0")
    st = np.load(d / f"{tname}_qtcn_stream.npz")
    nv = int(st["n_val"]); t = st["t"]
    q_te = st["q"][nv:].astype(float); y_te = st["y"][nv:].astype(float); reg_te = st["reg"][nv:]
    resid = st["y"].astype(float) - st["q"][..., taus.searchsorted(0.5)].astype(float)
    avail = np.searchsorted(t + H, t, side="right")[nv:]
    # same evaluation subset as run_scenarios.py
    sel = np.sort(np.random.default_rng(0).choice(len(y_te), min(3000, len(y_te)), replace=False))
    ind = sample(q_te[sel], taus, 200, "independent", seed=1)
    sch = sample(q_te[sel], taus, 200, "schaake", resid_hist=resid, resid_avail=avail[sel], seed=1)
    ramp = lambda a: np.diff(a, axis=-1).max(axis=-1)
    # example origin: largest observed ramp among origins in a non-low regime staying above the idle floor
    obs_r = ramp(y_te[sel])
    ok = (reg_te[sel] != 0) & (y_te[sel].min(axis=1) > 20)
    i = np.flatnonzero(ok)[np.argmax(obs_r[ok])]
    hrs = np.arange(1, H + 1)
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 1.9), gridspec_kw={"wspace": 0.28})
    for a, sc, lab in ((ax[0], ind, "(a) independent"), (ax[1], sch, "(b) Schaake shuffle")):
        for s in range(20):
            a.plot(hrs, sc[i, s], color=GRAY, lw=0.4, alpha=0.6)
        a.plot(hrs, q_te[sel][i, :, taus.searchsorted(0.5)], color=BLUE, lw=1.0, ls=(0, (4, 2)), label="median forecast")
        a.plot(hrs, y_te[sel][i], color=INK, lw=1.2, label="observed")
        a.set_xlabel("Hours ahead"); a.set_ylabel("GPU power (kW)"); panel(a, lab)
    lo = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0]); hi = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])
    for a in ax[:2]:
        a.set_ylim(lo, hi + 0.12 * (hi - lo))
    h_, l_ = ax[0].get_legend_handles_labels()
    h_.append(matplotlib.lines.Line2D([], [], color=GRAY, lw=0.6)); l_.append("20 scenarios")
    fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.35, 1.0), ncol=3, handlelength=1.8, columnspacing=1.2)
    for vals, lab, st in ((obs_r, "observed", dict(color=INK, lw=1.2)),
                          (ramp(ind).ravel(), "independent", dict(color=GRAY, lw=1.0, ls=(0, (4, 2)))),
                          (ramp(sch).ravel(), "Schaake shuffle", dict(color=BLUE, lw=1.0))):
        v = np.sort(vals); ax[2].plot(v, np.arange(1, len(v) + 1) / len(v), label=lab, **st)
    ax[2].set_xlabel("Max 1-h ramp-up in horizon (kW)"); ax[2].set_ylabel("Empirical CDF")
    ax[2].set_xlim(0, 20); ax[2].set_ylim(0, 1.02)
    ax[2].legend(loc="lower right", handlelength=1.6); panel(ax[2], "(c)")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.86, bottom=0.21)
    save(fig, "fig4_scenarios")
    return {"example_origin": str(pd.to_datetime(np.load(f"results/calibration/{tname}_truth.npz")["origins"][sel][i], utc=True))}


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    if not only or "1" in only: fig1()
    if not only or "2" in only: fig2()
    if not only or "3" in only: fig3()
    info = fig4() if (not only or "4" in only) else {}
    print("figures written to", OUT, info)
