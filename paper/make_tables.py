"""Collect result JSONs into paper tables (Markdown + LaTeX).

    python paper/make_tables.py
Writes paper/tables/forecast.{md,tex} and paper/tables/scenarios.{md,tex}.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
OUT = Path("paper/tables"); OUT.mkdir(parents=True, exist_ok=True)
TASKS = ["mit-5min-1h", "mit-15min-4h", "mit-1h-24h"]
VARIANTS = ["raw", "static", "online", "online_regime"]
KEYS = ["crps_kw", "mae_kw", "cov90", "width90_kw", "winkler90"]


def forecast_rows():
    rows = []
    for t in TASKS:
        p = R / "calibration" / f"{t}.json"
        if p.exists():
            js = json.load(open(p))
            for m, v in js.items():
                if m == "regime_share_test":
                    continue
                for var in VARIANTS:
                    rows.append({"task": t, "model": m, "calib": var, **{k: v[var][k] for k in KEYS},
                                 "cov90_min_regime": min(v[var]["cov90_by_regime"].values())})
        seeds = sorted((R / "calibration_tcn").glob(f"seed*/{t}.json"))
        if seeds:
            for var in VARIANTS:
                vals = [json.load(open(s))["qtcn"][var] for s in seeds]
                row = {"task": t, "model": f"qtcn (n={len(seeds)})", "calib": var}
                for k in KEYS:
                    a = np.array([x[k] for x in vals]); row[k] = a.mean(); row[k + "_sd"] = a.std(ddof=1) if len(a) > 1 else 0
                row["cov90_min_regime"] = np.mean([min(x["cov90_by_regime"].values()) for x in vals])
                rows.append(row)
    return pd.DataFrame(rows)


def scenario_rows():
    rows = []
    for p in sorted((R / "scenarios").glob("*.json")):
        js = json.load(open(p))
        for var in ("raw", "online_regime"):
            if var not in js:
                continue
            for g, r in js[var].items():
                if g == "per_step_reserve":
                    continue
                b = r["block_reserve"]["0.1"]
                rows.append({"task": js["task"], "model": js["model"], "calib": var, "generator": g,
                             "ES": r["energy_score"], "VS": r["variogram_score"], "brier_exceed": r["exceed_brier"],
                             "ks_peak": r["pit_ks"]["peak"], "ks_ramp_up": r["pit_ks"]["ramp_up"],
                             "ks_energy": r["pit_ks"]["energy"],
                             "block_cost": b["cost"], "block_reserve": b["reserve_kw"],
                             "block_short_p": b["shortfall_prob"]})
    return pd.DataFrame(rows)


for name, df in (("forecast", forecast_rows()), ("scenarios", scenario_rows())):
    if df.empty:
        continue
    df = df.round(3)
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(df.to_markdown(index=False))
    print(f"== {name}\n{df.to_string(index=False)}\n")


# ---------------------------------------------------------------- paper tables (booktabs)
TASK_LABEL = {"mit-5min-1h": "5-min / 1-h", "mit-15min-4h": "15-min / 4-h", "mit-1h-24h": "1-h / 24-h"}


def _bold(v, fmt, best):
    """Bold when v equals the best value at the displayed precision (ties all bold)."""
    s = fmt.format(v)
    return rf"\textbf{{{s}}}" if s == fmt.format(best) else s


def table1():
    f = forecast_rows()
    rows_def = [("persistence", "raw", "Persistence"), ("qlgbm_rel", "raw", "LightGBM"),
                ("qlgbm_rel", "online_regime", "LightGBM + OC"), ("qtcn (n=3)", "raw", "TCN"),
                ("qtcn (n=3)", "online_regime", "TCN + OC")]
    L = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Forecaster & CRPS & MAE & Cov. & Min cov. & Winkler \\",
         r" & (kW) & (kW) & (\%) & (\%) & \\", r"\midrule"]
    for k, t in enumerate(TASKS):
        blk = []
        for m, c, lab in rows_def:
            r = f[(f.task == t) & (f.model == m) & (f.calib == c)]
            if len(r):
                blk.append((lab, r.iloc[0]))
        best_crps = min(r.crps_kw for _, r in blk)
        best_cov = min((r.cov90 for _, r in blk), key=lambda c: abs(round(100 * c, 1) - 90))
        if k:
            L.append(r"\addlinespace")
        L.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{TASK_LABEL[t]}}}}} \\")
        for lab, r in blk:
            L.append(" & ".join([lab, _bold(r.crps_kw, "{:.3f}", best_crps), f"{r.mae_kw:.3f}",
                                 _bold(100 * r.cov90, "{:.1f}", 100 * best_cov),
                                 f"{100 * r.cov90_min_regime:.1f}", f"{r.winkler90:.2f}"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table2(model="qtcn"):
    s = scenario_rows()
    s = s[(s.model == model) & (s.calib == "raw")]
    gens = [("independent", "Independent"), ("copula", "Gaussian copula"),
            ("bootstrap", "Residual bootstrap"), ("schaake", "Schaake shuffle")]
    L = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Generator & ES & VS & KS & KS & Reserve & Short- \\",
         r" & & & peak & ramp & cost & fall (\%) \\", r"\midrule"]
    for k, t in enumerate(TASKS):
        b = s[s.task == t].set_index("generator")
        if b.empty:
            continue
        if k:
            L.append(r"\addlinespace")
        L.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{TASK_LABEL[t]}}}}} \\")
        best = {c: b.loc[[g for g, _ in gens], c].min() for c in ("ES", "VS", "ks_peak", "ks_ramp_up", "block_cost")}
        sp = b.loc[[g for g, _ in gens], "block_short_p"]
        best_sp = sp.iloc[(100 * sp).round(1).sub(10).abs().argmin()]
        for g, lab in gens:
            r = b.loc[g]
            L.append(" & ".join([lab, _bold(r.ES, "{:.2f}", best["ES"]),
                                 _bold(r.VS, "{:.1f}", best["VS"]),
                                 _bold(r.ks_peak, "{:.2f}", best["ks_peak"]),
                                 _bold(r.ks_ramp_up, "{:.2f}", best["ks_ramp_up"]),
                                 _bold(r.block_cost, "{:.3f}", best["block_cost"]),
                                 _bold(100 * r.block_short_p, "{:.1f}", 100 * best_sp)]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def inject(tex_path, tag, body):
    """Replace the text between '% BEGIN AUTO <tag>' and '% END AUTO <tag>' in main.tex."""
    import re
    tex = Path(tex_path).read_text()
    pat = re.compile(rf"(% BEGIN AUTO {tag}[^\n]*\n).*?(\s*% END AUTO {tag})", re.S)
    if not pat.search(tex):
        raise SystemExit(f"marker {tag} not found in {tex_path}")
    Path(tex_path).write_text(pat.sub(lambda m: m.group(1) + body + m.group(2), tex))


MAIN = Path("paper/main.tex")
inject(MAIN, "TABLE-I", table1())
inject(MAIN, "TABLE-II", table2())
print("tables written into", MAIN)
