"""Build the static leaderboard site from result records.

    python scripts/build_leaderboard.py --results <results dir> [--selected <dir>] --out site

Reads every final <task>/<model>/seed<k>.json, aggregates mean and std over seeds and writes
site/data.json plus a self-contained site/index.html (plain HTML + JS, no server).
Configs with ``leaderboard: false`` are left out.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout without install

from dcgridbench.data.windows import TASKS  # noqa: E402
from dcgridbench.models import registry  # noqa: E402


def config_notes(name):
    try:
        return registry.load_config(name).get("notes") or ""
    except FileNotFoundError:
        return ""

HEAD_VAR, HEAD_GEN, HEAD_CR = "online_regime", "schaake", "0.1"


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# column id -> (stage, label, extractor, lower is better)
COLUMNS = {
    "crps": ("1", "CRPS (kW)", lambda m: get(m, "stage1", "crps_kw"), True),
    "mae": ("1", "MAE (kW)", lambda m: get(m, "stage1", "mae_kw"), True),
    "cov90": ("2", "Cov 90% raw", lambda m: get(m, "stage2", "cov90"), None),
    "width90": ("2", "Width 90% (kW)", lambda m: get(m, "stage2", "width90_kw"), True),
    "winkler90": ("2", "Winkler 90%", lambda m: get(m, "stage2", "winkler90"), True),
    "cov90_min_reg": ("2", "Worst-regime cov 90%", lambda m: min((get(m, "stage2", "cov90_by_regime") or {}).values(), default=None), None),
    "crps_cal": ("2", "CRPS online-reg", lambda m: get(m, "calibration", HEAD_VAR, "stage1", "crps_kw"), True),
    "cov90_cal": ("2", "Cov 90% online-reg", lambda m: get(m, "calibration", HEAD_VAR, "stage2", "cov90"), None),
    "cov90_cal_min_reg": ("2", "Worst-regime cov 90% online-reg",
                          lambda m: min((get(m, "calibration", HEAD_VAR, "stage2", "cov90_by_regime") or {}).values(), default=None), None),
    "es": ("3", "Energy score", lambda m: get(m, "stage3", HEAD_VAR, HEAD_GEN, "energy_score"), True),
    "vs": ("3", "Variogram score", lambda m: get(m, "stage3", HEAD_VAR, HEAD_GEN, "variogram_score"), True),
    "brier": ("3", "Exceed. Brier", lambda m: get(m, "stage3", HEAD_VAR, HEAD_GEN, "exceed_brier"), True),
    "ks_mean": ("3", "Mean KS (grid stats)",
                lambda m: (lambda d: float(np.mean(list(d.values()))) if d else None)(get(m, "stage3", HEAD_VAR, HEAD_GEN, "pit_ks")), True),
    "es_native": ("3", "Energy score, own paths", lambda m: get(m, "stage3", "raw", "native", "energy_score"), True),
    "block_cost": ("4", "Block reserve cost @0.1", lambda m: get(m, "stage4", HEAD_VAR, "block", HEAD_GEN, HEAD_CR, "cost"), True),
    "block_short": ("4", "Block shortfall prob @0.1", lambda m: get(m, "stage4", HEAD_VAR, "block", HEAD_GEN, HEAD_CR, "shortfall_prob"), None),
    "block_reserve": ("4", "Block reserve (kW)", lambda m: get(m, "stage4", HEAD_VAR, "block", HEAD_GEN, HEAD_CR, "reserve_kw"), None),
    "step_cost": ("4", "Per-step cost @0.1", lambda m: get(m, "stage4", HEAD_VAR, "per_step", HEAD_CR, "cost"), True),
    "block_cost_native": ("4", "Block cost @0.1, own paths", lambda m: get(m, "stage4", "raw", "block", "native", HEAD_CR, "cost"), True),
}
NOMINAL = {"cov90": 0.9, "cov90_min_reg": 0.9, "cov90_cal": 0.9, "cov90_cal_min_reg": 0.9, "block_short": 0.1}


def aggregate(results: Path, selected: Path | None):
    groups = defaultdict(list)
    for p in sorted(results.glob("*/*/seed*.json")):
        r = json.loads(p.read_text())
        if r.get("purpose", "final") != "final" or not r["model"].get("leaderboard", True):
            continue
        groups[(r["task"], r["model"]["name"])].append(r)
    rows = []
    for (task, name), rs in sorted(groups.items()):
        m0 = rs[0]["model"]
        row = {"task": task, "model": name, "display_name": m0.get("display_name", name),
               "family": m0["family"], "library": m0["library"], "library_version": m0.get("library_version"),
               "zero_shot": m0.get("zero_shot", False), "notes": m0.get("notes") or config_notes(name), "seeds": sorted(r["seed"] for r in rs),
               "fit_seconds": float(np.mean([r["runtime"]["fit_seconds"] for r in rs])),
               "metrics": {}}
        for cid, (_, _, f, _) in COLUMNS.items():
            v = [f(r["metrics"]["test"]) for r in rs]
            v = [x for x in v if x is not None]
            if v:
                row["metrics"][cid] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        if selected is not None and (selected / task / f"{name}.json").exists():
            s = json.loads((selected / task / f"{name}.json").read_text())
            row["selection"] = {"trial": s["trial"], "val_crps_mean": s["val_crps_mean"],
                                "n_trials": s["n_trials"], "n_complete": s["n_complete"], "override": s["override"]}
        rows.append(row)
    return rows


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DCGridBench leaderboard</title>
<style>
:root{--bg:#fff;--fg:#1d2330;--mut:#5c6578;--line:#dfe3ea;--head:#f3f5f8;--best:#e3f2e6;--acc:#2f5fb3}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e4e7ec;--mut:#9aa3b2;--line:#2c323c;--head:#1c2027;--best:#1f3a27;--acc:#7aa7f0}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:16px}
main{max-width:1200px;margin:0 auto}h1{font-size:22px;margin:4px 0}h2{font-size:17px;margin:22px 0 6px}
p,li{color:var(--mut)}a{color:var(--acc)}.tabs button{font:inherit;border:1px solid var(--line);background:var(--head);color:var(--fg);padding:6px 12px;border-radius:6px;cursor:pointer;margin:0 6px 6px 0}
.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap}
th{background:var(--head);cursor:pointer;user-select:none;position:sticky;top:0}th:first-child,td:first-child{text-align:left}
td.note{text-align:left;color:var(--mut);white-space:normal;min-width:180px}td.best{background:var(--best);font-weight:600}
.std{color:var(--mut);font-size:12px}
</style></head><body><main>
<h1>DCGridBench leaderboard</h1>
<p>Grid-oriented probabilistic load forecasting of an AI data center (MIT Supercloud TX-Gaia, 448 V100).
Test period 2021-08-01 to 09-30, forecasts at every step. Values are mean &plusmn; std over seeds (n in brackets).
Stage 3&ndash;4 headline: online regime-conditioned conformal quantiles with the Schaake shuffle; reserve cost with c<sub>S</sub>=1, c<sub>R</sub>=0.1.
Coverage targets: 0.90; shortfall target: 0.10. Protocol: <code>docs/protocol.md</code>. Generated __GENERATED__.</p>
<div class="tabs" id="tabs"></div><div id="body"></div>
<script>
const DATA = __DATA__;
const STAGES = {"1":"Stage 1: forecast accuracy","2":"Stage 2: calibration","3":"Stage 3: scenarios","4":"Stage 4: reserve decision"};
let task = DATA.tasks[0];
function fmt(m){ if(!m) return "&ndash;"; const d = Math.abs(m.mean)>=100?1:3;
  return m.mean.toFixed(d)+(m.n>1?' <span class="std">&plusmn;'+m.std.toFixed(d)+' ('+m.n+')</span>':''); }
function score(cid, v){ const c = DATA.columns[cid]; if(v==null) return null;
  if(c.lower===true) return v; if(c.nominal!=null) return Math.abs(v-c.nominal); return null; }
function table(stage){
  const cols = Object.keys(DATA.columns).filter(c=>DATA.columns[c].stage===stage);
  const rows = DATA.rows.filter(r=>r.task===task && cols.some(c=>r.metrics[c]));
  const best = {}; cols.forEach(c=>{ let b=null; rows.forEach(r=>{const s=score(c, r.metrics[c]&&r.metrics[c].mean); if(s!=null&&(b==null||s<b)) b=s;}); best[c]=b; });
  let h = '<h2>'+STAGES[stage]+'</h2><div class="wrap"><table data-stage="'+stage+'"><thead><tr><th>Model</th>';
  cols.forEach(c=>h+='<th data-c="'+c+'">'+DATA.columns[c].label+'</th>');
  h += '<th>Library</th><th>Notes</th></tr></thead><tbody>';
  rows.forEach(r=>{ h+='<tr><td>'+r.display_name+'</td>';
    cols.forEach(c=>{ const m=r.metrics[c]; const s=score(c, m&&m.mean);
      h+='<td data-v="'+(m?m.mean:'')+'"'+(s!=null&&s===best[c]?' class="best"':'')+'>'+fmt(m)+'</td>'; });
    h+='<td>'+r.library+' '+(r.library_version||'')+'</td><td class="note">'+r.notes+'</td></tr>'; });
  return h+'</tbody></table></div>';
}
function render(){
  document.getElementById('tabs').innerHTML = DATA.tasks.map(t=>'<button class="'+(t===task?'on':'')+'" onclick="task=\\''+t+'\\';render()">'+t+'</button>').join('');
  document.getElementById('body').innerHTML = ["1","2","3","4"].map(table).join('');
  document.querySelectorAll('th[data-c]').forEach(th=>th.onclick=()=>sortBy(th));
}
function sortBy(th){
  const tb = th.closest('table').tBodies[0], i = [...th.parentNode.children].indexOf(th);
  const asc = th.dataset.asc!=='1'; th.dataset.asc = asc?'1':'0';
  [...tb.rows].sort((a,b)=>{ const x=parseFloat(a.cells[i].dataset.v), y=parseFloat(b.cells[i].dataset.v);
    if(isNaN(x)) return 1; if(isNaN(y)) return -1; return asc?x-y:y-x; }).forEach(r=>tb.appendChild(r));
}
render();
</script></main></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--selected", default=None)
    ap.add_argument("--out", default="site")
    a = ap.parse_args()
    rows = aggregate(Path(a.results), Path(a.selected) if a.selected else None)
    from datetime import date
    data = {"tasks": [t for t in TASKS if any(r["task"] == t for r in rows)],
            "columns": {c: {"stage": s, "label": l, "lower": lo, "nominal": NOMINAL.get(c)}
                        for c, (s, l, _, lo) in COLUMNS.items()},
            "rows": rows, "generated": date.today().isoformat()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, indent=1))
    page = PAGE.replace("__DATA__", json.dumps(data)).replace("__GENERATED__", data["generated"])
    (out / "index.html").write_text(page)
    (out / ".nojekyll").write_text("")
    print(f"{len(rows)} rows, tasks {data['tasks']} -> {out}/index.html")


if __name__ == "__main__":
    main()
