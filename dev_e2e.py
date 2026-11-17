"""End-to-end fidelity check: run the REAL main.py (and v1 from git) on
pseudo-case dicts built exactly as a server case is shaped, and score.
The numbers must match the curve-harness LOO (dev_loo9) for the shipped
config and dev_loo4 for v1. Dev tool only."""
import importlib.util
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import dev_loo4 as L

KIT = L.KIT
REPO = Path.home() / "workspace/makefun/bot-gta6"


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pseudo_case(data, target_name):
    comps = data["comparables"]
    tgt = next(c for c in comps if c["name"] == target_name)
    ds = L.parse_ts(tgt["day_start"])
    checkpoint = ds + timedelta(hours=1)
    hist = [p for p in tgt["hourly"] if L.parse_ts(p["from"]) < checkpoint]
    cv = L.day_curve(tgt)
    return {
        "id": "pseudo",
        "day_start": tgt["day_start"],
        "known": [{"hour": 1, "viewers": cv[1]}],
        "forecast": [{"hour": h} for h in range(2, 25)],
        "comparables": [c for c in comps if c["name"] != target_name],
        "history_hourly": hist,
    }


def e2e(mod, data, curves):
    roll, all_, per = [], [], {}
    for name, d in sorted(curves.items()):
        case = pseudo_case(data, name)
        res = mod.forecast_case(case)
        fc = {r["hour"]: r["q"] for r in res["hours"]}
        answers = {h: d["cv"][h] for h in d["cv"] if h >= 2}
        s = L.score(fc, answers)
        all_.append(s)
        if d["rolling"]:
            roll.append(s)
            per[name] = s
    return sum(roll) / len(roll), sum(all_) / len(all_), per


def main():
    data = json.loads((KIT / "input" / "p00.in").read_text())
    curves = L.load()
    v1_src = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:main.py"],
                            capture_output=True, text=True, check=True).stdout
    Path("/tmp/main_v1.py").write_text(v1_src)
    v1 = load_mod("/tmp/main_v1.py", "main_v1")
    v2 = load_mod(REPO / "main.py", "main_v2")
    for label, mod in [("v1 (git HEAD)", v1), ("v2 (working tree)", v2)]:
        r, a, _per = e2e(mod, data, curves)
        print(f"{label:20s} e2e rolling {r:.4f}   all36 {a:.4f}")
    _r, _a, per = e2e(v2, data, curves)
    print("\nv2 per-case (rolling targets):")
    for n, s in sorted(per.items()):
        print(f"  {n:38s} {s:.4f}")


if __name__ == "__main__":
    main()
