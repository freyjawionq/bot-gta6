"""LOO round 2: pool selection, shrinkage, and quantile blending, judged on
rolling-unlock pseudo-cases (the final and all four practice cases are
rolling-midnight console launches). Dev tool only; not part of the entry.
"""
import json
import math
from pathlib import Path

KIT = Path.home() / "workspace/makefun/gta6-twitch-ccu-forecast-01-kit"
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]


def day_curve(comp):
    ds = comp["day_start"][:10]
    out = {}
    for p in comp["hourly"]:
        if p["from"][:10] == ds and p.get("viewers"):
            out[int(p["from"][11:13]) + 1] = float(p["viewers"])
    return out


def is_rolling(comp):
    u = comp["unlock"].lower()
    return ("midnight local" in u or "rolling" in u) and "simultaneous" not in u.split(";")[0]


def is_console(comp):
    p = comp["platforms"].lower()
    return ("ps" in p or "xbox" in p or "playstation" in p) and "pc only" not in p


def load(casefile):
    data = json.loads(Path(casefile).read_text())
    curves = {}
    for c in data["comparables"]:
        if "not a launch" in c["unlock"].lower():
            continue
        cv = day_curve(c)
        if 1 in cv and cv[1] >= 200 and sum(1 for h in cv if 2 <= h <= 24) >= 20:
            curves[c["name"]] = {"cv": cv, "rolling": is_rolling(c), "console": is_console(c)}
    return curves


def quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def pinball(y, q, tau):
    d = math.log(y) - math.log(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score(forecast, answers):
    tot, n = 0.0, 0
    for h, qs in forecast.items():
        if h not in answers:
            continue
        for tau, q in zip(LEVELS, qs):
            tot += pinball(answers[h], q, tau)
            n += 1
    return tot / n if n else float("nan")


def fc_normal(train, p1, hours, shrink=1.0, lam=1.0):
    out = {}
    for h in hours:
        vals = [lr[h] for lr in train.values() if h in lr]
        med = quantile(sorted(vals), 0.5)
        mu = sum(vals) / len(vals)
        sig = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        sig = max(sig * shrink, 0.05)
        out[h] = [p1 * math.exp(lam * med + z * sig) for z in Z]
    return out


def fc_empirical(train, p1, hours, lam=1.0):
    out = {}
    for h in hours:
        vals = sorted(lr[h] for lr in train.values() if h in lr)
        qs = [quantile(vals, t) for t in LEVELS]
        med = qs[2]
        out[h] = [p1 * math.exp(lam * med + (q - med)) for q in qs]
    return out


def blend(fa, fb, w=0.5):
    return {h: [math.exp(w * math.log(a) + (1 - w) * math.log(b)) for a, b in zip(fa[h], fb[h])] for h in fa}


def main():
    curves = load(KIT / "input" / "p00.in")
    lr = {}
    for name, d in curves.items():
        base = math.log(d["cv"][1])
        lr[name] = {h: math.log(v) - base for h, v in d["cv"].items() if h >= 2}
    pools = {
        "all": lambda n: True,
        "rolling": lambda n: curves[n]["rolling"],
        "console": lambda n: curves[n]["console"],
        "roll+cons": lambda n: curves[n]["rolling"] or curves[n]["console"],
    }
    results = {}
    per_case = {}
    for name, d in sorted(curves.items()):
        if not d["rolling"]:
            continue  # judge only on rolling targets: the final + practice are rolling
        p1 = d["cv"][1]
        answers = {h: d["cv"][h] for h in d["cv"] if h >= 2}
        hours = sorted(answers)
        trains = {pn: {n: lr[n] for n in lr if n != name and pf(n)} for pn, pf in pools.items()}
        cands = {
            "baseline": {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z] for h in hours},
            "norm-all": fc_normal(trains["all"], p1, hours),
            "norm-rolling": fc_normal(trains["rolling"], p1, hours),
            "norm-console": fc_normal(trains["console"], p1, hours),
            "norm-roll+cons": fc_normal(trains["roll+cons"], p1, hours),
            "norm-rolling-lam0.85": fc_normal(trains["rolling"], p1, hours, lam=0.85),
            "emp-rolling": fc_empirical(trains["rolling"], p1, hours),
            "blend-roll-all": blend(fc_normal(trains["rolling"], p1, hours), fc_normal(trains["all"], p1, hours)),
        }
        per_case[name] = {}
        for k, f in cands.items():
            s = score(f, answers)
            results.setdefault(k, []).append(s)
            per_case[name][k] = s
    print("== rolling pseudo-cases only (n=%d) ==" % len(per_case))
    for k, v in sorted(results.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"{k:22s} mean {sum(v) / len(v):.4f}")
    print("\nper-case (best three variants):")
    for name, row in per_case.items():
        print(f"  {name:35s} base {row['baseline']:.3f}  norm-roll {row['norm-rolling']:.3f}  blend {row['blend-roll-all']:.3f}")


if __name__ == "__main__":
    main()
