"""Leave-one-out validation of forecast models over the kit's comparables.

Each comparable with a usable release-day curve becomes a pseudo-case:
known = its release-day hour 1, answers = hours 2..24, training = the other
curves. Losses are the contest's mean pinball loss on the log scale.
Dev tool only; not part of the entry.
"""
import json
import math
import sys
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


def load_curves(casefile):
    data = json.loads(Path(casefile).read_text())
    curves = {}
    for c in data["comparables"]:
        if "not a launch" in c["unlock"].lower():
            continue
        cv = day_curve(c)
        if 1 in cv and cv[1] >= 200 and sum(1 for h in cv if 2 <= h <= 24) >= 20:
            curves[c["name"]] = (cv, is_rolling(c))
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


def log_ratios(curves):
    """name -> {h: ln(v_h / v_1)}"""
    out = {}
    for name, (cv, _r) in curves.items():
        base = math.log(cv[1])
        out[name] = {h: math.log(v) - base for h, v in cv.items() if h >= 2}
    return out


def fc_baseline(p1, hours):
    return {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z] for h in hours}


def fc_empirical(train_lr, p1, hours):
    out = {}
    for h in hours:
        vals = sorted(lr[h] for lr in train_lr.values() if h in lr)
        out[h] = [p1 * math.exp(quantile(vals, t)) for t in LEVELS]
    return out


def fc_normal(train_lr, p1, hours, sigma_mode="mad", shrink=1.0, floor=0.05):
    out = {}
    for h in hours:
        vals = [lr[h] for lr in train_lr.values() if h in lr]
        med = quantile(sorted(vals), 0.5)
        if sigma_mode == "mad":
            dev = sorted(abs(v - med) for v in vals)
            sig = 1.4826 * quantile(dev, 0.5)
        else:  # std
            mu = sum(vals) / len(vals)
            sig = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        sig = max(sig * shrink, floor)
        out[h] = [p1 * math.exp(med + z * sig) for z in Z]
    return out


def main():
    curves = load_curves(KIT / "input" / "p00.in")
    print(f"usable curves: {len(curves)} (rolling: {sum(1 for _c, r in curves.values() if r)})")
    lr = log_ratios(curves)
    variants = {
        "baseline": lambda tr, p1, hrs: fc_baseline(p1, hrs),
        "empirical": lambda tr, p1, hrs: fc_empirical(tr, p1, hrs),
        "normal-mad": lambda tr, p1, hrs: fc_normal(tr, p1, hrs, "mad"),
        "normal-std": lambda tr, p1, hrs: fc_normal(tr, p1, hrs, "std"),
        "normal-mad-x1.15": lambda tr, p1, hrs: fc_normal(tr, p1, hrs, "mad", 1.15),
        "normal-mad-x0.85": lambda tr, p1, hrs: fc_normal(tr, p1, hrs, "mad", 0.85),
    }
    rolling_only = {k: v for k, v in variants.items() if k != "baseline"}
    agg = {k: [] for k in variants}
    agg_r = {k: [] for k in rolling_only}
    for name, (cv, roll) in sorted(curves.items()):
        p1 = cv[1]
        answers = {h: cv[h] for h in cv if h >= 2}
        hours = sorted(answers)
        train = {n: lr[n] for n in lr if n != name}
        train_r = {n: lr[n] for n in lr if n != name and curves[n][1]}
        for k, f in variants.items():
            agg[k].append(score(f(train, p1, hours), answers))
        if roll:
            for k, f in rolling_only.items():
                agg_r[k].append(score(f(train_r, p1, hours), answers))
    print("\n== all pseudo-cases, full training pool ==")
    for k, v in agg.items():
        print(f"{k:20s} mean loss {sum(v) / len(v):.4f}  (n={len(v)})")
    print("\n== rolling-unlock pseudo-cases, rolling-only training pool ==")
    for k, v in agg_r.items():
        print(f"{k:20s} mean loss {sum(v) / len(v):.4f}  (n={len(v)})")


if __name__ == "__main__":
    main()
