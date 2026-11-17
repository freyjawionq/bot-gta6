"""LOO round 4: v2 candidate search for the GTA VI CCU forecaster.

Same protocol as dev_loo/dev_loo2/dev_loo3: pseudo-cases are the usable
comparables of p00.in (36; 10 rolling-midnight), each forecast from the
other 35, scored with the contest pinball loss on the log scale
(2 * max(tau*d, (tau-1)*d), d = ln y - ln q), meaned per case then over
cases. Primary metric: rolling-target mean (v1 shipped = 0.3149,
persistence baseline = 0.3743). Guard metric: all-36 mean.

Dev tool only; not part of the entry.
"""
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

KIT = Path.home() / "workspace/makefun/gta6-twitch-ccu-forecast-01-kit"
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def is_rolling(comp):
    u = (comp.get("unlock") or "").lower()
    return ("midnight local" in u or "rolling" in u) and "simultaneous" not in u.split(";")[0]


def is_console(comp):
    p = (comp.get("platforms") or "").lower()
    return ("ps" in p or "xbox" in p or "playstation" in p) and "pc only" not in p


def day_curve(comp):
    ds = comp["day_start"][:10]
    out = {}
    for p in comp["hourly"]:
        if p["from"][:10] == ds and p.get("viewers"):
            out[int(p["from"][11:13]) + 1] = float(p["viewers"])
    return out


def features(comp, cv):
    """Features computable identically from a real case (history_hourly +
    known) and from a comparable's own hourly series. All pre-release
    features use only points strictly before day_start."""
    ds = parse_ts(comp["day_start"])
    logp1 = math.log(cv[1])
    pts = []
    for p in comp["hourly"]:
        if p.get("viewers"):
            pts.append((parse_ts(p["from"]), float(p["viewers"]), p.get("channels")))
    pts.sort(key=lambda t: t[0])
    pre = [(t, v, c) for t, v, c in pts if t < ds]
    f = {"logp1": logp1}

    def mean_log_last(hours):
        lo = ds - timedelta(hours=hours)
        vals = [math.log(v) for t, v, _c in pre if t >= lo]
        return sum(vals) / len(vals) if vals else None

    m24 = mean_log_last(24)
    m6 = mean_log_last(6)
    f["pre24"] = (m24 - logp1) if m24 is not None else None  # anticipation level vs hour 1
    f["pre6"] = (m6 - logp1) if m6 is not None else None
    # slope of ln viewers over the last 12 pre-release hours (per hour)
    lo = ds - timedelta(hours=12)
    seg = [(t, math.log(v)) for t, v, _c in pre if t >= lo]
    if len(seg) >= 6:
        xs = [(t - seg[0][0]).total_seconds() / 3600.0 for t, _v in seg]
        ys = [v for _t, v in seg]
        xm = sum(xs) / len(xs)
        ym = sum(ys) / len(ys)
        den = sum((x - xm) ** 2 for x in xs)
        f["slope12"] = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den if den > 0 else None
    else:
        f["slope12"] = None
    # channels at hour 1 and viewers per channel
    ch1 = None
    for t, _v, c in pts:
        if t == ds:
            ch1 = c
            break
    f["ch1"] = float(ch1) if ch1 else None
    f["vpc1"] = (cv[1] / ch1) if ch1 else None
    f["logvpc1"] = math.log(cv[1] / ch1) if ch1 else None
    return f


def load(casefile="p00.in"):
    data = json.loads((KIT / "input" / casefile).read_text())
    curves = {}
    for c in data["comparables"]:
        if "not a launch" in (c.get("unlock") or "").lower():
            continue
        cv = day_curve(c)
        if 1 in cv and cv[1] >= 200 and sum(1 for h in cv if 2 <= h <= 24) >= 20:
            curves[c["name"]] = {
                "cv": cv,
                "rolling": is_rolling(c),
                "console": is_console(c),
                "feat": features(c, cv),
                "lr": None,
            }
    for d in curves.values():
        base = math.log(d["cv"][1])
        d["lr"] = {h: math.log(v) - base for h, v in d["cv"].items() if h >= 2}
    return curves


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


def wquantile(pairs, tau):
    """pairs: [(value, weight, name)] — main.py's weighted quantile."""
    ordered = sorted(pairs, key=lambda p: (p[0], p[2]))
    total = sum(w for _v, w, _n in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    acc = 0.0
    for i, (value, weight, _n) in enumerate(ordered):
        acc += weight
        if acc >= tau * total:
            if acc == tau * total and i + 1 < len(ordered):
                return 0.5 * (value + ordered[i + 1][0])
            return value
    return ordered[-1][0]


# ---------------------------------------------------------------- models
# A model takes (train, target_feat, p1, hours) where train is a list of
# (name, curve-dict) excluding the target, and returns {h: [q05..q95]}.

def fc_baseline(train, tfeat, p1, hours):
    return {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z] for h in hours}


def fc_v1(train, tfeat, p1, hours):
    out = {}
    for h in hours:
        pairs = [(d["lr"][h], 1.0 if d["rolling"] else 0.5, n) for n, d in train if h in d["lr"]]
        if not pairs:
            out[h] = [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
            continue
        med = wquantile(pairs, 0.5)
        devs = [(abs(v - med), w, n) for v, w, n in pairs]
        sigma = 1.4826 * wquantile(devs, 0.5) * 0.95
        sigma = max(sigma, 0.05, 0.5 * 0.15 * math.sqrt(h - 1))
        out[h] = [p1 * math.exp(0.9 * med + z * sigma) for z in Z]
    return out


def evaluate(curves, model, rolling_only_targets=True):
    """Returns (rolling_mean, all_mean, per_case_rolling dict)."""
    roll_losses, all_losses, per_case = [], [], {}
    for name, d in sorted(curves.items()):
        p1 = d["cv"][1]
        answers = {h: d["cv"][h] for h in d["cv"] if h >= 2}
        hours = sorted(answers)
        train = [(n, curves[n]) for n in curves if n != name]
        s = score(model(train, d["feat"], p1, hours), answers)
        all_losses.append(s)
        if d["rolling"]:
            roll_losses.append(s)
            per_case[name] = s
    return (sum(roll_losses) / len(roll_losses), sum(all_losses) / len(all_losses), per_case)


def main():
    curves = load()
    print(f"usable curves: {len(curves)} (rolling: {sum(1 for d in curves.values() if d['rolling'])})")
    feat_keys = ["pre24", "pre6", "slope12", "ch1", "logvpc1"]
    for k in feat_keys:
        have = sum(1 for d in curves.values() if d["feat"][k] is not None)
        print(f"  feature {k}: available for {have}/{len(curves)}")
    for label, model in [("baseline", fc_baseline), ("v1", fc_v1)]:
        r, a, _pc = evaluate(curves, model)
        print(f"{label:12s} rolling {r:.4f}   all36 {a:.4f}")


if __name__ == "__main__":
    main()
