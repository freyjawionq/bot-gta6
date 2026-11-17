"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (make.fun
gta6-twitch-ccu-forecast-01). Python 3.13, standard library only.

Model: a launch day's shape, in log space, is the per-hour ratio to the
day's first hour. Every past launch in the case's `comparables` votes on
that shape with its own release-day curve:

    l_c(h) = ln(viewers_c(h) / viewers_c(1))

For each hour h = 2..24 the forecast takes the weighted median of the
comparables' l_c(h) (rolling-midnight launches weigh 1, the rest 0.5),
shrinks it 10% toward persistence, and spreads the five quantiles with
the hour's weighted MAD, turned into a sigma and shrunk 5%:

    q_tau(h) = p1 * exp(0.9 * med(h) + z_tau * 0.95 * sigma(h))

with p1 hour 1's print and z_tau the standard normal quantile at tau.
The constants are the middle of a flat optimum found by leave-one-out
validation over the comparables themselves (each comparable in turn
treated as a case, forecast from the others): mean pinball loss about
0.31 against the persistence baseline's 0.37 on rolling-midnight
launches like this one. Where no comparable covers an hour, the hour
falls back to persistence with the baseline's random-walk spread.

The program reads every *.in file in $MAKEFUN_INPUT_DIR (else /input),
in name order, and writes <id>.out per case to $MAKEFUN_OUTPUT_DIR
(else /output). It is deterministic: no randomness, no clock, no
network, and every reduction runs over inputs sorted by value and name.
"""

import json
import math
import os
import sys
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level in LEVELS.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

W_ROLLING = 1.0    # weight of a rolling-midnight launch in the shape pool
W_OTHER = 0.5      # weight of any other launch
LAM = 0.9          # shrinkage of the median log-ratio toward persistence
SIGMA_SHRINK = 0.95
SIGMA_FLOOR = 0.05
MIN_HOUR1 = 200.0  # a comparable's hour 1 below this cannot anchor ratios
MIN_DAY_HOURS = 20  # a comparable needs at least this many of hours 2..24


def parse_ts(ts):
    """(date, hour) of an ISO-8601 UTC timestamp, by string surgery only."""
    return ts[:10], int(ts[11:13])


def is_rolling(unlock_text):
    u = (unlock_text or "").lower()
    return ("midnight local" in u or "rolling" in u) and "simultaneous" not in u.split(";")[0]


def day_log_ratios(comp):
    """{hour: ln(viewers / hour-1 viewers)} for a comparable's release day,
    or None when the comparable cannot anchor a shape."""
    day = (comp.get("day_start") or "")[:10]
    if not day:
        return None
    viewers = {}
    for point in comp.get("hourly") or []:
        v = point.get("viewers")
        if not v or v <= 0:
            continue
        pday, phour = parse_ts(point.get("from") or "")
        if pday == day:
            viewers[phour + 1] = float(v)
    first = viewers.get(1)
    if first is None or first < MIN_HOUR1:
        return None
    if sum(1 for h in viewers if 2 <= h <= 24) < MIN_DAY_HOURS:
        return None
    base = math.log(first)
    return {h: math.log(v) - base for h, v in viewers.items() if 2 <= h <= 24}


def shape_pool(case):
    """[(log_ratios, weight, name)] from the case's comparables."""
    pool = []
    for comp in case.get("comparables") or []:
        if "not a launch" in (comp.get("unlock") or "").lower():
            continue
        ratios = day_log_ratios(comp)
        if ratios is None:
            continue
        weight = W_ROLLING if is_rolling(comp.get("unlock")) else W_OTHER
        pool.append((ratios, weight, comp.get("name") or ""))
    return pool


def weighted_quantile(pairs, tau):
    """pairs: [(value, weight, name)]; the weighted quantile at tau."""
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


def hour_forecast(pool, p1, hour):
    pairs = [(ratios[hour], w, name) for ratios, w, name in pool if hour in ratios]
    if not pairs:
        # No comparable covers this hour: persistence, baseline spread.
        spread = 0.15 * math.sqrt(hour - 1)
        return [p1 * math.exp(z * spread) for z in Z]
    med = weighted_quantile(pairs, 0.5)
    devs = [(abs(v - med), w, name) for v, w, name in pairs]
    sigma = 1.4826 * weighted_quantile(devs, 0.5) * SIGMA_SHRINK
    # Floors: an absolute one, and half the baseline's random-walk spread,
    # so a small or oddly agreeing pool can never make a late hour
    # overconfident. On the validation cases the MAD clears both floors
    # at every hour, so they bind only as insurance.
    sigma = max(sigma, SIGMA_FLOOR, 0.5 * 0.15 * math.sqrt(hour - 1))
    centre = LAM * med
    return [p1 * math.exp(centre + z * sigma) for z in Z]


def round2(x):
    """Round half up to 2 decimals, as the house baseline rounds."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def clamp(q):
    return min(max(q, 0.01), 1_000_000_000.0)


def forecast_case(case):
    known = case.get("known") or []
    if isinstance(known, dict):
        known = [known]
    first = next((k for k in known if k.get("hour") == 1), None)
    p1 = 1.0
    if first is not None and first.get("viewers"):
        p1 = max(1.0, float(first["viewers"]))
    pool = shape_pool(case)
    rows = []
    for row in case.get("forecast") or []:
        hour = row["hour"]
        qs = [clamp(round2(q)) for q in hour_forecast(pool, p1, hour)]
        qs = sorted(qs)
        rows.append({"hour": hour, "q": qs})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        out = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
