"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (make.fun
gta6-twitch-ccu-forecast-01). Python 3.13, standard library only.

Model (v2): a launch day's shape, in log space, is the per-hour ratio to
the day's first hour,

    l_c(h) = ln(viewers_c(h) / viewers_c(1)),

and that shape depends strongly on the launch's SCALE: a 400-viewer
hour 1 and a 900,000-viewer hour 1 bend very differently over the day.
So for each hour h = 2..24 the forecast fits, over the case's
comparables, a kernel-weighted linear regression of l_c(h) on the
comparable's log hour-1 size relative to this case's,

    l(h) ~ a(h) + b(h) * (ln p1_c - ln p1),

with Gaussian kernel bandwidth 2.0 in log size and base weights 1 for
rolling-midnight launches, 0.5 for the rest. The forecast centre is the
fit's intercept at this case's own scale, blended 30% toward the
kernel-weighted median for robustness, clamped to the pool's observed
range at that hour (plus 0.3), shifted by the case's pre-release
anticipation (the mean log viewers of the six hours before release
relative to hour 1, centred on the pool's kernel-weighted mean of the
same feature, coefficient 0.15), and finally shrunk 5% toward
persistence:

    centre(h) = 0.95 * fit(h)

The five quantiles spread around the centre with the hour's residual
dispersion: sigma is 1.4826 times the kernel-weighted MAD of the fit's
residuals, shrunk 5%, floored as in v1:

    q_tau(h) = p1 * exp(centre(h) + z_tau * sigma(h))

Validation is leave-one-out over the comparables themselves (each
usable comparable in turn treated as a case, forecast from the others;
mean contest pinball loss on the log scale). On rolling-midnight
launches (n = 10; the final and all four practice cases are of this
kind) v2 scores 0.240 against v1's 0.315 and the persistence baseline's
0.374; over all 36 usable comparables it scores 0.412 against v1's
0.745 (v1's global pool fails outright on the smallest launches, whose
shapes it cannot represent). Where no comparable covers an hour, the
hour falls back to persistence with the baseline's random-walk spread.

The program reads every *.in file in $MAKEFUN_INPUT_DIR (else /input),
in name order, and writes <id>.out per case to $MAKEFUN_OUTPUT_DIR
(else /output). It is deterministic: no randomness, no clock, no
network, and every reduction runs over inputs in a fixed order.
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
BW = 2.0           # Gaussian kernel bandwidth on log hour-1 size
LAM = 0.95         # shrinkage of the fitted centre toward persistence
BLEND_MED = 0.3    # centre blend toward the kernel-weighted median
CLAMP_PAD = 0.3    # centre clamp: pool range at the hour, plus this
PRE6_BETA = 0.15   # centre shift per unit of pre-release anticipation
SIGMA_SHRINK = 0.95
SIGMA_FLOOR = 0.05
MIN_HOUR1 = 200.0  # a comparable's hour 1 below this cannot anchor ratios
MIN_DAY_HOURS = 20  # a comparable needs at least this many of hours 2..24


def parse_ts(ts):
    """(date, hour) of an ISO-8601 UTC timestamp, by string surgery only."""
    return ts[:10], int(ts[11:13])


def serial_hour(ts):
    """Whole hours since a fixed epoch, from an ISO-8601 UTC timestamp,
    by integer arithmetic on its string parts (Howard Hinnant's
    days-from-civil). Only differences are ever used."""
    date, hour = parse_ts(ts)
    y = int(date[0:4])
    m = int(date[5:7])
    d = int(date[8:10])
    y -= 1 if m <= 2 else 0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return days * 24 + hour


def is_rolling(unlock_text):
    u = (unlock_text or "").lower()
    return ("midnight local" in u or "rolling" in u) and "simultaneous" not in u.split(";")[0]


def pre6_of(points, day_start):
    """Mean log viewers over the six hours before day_start, relative to
    nothing (raw mean); points are dicts with 'from' and 'viewers'.
    None when no point falls in the window."""
    try:
        end = serial_hour(day_start)
    except Exception:
        return None
    vals = []
    for point in points or []:
        v = point.get("viewers")
        if not v or v <= 0:
            continue
        try:
            s = serial_hour(point.get("from") or "")
        except Exception:
            continue
        if end - 6 <= s < end:
            vals.append(math.log(float(v)))
    if not vals:
        return None
    return sum(vals) / len(vals)


def day_shape(comp):
    """(log_ratios, log_hour1, pre6) for a comparable's release day, or
    None when the comparable cannot anchor a shape. pre6 is the mean log
    viewers of the six pre-release hours minus log hour 1 (or None)."""
    day = (comp.get("day_start") or "")[:10]
    if not day:
        return None
    viewers = {}
    for point in comp.get("hourly") or []:
        v = point.get("viewers")
        if not v or v <= 0:
            continue
        try:
            pday, phour = parse_ts(point.get("from") or "")
        except Exception:
            continue
        if pday == day:
            viewers[phour + 1] = float(v)
    first = viewers.get(1)
    if first is None or first < MIN_HOUR1:
        return None
    if sum(1 for h in viewers if 2 <= h <= 24) < MIN_DAY_HOURS:
        return None
    base = math.log(first)
    ratios = {h: math.log(v) - base for h, v in viewers.items() if 2 <= h <= 24}
    pre = pre6_of(comp.get("hourly"), comp.get("day_start") or "")
    return ratios, base, (pre - base if pre is not None else None)


def shape_pool(case):
    """[(log_ratios, weight, name, log_hour1, pre6)] from comparables."""
    pool = []
    for comp in case.get("comparables") or []:
        if "not a launch" in (comp.get("unlock") or "").lower():
            continue
        shape = day_shape(comp)
        if shape is None:
            continue
        ratios, logp1_c, pre6_c = shape
        weight = W_ROLLING if is_rolling(comp.get("unlock")) else W_OTHER
        pool.append((ratios, weight, comp.get("name") or "", logp1_c, pre6_c))
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


def hour_forecast(pool, p1, logp1, pre6, hour):
    rows = []
    for ratios, w0, name, logp1_c, pre6_c in pool:
        if hour not in ratios:
            continue
        x = logp1_c - logp1
        w = w0 * math.exp(-(x * x) / (2.0 * BW * BW))
        rows.append((ratios[hour], w, x, name, pre6_c))
    if not rows:
        # No comparable covers this hour: persistence, baseline spread.
        spread = 0.15 * math.sqrt(hour - 1)
        return [p1 * math.exp(z * spread) for z in Z]
    sw = sum(w for _v, w, _x, _n, _p in rows)
    mx = sum(w * x for _v, w, x, _n, _p in rows) / sw
    mv = sum(w * v for v, w, _x, _n, _p in rows) / sw
    sxx = sum(w * (x - mx) ** 2 for _v, w, x, _n, _p in rows)
    sxv = sum(w * (x - mx) * (v - mv) for v, w, x, _n, _p in rows)
    slope = sxv / sxx if sxx > 1e-9 else 0.0
    centre = mv + slope * (0.0 - mx)
    pairs = [(v, w, n) for v, w, _x, n, _p in rows]
    med = weighted_quantile(pairs, 0.5)
    centre = (1.0 - BLEND_MED) * centre + BLEND_MED * med
    lo = min(v for v, _w, _x, _n, _p in rows) - CLAMP_PAD
    hi = max(v for v, _w, _x, _n, _p in rows) + CLAMP_PAD
    centre = min(max(centre, lo), hi)
    if pre6 is not None:
        num = sum(w * pc for _v, w, _x, _n, pc in rows if pc is not None)
        den = sum(w for _v, w, _x, _n, pc in rows if pc is not None)
        if den > 0:
            centre += PRE6_BETA * (pre6 - num / den)
    resid = [(abs(v - (mv + slope * (x - mx))), w, n) for v, w, x, n, _p in rows]
    sigma = 1.4826 * weighted_quantile(resid, 0.5) * SIGMA_SHRINK
    # Floors: an absolute one, and half the baseline's random-walk spread,
    # so a small or oddly agreeing pool can never make a late hour
    # overconfident.
    sigma = max(sigma, SIGMA_FLOOR, 0.5 * 0.15 * math.sqrt(hour - 1))
    centre = LAM * centre
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
    logp1 = math.log(p1)
    pool = shape_pool(case)
    history = case.get("history_hourly") or case.get("history") or []
    pre = pre6_of(history, case.get("day_start") or "")
    pre6 = (pre - logp1) if pre is not None else None
    wanted = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    rows = []
    for row in wanted:
        hour = row["hour"]
        qs = [clamp(round2(q)) for q in hour_forecast(pool, p1, logp1, pre6, hour)]
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
