"""LOO round 7: local-linear scale regression + re-tuning under scale
weighting. Same protocol/metrics as dev_loo4. Dev tool only."""
import math

import dev_loo4 as L
from dev_loo5 import make_model, run

Z = L.Z


def ess(weights):
    s = sum(weights)
    return s * s / sum(w * w for w in weights) if s > 0 else 0.0


def fc_loclin(train, tfeat, p1, hours, bw=1.0, lam=0.9, sigma_shrink=0.95,
              adaptive_ess=0.0, clamp_pad=0.3, slope_shrink=1.0):
    """Per hour: kernel-weighted linear fit of log-ratio on (logp1 - target
    logp1); centre = intercept (optionally slope-shrunk), sigma from the
    weighted MAD of residuals. Falls back to the weighted median when the
    fit is degenerate."""
    tl = tfeat["logp1"]
    base = []
    for n, d in train:
        w0 = 1.0 if d["rolling"] else 0.5
        base.append((n, d, w0))
    out = {}
    for h in hours:
        rows = [(d["lr"][h], w0, d["feat"]["logp1"] - tl, n) for n, d, w0 in base if h in d["lr"]]
        if not rows:
            out[h] = [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
            continue
        b = bw
        while True:
            ws = [w0 * math.exp(-(x ** 2) / (2 * b * b)) for _v, w0, x, _n in rows]
            if adaptive_ess <= 0 or ess(ws) >= adaptive_ess or b >= 3.0:
                break
            b *= 1.25
        pairs = [(v, w, n) for (v, _w0, _x, n), w in zip(rows, ws)]
        med = L.wquantile(pairs, 0.5)
        # weighted linear fit v ~ a + c*x
        sw = sum(ws)
        mx = sum(w * x for (_v, _w0, x, _n), w in zip(rows, ws)) / sw
        mv = sum(w * v for (v, _w0, _x, _n), w in zip(rows, ws)) / sw
        sxx = sum(w * (x - mx) ** 2 for (_v, _w0, x, _n), w in zip(rows, ws))
        sxv = sum(w * (x - mx) * (v - mv) for (v, _w0, x, _n), w in zip(rows, ws))
        if sxx > 1e-9 and ess(ws) >= 4:
            slope = slope_shrink * sxv / sxx
            centre_raw = mv + slope * (0.0 - mx)  # predict at x=0 (target scale)
            resid = [(abs(v - (mv + slope * (x - mx))), w, n) for (v, _w0, x, n), w in zip(rows, ws)]
        else:
            centre_raw = med
            resid = [(abs(v - med), w, n) for v, w, n in pairs]
        lo = min(v for v, _w, _x, _n in rows) - clamp_pad
        hi = max(v for v, _w, _x, _n in rows) + clamp_pad
        centre_raw = min(max(centre_raw, lo), hi)
        sigma = 1.4826 * L.wquantile(resid, 0.5) * sigma_shrink
        sigma = max(sigma, 0.05, 0.5 * 0.15 * math.sqrt(h - 1))
        out[h] = [p1 * math.exp(lam * centre_raw + z * sigma) for z in Z]
    return out


def main():
    curves = L.load()
    for d in curves.values():
        d["feat"]["console"] = d["console"]
    _r, _a, v1_pc = L.evaluate(curves, L.fc_v1)
    print(f"{'v1 (reference)':46s} rolling {_r:.4f}   all36 {_a:.4f}")
    print("\n-- local-linear on logp1 --")
    for bw in [0.75, 1.0, 1.5]:
        for lam in [0.9, 1.0]:
            run(curves, f"loclin bw={bw} lam={lam}",
                lambda tr, tf, p1, hrs, bw=bw, lam=lam: fc_loclin(tr, tf, p1, hrs, bw=bw, lam=lam), v1_pc)
    run(curves, "loclin bw=1.0 lam=0.9 ess>=8",
        lambda tr, tf, p1, hrs: fc_loclin(tr, tf, p1, hrs, bw=1.0, lam=0.9, adaptive_ess=8.0), v1_pc)
    run(curves, "loclin bw=1.0 lam=0.9 slope_shrink=0.5",
        lambda tr, tf, p1, hrs: fc_loclin(tr, tf, p1, hrs, bw=1.0, lam=0.9, slope_shrink=0.5), v1_pc)
    print("\n-- re-tune lam / sigma_shrink under pure kernel bw=1.0 --")
    for lam in [0.85, 0.95, 1.0]:
        run(curves, f"kernel bw=1.0 lam={lam}", make_model(bw_logp1=1.0, lam0=lam), v1_pc)
    for sh in [0.85, 0.9, 1.0]:
        run(curves, f"kernel bw=1.0 shrink={sh}", make_model(bw_logp1=1.0, sigma_shrink=sh), v1_pc)
    print("\n-- kernel bw grid finer --")
    for bw in [0.6, 0.85, 1.15, 1.3]:
        run(curves, f"kernel bw={bw}", make_model(bw_logp1=bw), v1_pc)
    print("\n-- add feature betas on top of kernel bw=1.0 --")
    run(curves, "kernel + pre6 beta=0.3", make_model(bw_logp1=1.0, feat="pre6", beta=0.3), v1_pc)
    run(curves, "kernel + pre6 beta=0.15", make_model(bw_logp1=1.0, feat="pre6", beta=0.15), v1_pc)
    run(curves, "kernel + slope12 beta=-0.3", make_model(bw_logp1=1.0, feat="slope12", beta=-0.3), v1_pc)
    run(curves, "kernel + bw_pre24=0.8", make_model(bw_logp1=1.0, bw_pre24=0.8), v1_pc)
    print("\n-- smoothing on top of kernel bw=1.0 --")
    run(curves, "kernel + smooth med=1 sig=1", make_model(bw_logp1=1.0, smooth_med=1, smooth_sig=1), v1_pc)


if __name__ == "__main__":
    main()
