"""LOO round 9: final combination grid in the promising region + clamp
sensitivity. Same protocol. Dev tool only."""
import math

import dev_loo4 as L
from dev_loo5 import run


def fc_v2(train, tfeat, p1, hours, bw=2.0, lam=0.95, sigma_shrink=0.95,
          pre6_beta=0.0, blend_med=0.0, console_mult=1.0, clamp_pad=0.3):
    tl = tfeat["logp1"]
    tcons = tfeat.get("console")
    out = {}
    for h in hours:
        rows = []
        for n, d in train:
            if h not in d["lr"]:
                continue
            w0 = 1.0 if d["rolling"] else 0.5
            if console_mult != 1.0 and d["feat"].get("console") != tcons:
                w0 *= console_mult
            x = d["feat"]["logp1"] - tl
            w = w0 * math.exp(-(x ** 2) / (2 * bw * bw))
            rows.append((d["lr"][h], w, x, n, d))
        if not rows:
            out[h] = [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in L.Z]
            continue
        sw = sum(w for _v, w, _x, _n, _d in rows)
        mx = sum(w * x for _v, w, x, _n, _d in rows) / sw
        mv = sum(w * v for v, w, _x, _n, _d in rows) / sw
        sxx = sum(w * (x - mx) ** 2 for _v, w, x, _n, _d in rows)
        sxv = sum(w * (x - mx) * (v - mv) for v, w, x, _n, _d in rows)
        slope = sxv / sxx if sxx > 1e-9 else 0.0
        centre_raw = mv + slope * (0.0 - mx)
        resid = [(abs(v - (mv + slope * (x - mx))), w, n) for v, w, x, n, _d in rows]
        if blend_med > 0:
            pairs = [(v, w, n) for v, w, _x, n, _d in rows]
            med = L.wquantile(pairs, 0.5)
            centre_raw = (1 - blend_med) * centre_raw + blend_med * med
        if clamp_pad is not None:
            lo = min(v for v, _w, _x, _n, _d in rows) - clamp_pad
            hi = max(v for v, _w, _x, _n, _d in rows) + clamp_pad
            centre_raw = min(max(centre_raw, lo), hi)
        if pre6_beta and tfeat.get("pre6") is not None:
            num = sum(w * d["feat"]["pre6"] for _v, w, _x, _n, d in rows if d["feat"]["pre6"] is not None)
            den = sum(w for _v, w, _x, _n, d in rows if d["feat"]["pre6"] is not None)
            if den > 0:
                centre_raw += pre6_beta * (tfeat["pre6"] - num / den)
        sigma = 1.4826 * L.wquantile(resid, 0.5) * sigma_shrink
        sigma = max(sigma, 0.05, 0.5 * 0.15 * math.sqrt(h - 1))
        out[h] = [p1 * math.exp(lam * centre_raw + z * sigma) for z in L.Z]
    return out


def main():
    curves = L.load()
    for d in curves.values():
        d["feat"]["console"] = d["console"]
    _r, _a, v1_pc = L.evaluate(curves, L.fc_v1)
    print(f"{'v1 (reference)':46s} rolling {_r:.4f}   all36 {_a:.4f}")
    cands = [
        ("bw2.0 lam0.95 (leader)", dict()),
        ("bw2.0 lam0.95 + pre6b .15", dict(pre6_beta=0.15)),
        ("bw1.5 lam1.0 + pre6b .15", dict(bw=1.5, lam=1.0, pre6_beta=0.15)),
        ("bw1.75 lam0.975 + pre6b .15", dict(bw=1.75, lam=0.975, pre6_beta=0.15)),
        ("bw2.0 lam0.95 + blend .3", dict(blend_med=0.3)),
        ("bw2.0 lam0.95 + console .67", dict(console_mult=0.67)),
        ("bw2.0 lam0.95 + console .5", dict(console_mult=0.5)),
        ("bw2.0 lam0.95 + pre6b .15 + blend .3", dict(pre6_beta=0.15, blend_med=0.3)),
        ("bw2.0 lam0.95 shrink 1.0", dict(sigma_shrink=1.0)),
        ("bw2.0 lam0.95 shrink 0.9", dict(sigma_shrink=0.9)),
        ("bw2.0 lam1.0 + pre6b .15", dict(lam=1.0, pre6_beta=0.15)),
        ("bw2.25 lam0.95 + pre6b .15", dict(bw=2.25, pre6_beta=0.15)),
    ]
    results = {}
    for label, kw in cands:
        r, a, pc = run(curves, label, lambda tr, tf, p1, hrs, kw=kw: fc_v2(tr, tf, p1, hrs, **kw), v1_pc)
        results[label] = (r, a, pc)
    print("\n-- clamp sensitivity on the leader family --")
    for cp in [0.15, 0.5, None]:
        run(curves, f"bw2.0 lam0.95 clamp={cp}",
            lambda tr, tf, p1, hrs, cp=cp: fc_v2(tr, tf, p1, hrs, clamp_pad=cp), v1_pc)
    print("\n-- ranking by rolling --")
    for label, (r, a, _pc) in sorted(results.items(), key=lambda kv: kv[1][0]):
        print(f"  {label:38s} rolling {r:.4f}   all36 {a:.4f}")


if __name__ == "__main__":
    main()
