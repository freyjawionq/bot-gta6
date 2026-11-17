"""LOO round 8: push local-linear bandwidth wider, robustness variants,
per-case audit of the leader. Same protocol. Dev tool only."""
import math

import dev_loo4 as L
from dev_loo5 import run
from dev_loo7 import fc_loclin


def fc_loclin2(train, tfeat, p1, hours, bw=1.5, lam=1.0, sigma_shrink=0.95,
               quad=False, blend_med=0.0, pre6_beta=0.0):
    """fc_loclin + options: quadratic term in logp1, blending the fitted
    centre with the kernel weighted median, and a pre6 centre shift whose
    reference mean is computed with the same kernel weights."""
    tl = tfeat["logp1"]
    out = {}
    for h in hours:
        rows = []
        for n, d in train:
            if h not in d["lr"]:
                continue
            w0 = 1.0 if d["rolling"] else 0.5
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
        if quad:
            # add curvature: fit residual on (x-mx)^2 - sxx/sw
            basis = [(x - mx) ** 2 - sxx / sw for _v, _w, x, _n, _d in rows]
            sb = sum(w * b * b for (_v, w, _x, _n, _d), b in zip(rows, basis))
            sbv = sum(w * b * (v - mv - slope * (x - mx)) for (v, w, x, _n, _d), b in zip(rows, basis))
            if sb > 1e-9:
                curv = sbv / sb
                centre_raw += curv * ((0.0 - mx) ** 2 - sxx / sw)
        resid = [(abs(v - (mv + slope * (x - mx))), w, n) for v, w, x, n, _d in rows]
        if blend_med > 0:
            pairs = [(v, w, n) for v, w, _x, n, _d in rows]
            med = L.wquantile(pairs, 0.5)
            centre_raw = (1 - blend_med) * centre_raw + blend_med * med
        lo = min(v for v, _w, _x, _n, _d in rows) - 0.3
        hi = max(v for v, _w, _x, _n, _d in rows) + 0.3
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
    print("\n-- wider bandwidths --")
    for bw in [2.0, 2.5, 3.5, 1e9]:
        for lam in [0.95, 1.0]:
            run(curves, f"loclin bw={bw} lam={lam}",
                lambda tr, tf, p1, hrs, bw=bw, lam=lam: fc_loclin(tr, tf, p1, hrs, bw=bw, lam=lam), v1_pc)
    print("\n-- variants at bw=1.5/2.0 --")
    run(curves, "loclin2 bw=1.5 quad", lambda tr, tf, p1, hrs: fc_loclin2(tr, tf, p1, hrs, bw=1.5, quad=True), v1_pc)
    run(curves, "loclin2 bw=2.0 quad", lambda tr, tf, p1, hrs: fc_loclin2(tr, tf, p1, hrs, bw=2.0, quad=True), v1_pc)
    run(curves, "loclin2 bw=1.5 blend_med=0.3", lambda tr, tf, p1, hrs: fc_loclin2(tr, tf, p1, hrs, bw=1.5, blend_med=0.3), v1_pc)
    run(curves, "loclin2 bw=1.5 blend_med=0.5", lambda tr, tf, p1, hrs: fc_loclin2(tr, tf, p1, hrs, bw=1.5, blend_med=0.5), v1_pc)
    run(curves, "loclin2 bw=1.5 pre6_beta=0.15", lambda tr, tf, p1, hrs: fc_loclin2(tr, tf, p1, hrs, bw=1.5, pre6_beta=0.15), v1_pc)
    run(curves, "loclin2 bw=1.5 shrink=1.0", lambda tr, tf, p1, hrs: fc_loclin2(tr, tf, p1, hrs, bw=1.5, sigma_shrink=1.0), v1_pc)
    run(curves, "loclin2 bw=1.5 shrink=0.9", lambda tr, tf, p1, hrs: fc_loclin2(tr, tf, p1, hrs, bw=1.5, sigma_shrink=0.9), v1_pc)
    print("\n-- per-case audit: loclin bw=1.5 lam=1.0, rolling targets --")
    _r2, _a2, pc = L.evaluate(curves, lambda tr, tf, p1, hrs: fc_loclin(tr, tf, p1, hrs, bw=1.5, lam=1.0))
    for n in sorted(pc):
        print(f"  {n:38s} v1 {v1_pc[n]:.4f}   loclin {pc[n]:.4f}   diff {pc[n] - v1_pc[n]:+.4f}")
    print("\n-- all-36 worst regressions vs v1 (loclin bw=1.5 lam=1.0) --")
    _r3, _a3, pc_all = L.evaluate(curves, lambda tr, tf, p1, hrs: fc_loclin(tr, tf, p1, hrs, bw=1.5, lam=1.0), rolling_only_targets=False)
    # evaluate() only returns rolling per-case; recompute all per-case quickly
    diffs = []
    for name, d in sorted(curves.items()):
        p1 = d["cv"][1]
        answers = {h: d["cv"][h] for h in d["cv"] if h >= 2}
        hours = sorted(answers)
        train = [(n, curves[n]) for n in curves if n != name]
        s_new = L.score(fc_loclin(train, d["feat"], p1, hours, bw=1.5, lam=1.0), answers)
        s_v1 = L.score(L.fc_v1(train, d["feat"], p1, hours), answers)
        diffs.append((s_new - s_v1, name, s_v1, s_new))
    diffs.sort(reverse=True)
    for diff, name, s_v1, s_new in diffs[:6]:
        print(f"  {name:38s} v1 {s_v1:.4f}   loclin {s_new:.4f}   diff {diff:+.4f}")
    print("  ... best improvements:")
    for diff, name, s_v1, s_new in diffs[-3:]:
        print(f"  {name:38s} v1 {s_v1:.4f}   loclin {s_new:.4f}   diff {diff:+.4f}")


if __name__ == "__main__":
    main()
