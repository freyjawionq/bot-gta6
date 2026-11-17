"""LOO round 6: verify the scale-similarity effect, then tune around it.
Same protocol/metrics as dev_loo4/5. Dev tool only."""
import math

import dev_loo4 as L
from dev_loo5 import make_model, run

Z = L.Z


def fc_topk(train, tfeat, p1, hours, k=12):
    """v1 model but the pool is the k train curves nearest in logp1."""
    near = sorted(train, key=lambda nd: abs(nd[1]["feat"]["logp1"] - tfeat["logp1"]))[:k]
    return L.fc_v1(near, tfeat, p1, hours)


def fc_kernel_mix(train, tfeat, p1, hours, bw=1.0, uniform=0.15):
    """v1 with weights base * (uniform + (1-uniform) * exp(-d^2/2bw^2))."""
    out = {}
    tw = []
    for n, d in train:
        w = 1.0 if d["rolling"] else 0.5
        dd = d["feat"]["logp1"] - tfeat["logp1"]
        w *= uniform + (1 - uniform) * math.exp(-(dd ** 2) / (2 * bw ** 2))
        tw.append((n, d, w))
    for h in hours:
        pairs = [(d["lr"][h], w, n) for n, d, w in tw if h in d["lr"]]
        if not pairs:
            out[h] = [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
            continue
        med = L.wquantile(pairs, 0.5)
        devs = [(abs(v - med), w, n) for v, w, n in pairs]
        sigma = 1.4826 * L.wquantile(devs, 0.5) * 0.95
        sigma = max(sigma, 0.05, 0.5 * 0.15 * math.sqrt(h - 1))
        out[h] = [p1 * math.exp(0.9 * med + z * sigma) for z in Z]
    return out


def main():
    curves = L.load()
    for d in curves.values():
        d["feat"]["console"] = d["console"]
    _r, _a, v1_pc = L.evaluate(curves, L.fc_v1)
    print(f"{'v1 (reference)':46s} rolling {_r:.4f}   all36 {_a:.4f}")
    print("\nlogp1 distribution of the pool:")
    for n, d in sorted(curves.items(), key=lambda kv: kv[1]["feat"]["logp1"]):
        print(f"  {n:38s} logp1 {d['feat']['logp1']:6.2f}  p1 {d['cv'][1]:>10,.0f}  rolling={d['rolling']}")
    print("\n-- top-k nearest in scale (plain v1 inside) --")
    for k in [6, 8, 10, 12, 16, 20]:
        run(curves, f"topk k={k}", lambda tr, tf, p1, hrs, k=k: fc_topk(tr, tf, p1, hrs, k), v1_pc)
    print("\n-- kernel + uniform floor --")
    for bw, un in [(0.75, 0.1), (1.0, 0.1), (1.0, 0.2), (1.0, 0.3), (1.25, 0.2), (1.5, 0.2), (0.75, 0.0)]:
        run(curves, f"kernel bw={bw} uniform={un}",
            lambda tr, tf, p1, hrs, bw=bw, un=un: fc_kernel_mix(tr, tf, p1, hrs, bw, un), v1_pc)
    print("\n-- per-case detail: bw_logp1=1.0 vs v1 (rolling targets) --")
    _r2, _a2, pc = L.evaluate(curves, make_model(bw_logp1=1.0))
    for n in sorted(pc):
        print(f"  {n:38s} v1 {v1_pc[n]:.4f}   bw1.0 {pc[n]:.4f}   diff {pc[n] - v1_pc[n]:+.4f}")


if __name__ == "__main__":
    main()
