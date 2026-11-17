"""LOO round 5: candidate families around v1, one knob at a time, then
combinations. Same protocol/metrics as dev_loo4. Dev tool only."""
import itertools
import math

import dev_loo4 as L

Z = L.Z


def smooth(vals, half):
    """vals: {h: x} for h in 2..24 (some may miss). Triangular-kernel smooth."""
    if half <= 0:
        return dict(vals)
    out = {}
    for h, _x in vals.items():
        num = den = 0.0
        for hh, xx in vals.items():
            d = abs(hh - h)
            if d <= half:
                w = 1.0 - d / (half + 1.0)
                num += w * xx
                den += w
        out[h] = num / den if den else vals[h]
    return out


def make_model(
    lam0=0.9, lam1=0.0,                 # centre = (lam0 + lam1*(h-2)/22) * med
    sigma_mode="mad",                   # mad | iqr | blend (mad+std)/2 | trimstd
    sigma_shrink=0.95,
    tail_mult=1.0, shoulder_mult=1.0,   # per-level sigma multipliers
    smooth_med=0, smooth_sig=0,         # kernel half-widths across hours
    trim_k=0.0,                         # drop |lr-med| > trim_k * sigma_raw (0=off)
    bw_logp1=0.0,                       # similarity kernel on ln hour-1 (0=off)
    bw_pre24=0.0,                       # similarity kernel on pre24 feature
    console_mult=1.0,                   # extra weight when platform class matches
    feat=None, beta=0.0, beta_grow=0.0, # centre += beta*(f_t - f_train_mean) * (1+beta_grow*(h-2)/22)
    emp_blend=0.0,                      # blend with empirical weighted quantiles
):
    def model(train, tfeat, p1, hours):
        # per-target weights
        tw = []
        for n, d in train:
            w = 1.0 if d["rolling"] else 0.5
            if bw_logp1 > 0:
                w *= math.exp(-((d["feat"]["logp1"] - tfeat["logp1"]) ** 2) / (2 * bw_logp1 ** 2))
            if bw_pre24 > 0 and d["feat"]["pre24"] is not None and tfeat["pre24"] is not None:
                w *= math.exp(-((d["feat"]["pre24"] - tfeat["pre24"]) ** 2) / (2 * bw_pre24 ** 2))
            if console_mult != 1.0 and d["feat"].get("console") == tfeat.get("console"):
                w *= console_mult
            tw.append((n, d, w))
        # feature centring reference: weighted mean of the feature over train
        fmean = None
        if feat and tfeat.get(feat) is not None:
            num = den = 0.0
            for _n, d, w in tw:
                fv = d["feat"].get(feat)
                if fv is not None:
                    num += w * fv
                    den += w
            if den > 0:
                fmean = num / den
        meds, sigs, emps = {}, {}, {}
        for h in hours:
            pairs = [(d["lr"][h], w, n) for n, d, w in tw if h in d["lr"]]
            if not pairs:
                continue
            med = L.wquantile(pairs, 0.5)
            if trim_k > 0 and len(pairs) >= 8:
                dev0 = [(abs(v - med), w, n) for v, w, n in pairs]
                s0 = 1.4826 * L.wquantile(dev0, 0.5)
                if s0 > 0:
                    kept = [(v, w, n) for v, w, n in pairs if abs(v - med) <= trim_k * s0]
                    if len(kept) >= 5:
                        pairs = kept
                        med = L.wquantile(pairs, 0.5)
            if sigma_mode == "mad":
                devs = [(abs(v - med), w, n) for v, w, n in pairs]
                sig = 1.4826 * L.wquantile(devs, 0.5)
            elif sigma_mode == "iqr":
                sig = (L.wquantile(pairs, 0.75) - L.wquantile(pairs, 0.25)) / 1.3489795003921634
            else:
                tw_ = sum(w for _v, w, _n in pairs)
                mu = sum(v * w for v, w, _n in pairs) / tw_
                std = math.sqrt(sum(w * (v - mu) ** 2 for v, w, _n in pairs) / tw_)
                if sigma_mode == "trimstd":
                    sig = std
                else:  # blend
                    devs = [(abs(v - med), w, n) for v, w, n in pairs]
                    sig = 0.5 * (1.4826 * L.wquantile(devs, 0.5) + std)
            meds[h] = med
            sigs[h] = max(sig * sigma_shrink, 0.05, 0.5 * 0.15 * math.sqrt(h - 1))
            if emp_blend > 0:
                emps[h] = [L.wquantile(pairs, t) for t in L.LEVELS]
        meds = smooth(meds, smooth_med)
        sigs = {h: math.exp(x) for h, x in smooth({h: math.log(s) for h, s in sigs.items()}, smooth_sig).items()}
        out = {}
        for h in hours:
            if h not in meds:
                out[h] = [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
                continue
            lam = lam0 + lam1 * (h - 2) / 22.0
            centre = lam * meds[h]
            if fmean is not None:
                centre += beta * (tfeat[feat] - fmean) * (1.0 + beta_grow * (h - 2) / 22.0)
            sig = sigs[h]
            mults = [tail_mult, shoulder_mult, 1.0, shoulder_mult, tail_mult]
            qs = [centre + z * sig * m for z, m in zip(Z, mults)]
            if emp_blend > 0 and h in emps:
                eq = [lam * meds[h] + (e - meds[h]) for e in emps[h]]  # empirical deviations around shrunk centre
                qs = [(1 - emp_blend) * q + emp_blend * e for q, e in zip(qs, eq)]
            out[h] = [p1 * math.exp(q) for q in qs]
        return out

    return model


def run(curves, label, model, v1_pc=None):
    r, a, pc = L.evaluate(curves, model)
    extra = ""
    if v1_pc is not None:
        wins = sum(1 for k in pc if pc[k] < v1_pc[k] - 1e-9)
        ties = sum(1 for k in pc if abs(pc[k] - v1_pc[k]) <= 1e-9)
        extra = f"  vs v1 per-case: {wins}W/{ties}T/{len(pc) - wins - ties}L"
    print(f"{label:46s} rolling {r:.4f}   all36 {a:.4f}{extra}")
    return r, a, pc


def main():
    curves = L.load()
    for d in curves.values():
        d["feat"]["console"] = d["console"]
    _r, _a, v1_pc = L.evaluate(curves, L.fc_v1)
    print(f"{'v1 (reference)':46s} rolling {_r:.4f}   all36 {_a:.4f}")
    print("\n-- A. smoothing across hours --")
    for sm, ss in [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1), (2, 2), (3, 2)]:
        run(curves, f"smooth med={sm} sig={ss}", make_model(smooth_med=sm, smooth_sig=ss), v1_pc)
    print("\n-- B. sigma estimators --")
    for mode in ["iqr", "trimstd", "blend"]:
        run(curves, f"sigma={mode}", make_model(sigma_mode=mode), v1_pc)
    for sh in [0.85, 0.9, 1.0, 1.05]:
        run(curves, f"sigma_shrink={sh}", make_model(sigma_shrink=sh), v1_pc)
    print("\n-- C. per-level sigma multipliers --")
    for tm, shm in [(1.1, 1.0), (1.2, 1.0), (1.15, 1.05), (0.9, 1.0), (1.1, 1.1), (1.25, 1.1)]:
        run(curves, f"tail={tm} shoulder={shm}", make_model(tail_mult=tm, shoulder_mult=shm), v1_pc)
    print("\n-- D. per-hour lambda --")
    for l0, l1 in [(0.9, -0.2), (0.9, -0.1), (0.9, 0.1), (0.9, 0.2), (0.85, 0.2), (0.95, -0.2), (1.0, -0.3)]:
        run(curves, f"lam0={l0} lam1={l1}", make_model(lam0=l0, lam1=l1), v1_pc)
    print("\n-- E. similarity weighting --")
    for bw in [0.75, 1.0, 1.5, 2.0]:
        run(curves, f"bw_logp1={bw}", make_model(bw_logp1=bw), v1_pc)
    for bw in [0.3, 0.5, 0.8]:
        run(curves, f"bw_pre24={bw}", make_model(bw_pre24=bw), v1_pc)
    for cm in [1.5, 2.0, 0.67]:
        run(curves, f"console_mult={cm}", make_model(console_mult=cm), v1_pc)
    print("\n-- F. outlier trimming --")
    for k in [2.0, 2.5, 3.0]:
        run(curves, f"trim_k={k}", make_model(trim_k=k), v1_pc)
    print("\n-- G. empirical blend --")
    for eb in [0.25, 0.5, 0.75]:
        run(curves, f"emp_blend={eb}", make_model(emp_blend=eb), v1_pc)
    print("\n-- H. feature-conditioned centre --")
    for feat in ["pre24", "pre6", "slope12", "logvpc1"]:
        for beta in [-0.3, -0.15, 0.15, 0.3]:
            run(curves, f"feat={feat} beta={beta}", make_model(feat=feat, beta=beta), v1_pc)


if __name__ == "__main__":
    main()
