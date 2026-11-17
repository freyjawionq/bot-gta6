"""LOO round 3: weighted-pool normal model tuning on rolling targets.
Weights: rolling curves weigh 1, others weigh w. Weighted median + weighted
sigma (std or MAD) per hour, median shrunk toward persistence by lam.
Dev tool only; not part of the entry.
"""
import itertools
import math
from pathlib import Path

import dev_loo2 as L

KIT = Path.home() / "workspace/makefun/gta6-twitch-ccu-forecast-01-kit"
LEVELS = L.LEVELS
Z = L.Z


def wquantile(pairs, tau):
    pairs = sorted(pairs)
    tot = sum(w for _v, w in pairs)
    if tot <= 0:
        return pairs[len(pairs) // 2][0]
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        acc += w
        if acc >= tau * tot:
            if i + 1 < len(pairs) and acc == tau * tot:
                return 0.5 * (v + pairs[i + 1][0])
            return v
    return pairs[-1][0]


def fc_weighted(train, p1, hours, lam, sigma_mode, shrink, floor=0.05):
    out = {}
    for h in hours:
        pairs = [(lr[h], w) for lr, w in train if h in lr]
        med = wquantile(pairs, 0.5)
        if sigma_mode == "mad":
            dev = [(abs(v - med), w) for v, w in pairs]
            sig = 1.4826 * wquantile(dev, 0.5)
        else:
            tw = sum(w for _v, w in pairs)
            mu = sum(v * w for v, w in pairs) / tw
            sig = math.sqrt(sum(w * (v - mu) ** 2 for v, w in pairs) / tw)
        sig = max(sig * shrink, floor)
        out[h] = [p1 * math.exp(lam * med + z * sig) for z in Z]
    return out


def main():
    curves = L.load(KIT / "input" / "p00.in")
    lr = {}
    for name, d in curves.items():
        base = math.log(d["cv"][1])
        lr[name] = {h: math.log(v) - base for h, v in d["cv"].items() if h >= 2}
    targets = [n for n, d in sorted(curves.items()) if d["rolling"]]
    grid = []
    for w, lam, sm, sh in itertools.product(
        [0.0, 0.25, 0.5, 1.0], [0.8, 0.9, 1.0], ["mad", "std"], [0.9, 1.0, 1.1]
    ):
        losses = []
        for name in targets:
            d = curves[name]
            p1 = d["cv"][1]
            answers = {h: d["cv"][h] for h in d["cv"] if h >= 2}
            hours = sorted(answers)
            train = [(lr[n], 1.0 if curves[n]["rolling"] else w) for n in lr if n != name]
            f = fc_weighted(train, p1, hours, lam, sm, sh)
            losses.append(L.score(f, answers))
        grid.append((sum(losses) / len(losses), w, lam, sm, sh))
    grid.sort()
    print("top 12 of", len(grid), "configs (rolling targets, n=%d):" % len(targets))
    for mean, w, lam, sm, sh in grid[:12]:
        print(f"  loss {mean:.4f}  w_nonroll={w:<5} lam={lam:<4} sigma={sm:<4} shrink={sh}")
    print("worst 3:")
    for mean, w, lam, sm, sh in grid[-3:]:
        print(f"  loss {mean:.4f}  w_nonroll={w:<5} lam={lam:<4} sigma={sm:<4} shrink={sh}")


if __name__ == "__main__":
    main()
