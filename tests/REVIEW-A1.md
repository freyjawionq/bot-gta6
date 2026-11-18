# Review A1 — dev_e2e.py v1 reference bug (2026-11-18, Muse/Kaname, sole editor under lock)

## Finding (re-verified on the official HEAD before editing)

`dev_e2e.py` loaded "v1" with `git show HEAD:main.py`, but HEAD is v2.
The old E2E therefore compared v2 against itself and could not support
any v1-vs-v2 improvement claim.

- Baseline (HEAD): `54cd11147c70c9d05310ad5318ad9c7bb13e636b` (= origin/master at lock time)
- Real v1 commit: `5b64e7e3bb9047922aaa25cb5f69a389cb1f3aaf` (HEAD~1)
- sha256 main.py @ v1 pin: `e4e59f28d4ea5a51480bb09f8631e58fc00388d83529b5458d799b332c6c2c68`
- sha256 main.py @ HEAD / working tree: `1357a53f31a7eefe759dbd06069815afd74b000afae313ec1570d323f2899c2b`

## Fix (dev_e2e.py only; main.py and dev_loo*.py untouched)

- Added `V1_COMMIT = "5b64e7e3bb9047922aaa25cb5f69a389cb1f3aaf"` and load
  v1 via `git show f"{V1_COMMIT}:main.py"`.
- Output label corrected: `v1 (git HEAD)` -> `v1 (pin 5b64e7e)`.
- Scoring behaviour otherwise unchanged.

## Regression test — tests/test_e2e_v1_reference.py

Plain `python3`, git-identity checks only, no external KIT data needed,
reproducible from a clean checkout:

    python3 tests/test_e2e_v1_reference.py

Checks: (a) dev_e2e.py no longer uses HEAD:main.py as the v1 source and
its V1_COMMIT pin is a 40-hex commit that exists in history and is an
ancestor of HEAD; (b) pinned main.py hash differs from HEAD main.py hash;
(c) pinned main.py sha256 matches the recorded v1 hash above.

### Baseline vs candidate (same test, exact outputs)

Baseline — clean clone at 54cd111 + the test file (also reproduced
against the unfixed working tree before editing):

    FAIL  a1: dev_e2e.py does not use HEAD:main.py as v1 source
    FAIL  a2: V1_COMMIT constant is a full 40-hex commit id
    RESULT: FAIL (2 check(s) failed)   exit=1

Candidate — fixed working tree:

    PASS  a1: dev_e2e.py does not use HEAD:main.py as v1 source
    PASS  a2: V1_COMMIT constant is a full 40-hex commit id
    PASS  a2: pinned commit exists in history  (5b64e7e3bb9047922aaa25cb5f69a389cb1f3aaf)
    PASS  a2: pinned commit is an ancestor of HEAD
    PASS  b: pinned main.py hash differs from HEAD main.py hash  (pin=e4e59f28d4ea head=1357a53f31a7)
    PASS  c: pinned main.py sha256 matches recorded v1 hash  (e4e59f28d4ea5a51480bb09f8631e58fc00388d83529b5458d799b332c6c2c68)
    RESULT: PASS (all checks)   exit=0

## Full E2E run (local kit present: gta6-twitch-ccu-forecast-01-kit/input/p00.in)

Command: `~/workspace/nonce/venv/bin/python dev_e2e.py`

Baseline (unfixed):

    v1 (git HEAD)        e2e rolling 0.2400   all36 0.4124
    v2 (working tree)    e2e rolling 0.2400   all36 0.4124

(identical numbers — the self-comparison bug, live.)

Candidate (fixed):

    v1 (pin 5b64e7e)     e2e rolling 0.3149   all36 0.7452
    v2 (working tree)    e2e rolling 0.2400   all36 0.4124

## Limitations — read before quoting any number

- These E2E/LOO numbers are scores on the TUNING data (the same
  rolling-midnight targets the model variants were tuned on). They are
  NOT an independent / held-out estimate; the size of the tuning bias
  is unknown. This fix only makes the v1-vs-v2 comparison refer to the
  real v1 — it does not by itself prove a generalisation improvement.
- No official Make.fun score is claimed or promised; Make.fun was not
  accessed for this task.
- The E2E run depends on the external local kit (p00.in); the identity
  regression test above does not.
