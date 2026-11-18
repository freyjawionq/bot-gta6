#!/usr/bin/env python3
"""Regression test for finding A1: dev_e2e.py must load v1 from an
explicit pinned commit, not from HEAD (HEAD is v2, so HEAD:main.py made
the old E2E compare v2 against itself).

Git-identity checks only — no external KIT data needed, reproducible
from a clean checkout. Run: python3 tests/test_e2e_v1_reference.py
Exit 0 = PASS, 1 = FAIL.
"""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPECTED_V1_MAIN_SHA256 = (
    "e4e59f28d4ea5a51480bb09f8631e58fc00388d83529b5458d799b332c6c2c68"
)

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def git(*args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, check=True,
    ).stdout


def main():
    src = (REPO / "dev_e2e.py").read_text()

    # (a1) v1 must not be sourced from HEAD.
    check("a1: dev_e2e.py does not use HEAD:main.py as v1 source",
          "HEAD:main.py" not in src)

    # (a2) an explicit V1_COMMIT pin: 40-hex, exists, ancestor of HEAD.
    m = re.search(r'V1_COMMIT\s*=\s*["\']([0-9a-f]{40})["\']', src)
    check("a2: V1_COMMIT constant is a full 40-hex commit id", m is not None)
    if not m:
        print(f"\nRESULT: FAIL ({len(failures)} check(s) failed)")
        return 1
    pin = m.group(1)
    try:
        obj_type = git("cat-file", "-t", pin).decode().strip()
        exists = obj_type == "commit"
    except subprocess.CalledProcessError:
        exists = False
    check("a2: pinned commit exists in history", exists, pin)
    anc = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", pin, "HEAD"]
    )
    check("a2: pinned commit is an ancestor of HEAD", anc.returncode == 0)

    # (b) pinned main.py differs from HEAD main.py.
    pin_src = git("show", f"{pin}:main.py")
    head_src = git("show", "HEAD:main.py")
    pin_hash = hashlib.sha256(pin_src).hexdigest()
    head_hash = hashlib.sha256(head_src).hexdigest()
    check("b: pinned main.py hash differs from HEAD main.py hash",
          pin_hash != head_hash, f"pin={pin_hash[:12]} head={head_hash[:12]}")

    # (c) pinned main.py hash matches the recorded v1 hash.
    check("c: pinned main.py sha256 matches recorded v1 hash",
          pin_hash == EXPECTED_V1_MAIN_SHA256, pin_hash)

    if failures:
        print(f"\nRESULT: FAIL ({len(failures)} check(s) failed)")
        return 1
    print("\nRESULT: PASS (all checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
