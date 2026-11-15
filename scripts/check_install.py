"""Check that the contestant setup works. Run by install.ps1 / install.sh.

Plays one full game between two random agents through the same per-seat view
the tournament server uses, then reports which optional packages are present.
Exits non-zero if anything required is missing or broken.
"""
import random
import sys
import time


def val(x):
    """Some accessors are properties and some are methods; accept either."""
    return x() if callable(x) else x


def main():
    try:
        import cr_sim
    except ImportError as e:
        print(f"  cr_sim could not be imported: {e}")
        print("  The simulator is out of date or missing. Rebuild it by running the install script again.")
        return 1
    print(f"  cr_sim {getattr(cr_sim, '__version__', '?')} imported")

    rng = random.Random(0)
    b = cr_sim.Battle()
    b.reset(1, "default", "default")
    views = [b.view(0), b.view(1)]

    def act(obs):
        ok = [i for i, legal in enumerate(obs.get("legal", [])) if legal]
        if not ok or rng.random() < 0.9:
            return None
        return (rng.choice(ok), rng.uniform(2, 16), rng.uniform(9, 14))  # own (bottom) side

    t0 = time.time()
    while not val(b.done):
        b.step(act(views[0].observe()), act(views[1].observe()))
    secs = time.time() - t0
    print(f"  played a full game in {secs:.2f}s: seat 0 {val(views[0].result)}, crowns {val(b.crowns)}")

    for name in ("numpy", "websockets"):
        try:
            __import__(name)
            print(f"  {name} ok")
        except ImportError:
            print(f"  {name} is missing (required)")
            return 1
    try:
        import torch
        print(f"  torch {torch.__version__} ok")
    except ImportError:
        print("  torch not installed (optional; re-run the installer with -WithTorch / --torch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
