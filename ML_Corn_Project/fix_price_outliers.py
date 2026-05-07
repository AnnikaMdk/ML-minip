"""
Fix week-on-week price jumps > 1000 Rs/quintal via iterative NaN-mask + linear interpolation.
Saves cleaned CSV alongside original, then updates arecanut_dashboard.py to use it.
"""
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

MARKETS = ["Mangaluru", "Puttur", "Sagar", "Shimoga", "Sirsi"]
BASE = Path(__file__).resolve().parent
SRC = BASE / "outputs/arecanut_prices_weekly_2000_2025_separate_markets.csv"
OUT = SRC.with_name("arecanut_prices_weekly_2000_2025_separate_markets_cleaned.csv")
MAX_STEP = 1000
MAX_ITER = 50


def fix_jumps(series: pd.Series, max_step: int = MAX_STEP, max_iter: int = MAX_ITER):
    """
    Iteratively mask the VIOLATING point (the second one in a pair with diff > max_step)
    and interpolate linearly until no violations remain or max_iter is reached.
    Edge NaNs are forward/back-filled.
    """
    s = series.copy().astype(float)
    n_changed = 0

    for iteration in range(max_iter):
        diff = s.diff().abs()
        bad_idx = diff[diff > max_step].index
        if len(bad_idx) == 0:
            break
        n_changed += len(bad_idx)
        s[bad_idx] = np.nan
        s = s.interpolate(method="linear", limit_direction="both")
        s = s.ffill().bfill()  # handle any leading/trailing NaNs
    else:
        # Max iterations reached – report remaining violations
        remaining = (s.diff().abs() > max_step).sum()
        print(f"  [WARNING] Max iterations reached; {remaining} violations remain "
              f"(likely genuine long-run price steps with no intermediate data to interpolate).")

    return s, n_changed


def main():
    print(f"Loading: {SRC}")
    df = pd.read_csv(SRC)
    df["date"] = pd.to_datetime(df["Week_Start_Date"], format="%d-%m-%Y", errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\nRows: {len(df)}  |  Markets: {MARKETS}")
    print(f"Fixing jumps > {MAX_STEP} Rs/quintal (max {MAX_ITER} iterations per market)\n")

    df_clean = df.copy()
    summary = {}

    for m in MARKETS:
        original = df_clean[m].copy()
        fixed, n_ops = fix_jumps(df_clean[m])
        df_clean[m] = fixed

        changed_mask = (original - fixed).abs() > 0.01
        n_rows_changed = changed_mask.sum()
        summary[m] = {
            "cells_changed": n_rows_changed,
            "ops": n_ops,
            "remaining_violations": (fixed.diff().abs() > MAX_STEP).sum(),
        }
        print(f"  {m}: {n_rows_changed} values replaced  "
              f"({n_ops} mask ops)  "
              f"|  remaining violations: {summary[m]['remaining_violations']}")

    # Drop helper column before saving
    save_cols = [c for c in df_clean.columns if c != "date"]
    df_clean[save_cols].to_csv(OUT, index=False)
    print(f"\nCleaned CSV saved to:\n  {OUT}")

    # Verify
    df_v = pd.read_csv(OUT)
    df_v["date"] = pd.to_datetime(df_v["Week_Start_Date"], format="%d-%m-%Y", errors="coerce")
    df_v = df_v.sort_values("date").reset_index(drop=True)
    print("\n=== VERIFICATION (cleaned CSV) ===")
    for m in MARKETS:
        remaining = (df_v[m].diff().abs() > MAX_STEP).sum()
        print(f"  {m}: {remaining} jumps > {MAX_STEP} remain")


if __name__ == "__main__":
    main()
