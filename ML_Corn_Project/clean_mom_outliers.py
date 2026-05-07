"""
Clean Month-over-Month (%) Change Outliers

Rule: For each month, if value differs from previous and next month by more than
a threshold, replace with average of previous and next month.

Usage:
  python clean_mom_outliers.py <input_csv> <output_csv> [--threshold 4]

Example:
  python clean_mom_outliers.py mom_pct_change.csv mom_pct_change_cleaned.csv --threshold 4
"""

import sys
import argparse
from pathlib import Path

import pandas as pd


def clean_mom_outliers(df: pd.DataFrame, threshold: float = 4.0) -> tuple:
    """
    Clean outlier months by replacing values that deviate >threshold from neighbors.

    Args:
        df: DataFrame with Year, Month, and columns ending in '_MoM_%'
        threshold: Max absolute difference allowed vs neighbors (default 4%)

    Returns:
        (cleaned_df, change_log)
    """
    df = df.copy()
    cols = [c for c in df.columns if c.endswith("_MoM_%")]
    changes = []

    # Get all unique (year, month) combinations
    months = sorted(
        [(int(y), int(m)) for y, m in df[["Year", "Month"]].drop_duplicates().to_numpy()]
    )

    for year, month in months:
        curr_idx = df.index[(df["Year"] == year) & (df["Month"] == month)]
        if len(curr_idx) == 0:
            continue
        curr_idx = curr_idx[0]

        # Determine previous and next month
        if month == 1:
            prev_year, prev_month = year - 1, 12
        else:
            prev_year, prev_month = year, month - 1

        if month == 12:
            next_year, next_month = year + 1, 1
        else:
            next_year, next_month = year, month + 1

        prev_idx = df.index[(df["Year"] == prev_year) & (df["Month"] == prev_month)]
        next_idx = df.index[(df["Year"] == next_year) & (df["Month"] == next_month)]

        if len(prev_idx) == 0 or len(next_idx) == 0:
            continue

        prev_idx = prev_idx[0]
        next_idx = next_idx[0]

        # Check each market column
        for col in cols:
            curr_val = df.at[curr_idx, col]
            prev_val = df.at[prev_idx, col]
            next_val = df.at[next_idx, col]

            if (
                pd.notna(curr_val)
                and pd.notna(prev_val)
                and pd.notna(next_val)
                and (curr_val - prev_val) > threshold
                and (curr_val - next_val) > threshold
            ):
                new_val = round((prev_val + next_val) / 2, 2)
                if abs(curr_val - new_val) > 1e-9:
                    changes.append(
                        {
                            "Year": year,
                            "Month": month,
                            "Market": col,
                            "Old_Value": curr_val,
                            "Prev": prev_val,
                            "Next": next_val,
                            "New_Value": new_val,
                        }
                    )
                    df.at[curr_idx, col] = new_val

    return df, pd.DataFrame(changes)


def main():
    parser = argparse.ArgumentParser(
        description="Clean Month-over-Month (%) Change outliers in price CSV"
    )
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_csv", help="Output CSV file path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Maximum allowed deviation from neighbors (default 4%%)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    threshold = args.threshold

    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        sys.exit(1)

    print(f"\n─ Loading {input_path.name}…")
    df = pd.read_csv(input_path)

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {[c for c in df.columns if c.endswith('_MoM_%')]}")
    print(f"  Years: {df['Year'].min():.0f}–{df['Year'].max():.0f}")

    print(f"\n─ Cleaning outliers (threshold ±{threshold}%)…")
    df_clean, changes = clean_mom_outliers(df, threshold=threshold)

    print(f"  Changes made: {len(changes)}")
    if len(changes) > 0:
        print(f"\n  Sample changes:")
        for i, row in changes.head(10).iterrows():
            print(
                f"    {int(row['Year'])}-M{int(row['Month']):02d} {row['Market']}: "
                f"{row['Old_Value']:.2f}% → {row['New_Value']:.2f}% "
                f"(neighbors: {row['Prev']:.2f}%, {row['Next']:.2f}%)"
            )

    df_clean.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned CSV saved: {output_path}")

    # Optional: save change log
    if len(changes) > 0:
        log_path = output_path.parent / (output_path.stem + "_changes.csv")
        changes.to_csv(log_path, index=False)
        print(f"✓ Change log saved: {log_path}")

    print()


if __name__ == "__main__":
    main()
