"""
assemble_arecanut_master.py

Discover arecanut monthly CSVs, normalize prices to INR per-kg (convert per-quintal -> per-kg),
aggregate by mandi and produce a national median series for a given year range (default 2000-2025).

Usage:
    python d:\ML minip\assemble_arecanut_master.py --input-dir "d:\ML minip" --verbose
"""
import os
import glob
import re
import argparse
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from dateutil import parser
import datetime


DEFAULT_GLOBS = ["arecanut_*monthly*.csv", "arecanut_*_monthly_*.csv"]
DEFAULT_START = 2000
DEFAULT_END = 2025


def discover_files(input_dir: str, patterns: List[str]) -> List[str]:
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(input_dir, p)))
        files.extend(glob.glob(p))  # also allow relative globs
    # preserve order and uniqueness
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def infer_mandi_from_filename(fname: str) -> Optional[str]:
    n = os.path.basename(fname).lower()
    # if exact names present
    if "puttur" in n:
        return "Puttur"
    if "kozhikode" in n or "kozhikkode" in n or "calicut" in n:
        return "Kozhikode"
    m = re.search(r"arecanut[_-]([^_/-]+)", n)
    if m:
        return m.group(1).replace("_", " ").title()
    return None


def pick_price_col(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    # Return (colname, unit_hint) where unit_hint is 'kg'|'quintal'|None
    candidates = []
    cols_lower = {c: c.lower().replace(" ", "") for c in df.columns}
    # first look for explicit columns by name
    for c, lc in cols_lower.items():
        if any(k in lc for k in ["perkg", "/kg", "kg", "rs/kg", "inr/kg"]):
            candidates.append((c, "kg"))
        elif any(k in lc for k in ["quintal", "qtl", "qtl.", "/qtl", "perq", "rs/qtl"]):
            candidates.append((c, "quintal"))
        elif any(k in lc for k in ["price", "modal", "avg", "value"]):
            candidates.append((c, None))
    # rank by numeric density
    ranked = []
    for c, hint in candidates:
        if c in df.columns:
            nonnull = df[c].notna().sum()
            ranked.append((nonnull, c, hint))
    if ranked:
        ranked.sort(reverse=True)
        return ranked[0][1], ranked[0][2]
    # fallback: numeric column with most non-nulls
    numeric_cols = [(df[c].notna().sum(), c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        numeric_cols.sort(reverse=True)
        return numeric_cols[0][1], None
    return None, None


def read_and_normalize(path: str, verbose: bool = False) -> pd.DataFrame:
    # Read CSV, try to find year/month and price, output standardized columns
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        # try with different engine
        df = pd.read_csv(path, dtype=str, engine="python", error_bad_lines=False)

    df = df.rename(columns=lambda c: str(c).strip())
    # attempt to parse date/year/month
    if "year" not in df.columns or "month" not in df.columns:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
        else:
            # try parse first column as date
            first_col = df.columns[0]
            try:
                df["__parsed_date"] = df[first_col].apply(
                    lambda x: parser.parse(str(x), fuzzy=True) if pd.notna(x) else pd.NaT
                )
                df["year"] = df["__parsed_date"].dt.year
                df["month"] = df["__parsed_date"].dt.month
            except Exception:
                pass

    # coerce numeric-looking columns
    for c in df.columns:
        try:
            sample = df[c].dropna().astype(str).head(10).tolist()
            if any(re.search(r"\d", s) for s in sample):
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.replace(" ", ""), errors="coerce")
        except Exception:
            pass

    # pick price column
    price_col, unit_hint = pick_price_col(df)
    price_per_kg = None
    if price_col:
        # convert depending on hint or median heuristic
        if unit_hint == "quintal":
            df["price_per_kg"] = df[price_col] / 100.0
            if verbose:
                print(f"[info] {os.path.basename(path)}: using '{price_col}' as per-quintal -> converted to per-kg.")
        elif unit_hint == "kg":
            df["price_per_kg"] = df[price_col]
            if verbose:
                print(f"[info] {os.path.basename(path)}: using '{price_col}' as per-kg.")
        else:
            med = float(df[price_col].dropna().median()) if df[price_col].notna().any() else 0.0
            if med > 100:
                df["price_per_kg"] = df[price_col] / 100.0
                if verbose:
                    print(f"[info] {os.path.basename(path)}: '{price_col}' med {med:.1f} -> treated as per-quintal.")
            else:
                df["price_per_kg"] = df[price_col]
                if verbose:
                    print(f"[info] {os.path.basename(path)}: '{price_col}' med {med:.1f} -> treated as per-kg.")
        price_per_kg = "price_per_kg"
    else:
        # explicit fallback names
        for k in ["price_quintal", "price_quintal_inr", "price_quintal_rs", "price"]:
            if k in df.columns:
                df[k] = pd.to_numeric(df[k], errors="coerce")
                df["price_per_kg"] = df[k] / 100.0
                price_per_kg = "price_per_kg"
                if verbose:
                    print(f"[info] {os.path.basename(path)}: fallback used column '{k}' as per-quintal -> per-kg.")
                break
    if price_per_kg is None:
        df["price_per_kg"] = np.nan
        if verbose:
            print(f"[warn] {os.path.basename(path)}: no price column detected - price_per_kg left NaN.")

    # infer mandi
    if "mandi" not in df.columns or df["mandi"].isnull().all():
        df["mandi"] = infer_mandi_from_filename(path)

    # keep only standardized columns
    # ensure ints for year/month where possible
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    out = df[["year", "month", "mandi", "price_per_kg"]].copy()
    # report
    if verbose:
        total = len(out)
        have = out["price_per_kg"].notna().sum()
        print(f"[debug] {os.path.basename(path)} -> rows: {total}, prices present: {have}")
    return out


def aggregate_by_mandi(df: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    # drop incomplete rows
    df = df.dropna(subset=["year", "month", "price_per_kg"])
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]
    # median price per mandi-month
    agg = df.groupby(["mandi", "year", "month"], as_index=False).price_per_kg.median()
    # ensure every mandi x month in grid
    mandis = sorted(agg["mandi"].dropna().unique().tolist())
    grid = []
    for m in mandis:
        for y in range(year_min, year_max + 1):
            for mo in range(1, 13):
                grid.append((m, y, mo))
    grid_df = pd.DataFrame(grid, columns=["mandi", "year", "month"])
    merged = pd.merge(grid_df, agg, on=["mandi", "year", "month"], how="left")
    merged = merged.sort_values(["mandi", "year", "month"]).reset_index(drop=True)
    return merged


def national_series(by_mandi_df: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    # median across mandis for each year-month
    nat = by_mandi_df.groupby(["year", "month"], as_index=False).price_per_kg.median()
    # ensure full grid
    grid = []
    for y in range(year_min, year_max + 1):
        for mo in range(1, 13):
            grid.append((y, mo))
    grid_df = pd.DataFrame(grid, columns=["year", "month"])
    nat = pd.merge(grid_df, nat, on=["year", "month"], how="left")
    nat = nat.sort_values(["year", "month"]).reset_index(drop=True)
    return nat


def safe_write_csv(df: pd.DataFrame, path: str, index: bool = False):
    try:
        df.to_csv(path, index=index)
        print(f"Wrote {path} rows: {len(df)}")
        return path
    except PermissionError as e:
        base, ext = os.path.splitext(path)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = f"{base}_{ts}{ext}"
        try:
            df.to_csv(alt, index=index)
            print(f"[warning] Could not write to {path} (PermissionError). Wrote instead to {alt}")
            return alt
        except Exception as e2:
            print(f"[error] Failed to write {path} and fallback {alt}: {e2}")
            raise


def main():
    parser_arg = argparse.ArgumentParser(description="Assemble arecanut monthly per-kg series")
    parser_arg.add_argument("--input-dir", default=".", help="Directory to search for CSVs (default cwd)")
    parser_arg.add_argument("--patterns", nargs="+", default=DEFAULT_GLOBS, help="Filename globs to find CSVs")
    parser_arg.add_argument("--start", type=int, default=DEFAULT_START, help="Start year (inclusive)")
    parser_arg.add_argument("--end", type=int, default=DEFAULT_END, help="End year (inclusive)")
    parser_arg.add_argument("--out-mandi", default="arecanut_monthly_perkg_2000_2025_by_mandi.csv")
    parser_arg.add_argument("--out-national", default="arecanut_monthly_perkg_2000_2025_national.csv")
    parser_arg.add_argument("--verbose", action="store_true")
    args = parser_arg.parse_args()

    files = discover_files(args.input_dir, args.patterns)
    if not files:
        print("No input CSVs found. Run fetch scripts to generate arecanut_*monthly*.csv files.")
        raise SystemExit(1)
    if args.verbose:
        print(f"Found {len(files)} files. Processing...")

    dfs = []
    for f in files:
        try:
            d = read_and_normalize(f, verbose=args.verbose)
            if not d.empty:
                d["source_file"] = os.path.basename(f)
                dfs.append(d)
        except Exception as e:
            print(f"[error] Failed reading {f}: {e}")

    if not dfs:
        print("No valid rows extracted from input files. Exiting.")
        raise SystemExit(1)

    all_df = pd.concat(dfs, ignore_index=True, sort=False)
    by_mandi = aggregate_by_mandi(all_df, args.start, args.end)
    safe_write_csv(by_mandi, args.out_mandi, index=False)

    nat = national_series(by_mandi, args.start, args.end)
    safe_write_csv(nat, args.out_national, index=False)


if __name__ == "__main__":
    main()