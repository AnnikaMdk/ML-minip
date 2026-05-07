# ...existing code...
import pandas as pd
import glob
from utils import parse_date, normalize_unit_price
import numpy as np

def read_all_sources(folder_glob="data/*.csv"):
    files = glob.glob(folder_glob)
    dfs = []
    for f in files:
        df = pd.read_csv(f, dtype=str)
        dfs.append(df)
    if not dfs:
        # return an empty DataFrame with expected columns to avoid KeyError downstream
        return pd.DataFrame(columns=[
            'date', 'min_price', 'max_price', 'modal_price', 'unit',
            'mandi', 'state', 'variety'
        ])
    df = pd.concat(dfs, ignore_index=True, sort=False)
    return df

def clean_dataframe(df):
    # Ensure date
    df = df.copy()
    if 'date' not in df.columns:
        print("Warning: 'date' column not found in input dataframe. Returning empty cleaned dataframe.")
        return pd.DataFrame(columns=['date','year','month','price_per_kg_inr'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    # numeric cleaning
    for col in ['min_price','max_price','modal_price']:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                       .str.replace(',','')
                       .str.extract(r'([0-9.]+)')[0]
                       .astype(float, errors='ignore'))
    # Normalize unit to price_per_kg_inr
    def norm_row(r):
        unit = r.get('unit', '')
        # pick modal_price > max > min
        p = r.get('modal_price', np.nan)
        if pd.isna(p):
            p = r.get('max_price', np.nan)
        if pd.isna(p):
            p = r.get('min_price', np.nan)
        if pd.isna(p):
            return np.nan
        try:
            return normalize_unit_price(float(p), unit)
        except:
            return np.nan
    df['price_per_kg_inr'] = df.apply(norm_row, axis=1)
    return df

def aggregate_monthly(df):
    # group by year, month, mandi, state, variety
    g = df.groupby(['year','month','mandi','state','variety'])
    agg = g['price_per_kg_inr'].agg(['count','mean','median','std','min','max']).reset_index()
    agg = agg.rename(columns={
        'count':'days_reported',
        'mean':'avg_price_per_kg',
        'median':'median_price_per_kg',
        'std':'std_price_per_kg',
        'min':'min_price_per_kg',
        'max':'max_price_per_kg'
    })
    return agg

if __name__ == "__main__":
    raw = read_all_sources("data/*.csv")
    print("Files read, rows:", len(raw))
    if raw.empty:
        print("No data files found. Exiting.")
        exit(0)
    cleaned = clean_dataframe(raw)
    monthly = aggregate_monthly(cleaned)
    monthly.to_csv("arecanut_monthly_aggregated.csv", index=False)
    print("Monthly aggregated written to arecanut_monthly_aggregated.csv")