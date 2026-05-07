"""
Aggregate ERA5 weather data grid points by Dakshina Kannada and Udupi places.
"""

import pandas as pd
from pathlib import Path

# Major places in Dakshina Kannada and Udupi with lat/lon
PLACES = {
    "Mangaluru": (12.87, 74.87),
    "Puttur": (12.76, 75.27),
    "Bantwal": (12.84, 75.10),
    "Sullia": (12.58, 75.53),
    "Udupi": (13.34, 74.75),
    "Kundapura": (13.71, 74.41),
    "Karkala": (13.13, 75.08),
}

BASE = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE / "6_data_outputs/era5_dakshina_udupi_weather/era5_dakshina_udupi_weather_2000_2025_weekly.csv"
OUTPUT_CSV = BASE / "6_data_outputs/era5_dakshina_udupi_weather/era5_dakshina_udupi_weather_2000_2025_weekly_by_place.csv"


def nearest_place(lat: float, lon: float) -> str:
    """Find nearest place to a grid point."""
    min_dist = float('inf')
    nearest = "Unknown"
    for place, (place_lat, place_lon) in PLACES.items():
        dist = ((lat - place_lat) ** 2 + (lon - place_lon) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest = place
    return nearest


def aggregate_by_place(df: pd.DataFrame) -> pd.DataFrame:
    """Assign grid points to nearest place and aggregate weekly weather."""
    # Assign each grid point to nearest place
    df["Place"] = df.apply(lambda row: nearest_place(row['latitude'], row['longitude']), axis=1)
    
    # Aggregate by place, year, week
    agg_dict = {
        'latitude': 'mean',
        'longitude': 'mean'
    }
    
    # Add weather variables if they exist
    for col in ['PRECIP_MM', 'DEW_POINT_C', 'SOLAR_RAD_J', 'WIND_SPEED_MS']:
        if col in df.columns:
            agg_dict[col] = 'mean'
    
    agg_df = df.groupby(['Place', 'YearWeek', 'Year', 'Week', 'WEEK_START_DATE']).agg(agg_dict).reset_index()
    
    agg_df = agg_df.sort_values(['Place', 'Year', 'Week']).reset_index(drop=True)
    
    return agg_df


def main():
    if not INPUT_CSV.exists():
        print(f"✗ Input file not found: {INPUT_CSV}")
        print("  Wait for weather download to complete first")
        return
    
    print("Loading combined weekly weather file...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df)} records, {df.groupby(['latitude', 'longitude']).ngroups} grid points")
    
    print("\nAggregating by nearest place...")
    agg_df = aggregate_by_place(df)
    
    print(f"  Aggregated to {len(agg_df)} records across {agg_df['Place'].nunique()} places")
    
    print("\nPlace summary:")
    for place in sorted(agg_df['Place'].unique()):
        subset = agg_df[agg_df['Place'] == place]
        count = len(subset)
        summary = f"  {place:15} {count:5} weeks"
        
        for col in ['PRECIP_MM', 'WIND_SPEED_MS', 'DEW_POINT_C', 'SOLAR_RAD_J']:
            if col in subset.columns:
                val_min = subset[col].min()
                val_max = subset[col].max()
                val_mean = subset[col].mean()
                summary += f" | {col}={val_min:.1f}–{val_max:.1f} (mean {val_mean:.1f})"
        
        print(summary)
    
    print(f"\nSaving to {OUTPUT_CSV.name}...")
    agg_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Saved {len(agg_df)} records")
    
    print("\nColumns:", list(agg_df.columns))
    print("\nFirst 10 rows:")
    print(agg_df.head(10))


if __name__ == "__main__":
    main()
