import pandas as pd
import numpy as np

print("Loading Karnataka daily rainfall data...")
df = pd.read_csv('RF25_karnataka_2000_2025.csv')

# Convert TIME to datetime
df['TIME'] = pd.to_datetime(df['TIME'])

# Add week number and year
df['Year'] = df['TIME'].dt.year
df['Week'] = df['TIME'].dt.isocalendar().week
df['YearWeek'] = df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2)

print(f"Date range: {df['TIME'].min()} to {df['TIME'].max()}")
print(f"Number of unique locations: {df.groupby(['LATITUDE', 'LONGITUDE']).ngroups}")

# Group by year-week and location, sum rainfall (weekly cumulative)
print("\nCalculating weekly cumulative rainfall...")
weekly_df = df.groupby(['YearWeek', 'LATITUDE', 'LONGITUDE']).agg({
    'RAINFALL': 'sum',
    'TIME': 'first'
}).reset_index()

# Rename columns
weekly_df.rename(columns={'TIME': 'WEEK_START_DATE'}, inplace=True)

# Sort by location and date
weekly_df = weekly_df.sort_values(['LATITUDE', 'LONGITUDE', 'WEEK_START_DATE']).reset_index(drop=True)

print(f"Original daily records: {len(df)}")
print(f"Weekly cumulative records: {len(weekly_df)}")
print(f"\nWeekly data sample:")
print(weekly_df.head(15))

# Save to CSV
output_file = 'RF25_karnataka_weekly_cumulative.csv'
weekly_df.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")
