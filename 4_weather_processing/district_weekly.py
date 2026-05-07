import pandas as pd
import numpy as np

# Karnataka districts with their approximate lat/lon boundaries
karnataka_districts = {
    'Bagalkot': {'lat_range': (15.2, 16.8), 'lon_range': (74.5, 75.8)},
    'Ballari': {'lat_range': (14.5, 15.9), 'lon_range': (75.5, 76.8)},
    'Belagavi': {'lat_range': (15.6, 16.9), 'lon_range': (74.2, 76.0)},
    'Bengaluru Rural': {'lat_range': (12.8, 13.6), 'lon_range': (77.2, 78.5)},
    'Bengaluru Urban': {'lat_range': (12.8, 13.3), 'lon_range': (77.3, 77.8)},
    'Bidar': {'lat_range': (17.3, 17.9), 'lon_range': (76.7, 78.0)},
    'Bijappur': {'lat_range': (16.6, 17.6), 'lon_range': (75.5, 76.8)},
    'Chikballapur': {'lat_range': (13.2, 13.9), 'lon_range': (77.3, 78.6)},
    'Chikmagalur': {'lat_range': (13.3, 14.2), 'lon_range': (75.2, 76.3)},
    'Chitradurga': {'lat_range': (13.2, 14.3), 'lon_range': (75.9, 77.5)},
    'Dakshina Kannada': {'lat_range': (12.3, 13.3), 'lon_range': (74.8, 75.8)},
    'Davanagere': {'lat_range': (14.0, 14.9), 'lon_range': (75.5, 76.5)},
    'Dharwad': {'lat_range': (15.5, 16.2), 'lon_range': (74.7, 75.6)},
    'Gadag': {'lat_range': (14.8, 15.6), 'lon_range': (74.8, 76.0)},
    'Gulbarga': {'lat_range': (17.3, 18.1), 'lon_range': (76.4, 77.5)},
    'Hassan': {'lat_range': (13.0, 13.9), 'lon_range': (75.5, 76.5)},
    'Haveri': {'lat_range': (14.3, 15.3), 'lon_range': (74.5, 75.8)},
    'Kodagu': {'lat_range': (11.9, 12.6), 'lon_range': (75.3, 76.0)},
    'Kolar': {'lat_range': (12.9, 13.6), 'lon_range': (78.0, 78.8)},
    'Kolhapur': {'lat_range': (16.2, 17.2), 'lon_range': (73.8, 74.8)},
    'Koppal': {'lat_range': (15.4, 16.2), 'lon_range': (75.8, 76.8)},
    'Mandya': {'lat_range': (12.5, 13.3), 'lon_range': (76.4, 77.3)},
    'Mysuru': {'lat_range': (11.6, 12.7), 'lon_range': (75.3, 76.8)},
    'Raichur': {'lat_range': (16.1, 17.1), 'lon_range': (76.3, 77.8)},
    'Ramanagara': {'lat_range': (12.7, 13.3), 'lon_range': (77.2, 77.8)},
    'Shimoga': {'lat_range': (13.6, 14.6), 'lon_range': (74.9, 76.2)},
    'Tumkur': {'lat_range': (13.2, 14.0), 'lon_range': (75.9, 77.0)},
    'Udupi': {'lat_range': (13.3, 13.8), 'lon_range': (74.6, 75.2)},
    'Uttara Kannada': {'lat_range': (14.6, 15.6), 'lon_range': (73.9, 75.3)},
    'Yadgir': {'lat_range': (16.7, 17.4), 'lon_range': (77.1, 78.0)},
}

def get_district(lat, lon):
    """Map lat/lon to district"""
    for district, bounds in karnataka_districts.items():
        lat_min, lat_max = bounds['lat_range']
        lon_min, lon_max = bounds['lon_range']
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return district
    return 'Unknown'

print("Loading Karnataka rainfall data...")
df = pd.read_csv('RF25_karnataka_2000_2025.csv')

# Convert TIME to datetime
df['TIME'] = pd.to_datetime(df['TIME'])

# Assign districts to each coordinate
print("Mapping coordinates to districts...")
df['DISTRICT'] = df.apply(lambda row: get_district(row['LATITUDE'], row['LONGITUDE']), axis=1)

# Count mapped coordinates
mapped = df[df['DISTRICT'] != 'Unknown'].shape[0]
total = df.shape[0]
print(f"✓ Mapped {mapped}/{total} coordinates ({100*mapped/total:.1f}%)")
print(f"\nDistricts found: {sorted(df['DISTRICT'].unique())}\n")

# Add week number
df['Year'] = df['TIME'].dt.year
df['Week'] = df['TIME'].dt.isocalendar().week
df['YearWeek'] = df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2)

# Aggregate by district and week (sum rainfall for all grid points in each district)
print("Calculating district-wise weekly cumulative rainfall...")
district_weekly = df.groupby(['DISTRICT', 'YearWeek']).agg({
    'RAINFALL': 'sum',
    'TIME': 'first',
    'Year': 'first',
    'Week': 'first'
}).reset_index()

# Sort
district_weekly = district_weekly.sort_values(['DISTRICT', 'Year', 'Week']).reset_index(drop=True)

print(f"Original records: {len(df)}")
print(f"District-weekly records: {len(district_weekly)}")
print(f"\nSample output:")
print(district_weekly.head(20))

# Save to CSV
output_file = 'RF25_karnataka_district_weekly.csv'
district_weekly.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# Also save a summary by district
print("\n" + "="*60)
print("SUMMARY BY DISTRICT")
print("="*60)
summary = df[df['DISTRICT'] != 'Unknown'].groupby('DISTRICT').agg({
    'RAINFALL': ['count', 'sum', 'mean'],
    'LATITUDE': 'count'
}).round(2)
print(summary)
