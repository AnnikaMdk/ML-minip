# ML-Minip Clean Upload

This folder contains the curated files for the final arecanut price forecasting project, focusing on data cleaning, preprocessing, and the main forecast pipeline.

## Included files

- `ML_Corn_Project/arecanut_price_forecast.py`
- `ML_Corn_Project/arecanut_weather_integrated_forecast.py`
- `ML_Corn_Project/fix_price_outliers.py`
- `ML_Corn_Project/clean_mom_outliers.py`
- `2_price_processing/assemble_arecanut_master.py`
- `2_price_processing/assemble_clean_aggregate.py`
- `3_weather_download/aggregate_weather_by_place.py`
- `4_weather_processing/district_weekly.py`
- `4_weather_processing/daily_to_weekly.py`
- `5_utilities/utils.py`

## Notes

- Raw data files, generated outputs, and `.nc` conversion utilities are intentionally excluded.
- The scripts use repository-relative paths where possible.
- The project expects input CSV files to be placed outside this folder or provided by the user.

## Requirements

Install the minimal dependencies before running the scripts:

```bash
pip install numpy pandas python-dateutil scikit-learn plotly matplotlib
```
