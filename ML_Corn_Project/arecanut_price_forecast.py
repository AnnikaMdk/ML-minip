"""
Very Simple Arecanut Price Forecast (Linear Regression)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Input and output paths
BASE = Path(__file__).resolve().parent.parent
PRICE_FILE = BASE / "ML_Corn_Project/outputs/arecanut_prices_weekly_2000_2025_separate_markets_cleaned.csv"
OUT_DIR = BASE / "ML_Corn_Project/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading weekly prices...")
# Load raw weekly market file
prices = pd.read_csv(PRICE_FILE)

# Parse date and clean
# Convert date text to datetime; invalid dates become NaT
prices["date"] = pd.to_datetime(prices["Week_Start_Date"], format="%d-%m-%Y", errors="coerce")
# Keep rows where both date and price exist
prices = prices.dropna(subset=["date", "Price_Rs_Quintal"]).copy()

# One weekly price series (average across markets)
weekly = (
    prices.groupby("date", as_index=False)["Price_Rs_Quintal"]
    .mean()
    .rename(columns={"Price_Rs_Quintal": "price"})
    .sort_values("date")
    .reset_index(drop=True)
)

# Simple features: time trend + lag prices
# t: linear time index to capture broad trend
weekly["t"] = np.arange(len(weekly))
# lag_1 and lag_2: previous 1 and 2 week prices
weekly["lag_1"] = weekly["price"].shift(1)
weekly["lag_2"] = weekly["price"].shift(2)
# Drop first two rows that cannot have lag values
weekly = weekly.dropna().reset_index(drop=True)

# X = features, y = target
X = weekly[["t", "lag_1", "lag_2"]]
y = weekly["price"]

# Chronological split (80/20)
# Use time-aware split so future data does not leak into training
split = int(len(weekly) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
dates_test = weekly["date"].iloc[split:]

# Train simple linear regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
# RMSE and MAE are in Rs/quintal; R2 is unitless fit score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nSimple Linear Regression Results")
print(f"Train samples: {len(X_train)}")
print(f"Test samples : {len(X_test)}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R2  : {r2:.4f}")

# Save predictions
# Keep one table with actual, prediction, and residual error
pred_df = pd.DataFrame(
    {
        "date": dates_test,
        "actual_price": y_test.values,
        "predicted_price": y_pred,
        "error": y_pred - y_test.values,
    }
)
pred_path = OUT_DIR / "arecanut_predictions_simple_linear.csv"
pred_df.to_csv(pred_path, index=False)
print(f"Saved: {pred_path.name}")

# Graph 1: actual vs predicted line chart
plt.figure(figsize=(12, 5))
plt.plot(dates_test, y_test.values, label="Actual", linewidth=2)
plt.plot(dates_test, y_pred, label="Predicted", linewidth=2)
plt.title("Arecanut Price Forecast (Simple Linear Regression)")
plt.xlabel("Date")
plt.ylabel("Price (Rs/quintal)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = OUT_DIR / "arecanut_forecast_simple_linear.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved: {plot_path.name}")

# Graph 2: residuals over time to spot drift/bias periods
plt.figure(figsize=(12, 4))
plt.plot(dates_test, pred_df["error"].values, linewidth=1.8, color="tab:orange")
plt.axhline(0, linestyle="--", linewidth=1.2, color="black")
plt.title("Residuals Over Time (Predicted - Actual)")
plt.xlabel("Date")
plt.ylabel("Residual (Rs/quintal)")
plt.grid(alpha=0.3)
plt.tight_layout()

residual_plot_path = OUT_DIR / "arecanut_residuals_over_time.png"
plt.savefig(residual_plot_path, dpi=150)
plt.close()
print(f"Saved: {residual_plot_path.name}")
