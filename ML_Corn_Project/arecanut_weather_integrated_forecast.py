"""
Weather-integrated arecanut forecasting with date-wise prediction.

What this script does:
1. Loads weekly arecanut prices and weekly weather (temp/rain).
2. Builds one merged weekly table and weather-aware features.
3. Trains a chronological weather-aware linear model and saves metrics/predictions.
4. Exports an HTML chart with a confusion matrix.
5. Supports date-driven prediction mode where you enter a target date (for example in 2026),
   and the script predicts that week's price and shows accuracy if actual price is available.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score


BASE = Path(__file__).resolve().parent.parent
PRICE_FILE = BASE / "ML_Corn_Project/outputs/arecanut_prices_weekly_2000_2025_separate_markets_cleaned.csv"
TEMP_FILE = BASE / "6_data_outputs/era5_dakshina_udupi_temp/era5_dakshina_udupi_2000_2025_weekly_temp_by_place.csv"
RAIN_FILE = BASE / "6_data_outputs/rf25_dakshina_udupi_rainfall/RF25_dakshina_udupi_2000_2025_weekly_rainfall.csv"

OUT_DIR = BASE / "ML_Corn_Project/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INT_DIR = OUT_DIR / "interactive"
INT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_OUT = OUT_DIR / "arecanut_weekly_weather_merged.csv"
METRICS_OUT = OUT_DIR / "arecanut_weather_model_metrics.csv"
PRED_OUT = OUT_DIR / "arecanut_weather_model_predictions.csv"
DATE_PRED_OUT = OUT_DIR / "arecanut_datewise_predictions.csv"
HTML_OUT = INT_DIR / "arecanut_weather_integrated_forecast.html"

MARKETS = ["Mangaluru", "Puttur", "Sagar", "Shimoga", "Sirsi"]
FEATURE_COLS = ["t", "lag_1", "lag_2", "temp_mean", "rain_mean", "temp_std", "rain_std"]


def load_price_weekly() -> pd.DataFrame:
    price = pd.read_csv(PRICE_FILE)
    price["date"] = pd.to_datetime(price["Week_Start_Date"], format="%d-%m-%Y", errors="coerce")
    price = price.dropna(subset=["date"]).copy()

    for market in MARKETS:
        price[market] = pd.to_numeric(price[market], errors="coerce")

    price["avg_price"] = price[MARKETS].mean(axis=1)

    iso = price["date"].dt.isocalendar()
    price["iso_year"] = iso.year.astype(int)
    price["iso_week"] = iso.week.astype(int)

    weekly = (
        price.groupby(["iso_year", "iso_week"], as_index=False)
        .agg(date=("date", "min"), avg_price=("avg_price", "mean"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    return weekly


def load_temp_weekly() -> pd.DataFrame:
    temp = pd.read_csv(TEMP_FILE)
    temp["Year"] = pd.to_numeric(temp["Year"], errors="coerce")
    temp["Week"] = pd.to_numeric(temp["Week"], errors="coerce")
    temp["TEMP_C"] = pd.to_numeric(temp["TEMP_C"], errors="coerce")
    temp = temp.dropna(subset=["Year", "Week", "TEMP_C"]).copy()

    wk = (
        temp.groupby(["Year", "Week"], as_index=False)
        .agg(temp_mean=("TEMP_C", "mean"), temp_std=("TEMP_C", "std"))
        .rename(columns={"Year": "iso_year", "Week": "iso_week"})
    )
    wk["temp_std"] = wk["temp_std"].fillna(0.0)
    wk["iso_year"] = wk["iso_year"].astype(int)
    wk["iso_week"] = wk["iso_week"].astype(int)
    return wk


def load_rain_weekly() -> pd.DataFrame:
    rain = pd.read_csv(RAIN_FILE)
    rain["Year"] = pd.to_numeric(rain["Year"], errors="coerce")
    rain["Week"] = pd.to_numeric(rain["Week"], errors="coerce")
    rain["RAINFALL_MM"] = pd.to_numeric(rain["RAINFALL_MM"], errors="coerce")
    rain = rain.dropna(subset=["Year", "Week", "RAINFALL_MM"]).copy()

    wk = (
        rain.groupby(["Year", "Week"], as_index=False)
        .agg(rain_mean=("RAINFALL_MM", "mean"), rain_std=("RAINFALL_MM", "std"))
        .rename(columns={"Year": "iso_year", "Week": "iso_week"})
    )
    wk["rain_std"] = wk["rain_std"].fillna(0.0)
    wk["iso_year"] = wk["iso_year"].astype(int)
    wk["iso_week"] = wk["iso_week"].astype(int)
    return wk


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["t"] = np.arange(len(df), dtype=float)
    df["lag_1"] = df["avg_price"].shift(1)
    df["lag_2"] = df["avg_price"].shift(2)
    return df


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray, prev_actual: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    true_dir = (y_true - prev_actual > 0).astype(int)
    pred_dir = (y_pred - prev_actual > 0).astype(int)
    dir_acc = float(accuracy_score(true_dir, pred_dir))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Direction_Accuracy": dir_acc,
    }


def train_holdout_model(
    merged: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LinearRegression]:
    model_df = merged.dropna(subset=FEATURE_COLS + ["avg_price"]).reset_index(drop=True)

    split = int(len(model_df) * 0.8)
    split = max(split, 10)
    split = min(split, len(model_df) - 2)

    train = model_df.iloc[:split].copy()
    test = model_df.iloc[split:].copy()

    y_train = train["avg_price"].values
    y_test = test["avg_price"].values

    model = LinearRegression()
    model.fit(train[FEATURE_COLS].values, y_train)
    pred_test = model.predict(test[FEATURE_COLS].values)

    prev_actual = test["lag_1"].values
    model_metrics = eval_regression(y_test, pred_test, prev_actual)

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Weather_Integrated_Linear",
                "Train_Rows": len(train),
                "Test_Rows": len(test),
                "RMSE": round(model_metrics["RMSE"], 4),
                "MAE": round(model_metrics["MAE"], 4),
                "R2": round(model_metrics["R2"], 6),
                "Direction_Accuracy_Percent": round(model_metrics["Direction_Accuracy"] * 100, 2),
            }
        ]
    )

    pred_df = test[
        ["date", "iso_year", "iso_week", "avg_price", "lag_1", "temp_mean", "rain_mean"]
    ].copy()
    pred_df = pred_df.rename(columns={"avg_price": "actual_price"})
    pred_df["predicted_price"] = pred_test
    pred_df["prediction_error"] = pred_df["predicted_price"] - pred_df["actual_price"]
    pred_df["actual_direction"] = (pred_df["actual_price"] - pred_df["lag_1"] > 0).astype(int)
    pred_df["predicted_direction"] = (pred_df["predicted_price"] - pred_df["lag_1"] > 0).astype(int)

    return model_df, metrics_df, pred_df, model


def build_html(pred_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{}], [{}]],
        row_heights=[0.7, 0.3],
        subplot_titles=(
            "Weekly Weather-Integrated Price Forecast",
            "Weather Model Direction Confusion Matrix",
        ),
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=pred_df["actual_price"],
            mode="lines",
            name="Actual Price",
            line=dict(color="#111827", width=2.4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=pred_df["predicted_price"],
            mode="lines",
            name="Predicted Price",
            line=dict(color="#ef4444", width=2.2),
        ),
        row=1,
        col=1,
    )

    cm = confusion_matrix(pred_df["actual_direction"], pred_df["predicted_direction"], labels=[0, 1])
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=["Pred Down", "Pred Up"],
            y=["Actual Down", "Actual Up"],
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=16),
            colorscale=[[0.0, "#fef2f2"], [1.0, "#dc2626"]],
            showscale=False,
            hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    w = metrics_df.loc[metrics_df["Model"] == "Weather_Integrated_Linear"].iloc[0]
    fig.update_layout(
        template="plotly_white",
        title=(
            "Arecanut Weekly Price Forecast: Weather-Integrated Linear Model"
            f"<br><sup>R2={w['R2']:.4f} | Direction Accuracy={w['Direction_Accuracy_Percent']:.2f}%</sup>"
        ),
        legend=dict(orientation="h", y=1.04, x=0.0),
        height=860,
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Price (Rs/quintal)", row=1, col=1)
    fig.update_xaxes(title_text="Predicted", row=2, col=1)
    fig.update_yaxes(title_text="Actual", row=2, col=1, autorange="reversed")

    fig.write_html(str(HTML_OUT), include_plotlyjs="cdn")


def read_float(prompt: str, default_value: float) -> float:
    while True:
        raw = input(f"{prompt} [{default_value:.3f}]: ").strip()
        if raw == "":
            return float(default_value)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid numeric value.")


def read_optional_float(prompt: str) -> float | None:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid numeric value, or press Enter to skip.")


def run_datewise_prediction_mode(merged: pd.DataFrame) -> pd.DataFrame:
    print("\nDate-wise testing mode (weather-integrated)")
    print("Enter target date as YYYY-MM-DD (example: 2026-03-15).")
    print("Type 'q' to stop.")

    records: list[dict] = []
    merged_sorted = merged.sort_values("date").reset_index(drop=True).copy()

    while True:
        raw_date = input("\nTarget date (YYYY-MM-DD): ").strip()
        if raw_date.lower() in {"q", "quit", "n", "no"}:
            break

        try:
            target_date = pd.to_datetime(raw_date, format="%Y-%m-%d", errors="raise")
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD.")
            continue

        train_df = merged_sorted[(merged_sorted["date"] < target_date)].dropna(
            subset=FEATURE_COLS + ["avg_price"]
        )
        if len(train_df) < 20:
            print("Not enough historical data before this date to train a reliable model.")
            continue

        model = LinearRegression()
        model.fit(train_df[FEATURE_COLS].values, train_df["avg_price"].values)

        iso = target_date.isocalendar()
        iso_year = int(iso.year)
        iso_week = int(iso.week)

        candidate = merged_sorted[
            (merged_sorted["iso_year"] == iso_year) & (merged_sorted["iso_week"] == iso_week)
        ].copy()

        if not candidate.empty and candidate[FEATURE_COLS].notna().all(axis=1).any():
            row = candidate[candidate[FEATURE_COLS].notna().all(axis=1)].iloc[0]
            feature_row = pd.DataFrame([row[FEATURE_COLS].to_dict()])
            actual_price = float(row["avg_price"]) if pd.notna(row["avg_price"]) else None
        else:
            last = train_df.iloc[-1]
            lag_1 = read_float("Enter lag_1 (last week price)", float(last["lag_1"]))
            lag_2 = read_float("Enter lag_2 (two weeks ago price)", float(last["lag_2"]))
            temp_mean = read_float("Enter temp_mean", float(last["temp_mean"]))
            rain_mean = read_float("Enter rain_mean", float(last["rain_mean"]))
            temp_std = read_float("Enter temp_std", float(last["temp_std"]))
            rain_std = read_float("Enter rain_std", float(last["rain_std"]))
            feature_row = pd.DataFrame(
                [
                    {
                        "t": float(len(train_df)),
                        "lag_1": lag_1,
                        "lag_2": lag_2,
                        "temp_mean": temp_mean,
                        "rain_mean": rain_mean,
                        "temp_std": temp_std,
                        "rain_std": rain_std,
                    }
                ]
            )
            actual_price = read_optional_float(
                "Enter actual observed price for this date to compute accuracy (or Enter to skip): "
            )

        predicted_price = float(model.predict(feature_row[FEATURE_COLS].values)[0])
        print(f"Predicted price for week {iso_year}-W{iso_week:02d}: {predicted_price:.2f} Rs/quintal")

        abs_error = None
        pct_error = None
        point_accuracy = None

        if actual_price is not None:
            abs_error = abs(predicted_price - actual_price)
            if actual_price != 0:
                pct_error = (abs_error / abs(actual_price)) * 100.0
                point_accuracy = max(0.0, 100.0 - pct_error)
            print(f"Actual: {actual_price:.2f}")
            print(f"Absolute Error: {abs_error:.2f}")
            if pct_error is not None:
                print(f"Percent Error: {pct_error:.2f}%")
                print(f"Point Accuracy: {point_accuracy:.2f}%")
        else:
            print("Actual price not available, so accuracy is not computed for this date.")

        records.append(
            {
                "target_date": target_date.date().isoformat(),
                "iso_year": iso_year,
                "iso_week": iso_week,
                "predicted_price": round(predicted_price, 4),
                "actual_price": None if actual_price is None else round(float(actual_price), 4),
                "absolute_error": None if abs_error is None else round(float(abs_error), 4),
                "percent_error": None if pct_error is None else round(float(pct_error), 4),
                "point_accuracy_percent": None
                if point_accuracy is None
                else round(float(point_accuracy), 4),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    price = load_price_weekly()
    temp = load_temp_weekly()
    rain = load_rain_weekly()

    merged = price.merge(temp, on=["iso_year", "iso_week"], how="left")
    merged = merged.merge(rain, on=["iso_year", "iso_week"], how="left")
    merged = add_features(merged)

    model_df, metrics_df, pred_df, _ = train_holdout_model(merged)

    model_df.to_csv(MERGED_OUT, index=False)
    metrics_df.to_csv(METRICS_OUT, index=False)
    pred_df.to_csv(PRED_OUT, index=False)
    build_html(pred_df, metrics_df)

    print("Saved:")
    print(" -", MERGED_OUT)
    print(" -", METRICS_OUT)
    print(" -", PRED_OUT)
    print(" -", HTML_OUT)
    print("\nModel comparison:")
    print(metrics_df.to_string(index=False))

    date_pred_df = run_datewise_prediction_mode(merged)
    if not date_pred_df.empty:
        date_pred_df.to_csv(DATE_PRED_OUT, index=False)
        print("\nSaved date-wise predictions:")
        print(" -", DATE_PRED_OUT)


if __name__ == "__main__":
    main()
