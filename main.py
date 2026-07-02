import argparse
import sys
import time
from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import joblib

HORIZON = 24  # forecast the next 24 hours
LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]  # hours in the past to look back
ROLL_WINDOWS = [24, 168]  # rolling mean/std windows (hours)
TEST_FRACTION = 0.15  # last 15% of hours (chronological) held out for test

# --------------------------------------------------------------------------
# 1. Data loading 
# --------------------------------------------------------------------------
def load_raw_minute_data(data_path: str | None) -> pd.DataFrame:
    """Return a minute-level DataFrame indexed by datetime with a
    'Global_active_power' column (kW). Requires a local file to be present."""

    # 1. If an explicit path is provided, verify and load it
    if data_path is not None:
        target_path = Path(data_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Specified local data file not found: {data_path}")
        print(f"Loading data from specified path: {data_path}")
        return _load_from_local_txt(str(target_path))

    # 2. Otherwise, look for the default filename in the current directory
    candidate = "household_power_consumption.txt"
    if Path(candidate).exists():
        print(f"Found local data file: {candidate}")
        return _load_from_local_txt(candidate)

    # 3. Fail immediately if no local file is found
    raise FileNotFoundError(
        f"Missing required dataset! Could not find '{candidate}' in the current directory, "
        "and no --data-path argument was provided. Please download the file locally to run this script."
    )


def _load_from_local_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )
    return _finalize_raw_frame(df)


def _finalize_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Since we only read local raw text files now, we strictly expect 'Date' and 'Time' columns
    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError("The local dataset is missing required 'Date' or 'Time' columns.")

    dt = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.copy()
    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    power_col = [c for c in df.columns if c.lower() == "global_active_power"][0]
    df["Global_active_power"] = pd.to_numeric(df[power_col], errors="coerce")
    
    return df[["Global_active_power"]]

# --------------------------------------------------------------------------
# 2. Resample to hourly
# --------------------------------------------------------------------------
def resample_hourly(minute_df: pd.DataFrame) -> pd.Series:
    """
    Convert minute-level Global_active_power (kW)
    into hourly energy consumption (kWh).
    """
    # Hourly energy (kWh)
    hourly = (
        minute_df["Global_active_power"]
        .resample("h")
        .sum() / 60
    )

    # Complete hourly index
    full_index = pd.date_range(
        hourly.index.min(),
        hourly.index.max(),
        freq="h"
    )
    hourly = hourly.reindex(full_index)
    n_missing_before = hourly.isna().sum()

    # Fill short gaps
    hourly = hourly.interpolate(
        method="time",
        limit=6,
        limit_direction="both"
    )

    # Fill longer gaps using same hour last week
    hourly = hourly.fillna(hourly.shift(24 * 7))

    # Final cleanup
    hourly = hourly.ffill().bfill()

    print(
        f"Hourly series: {len(hourly)} hours "
        f"({hourly.index.min()} -> {hourly.index.max()}), "
        f"filled {n_missing_before} missing hourly readings"
    )
    return hourly.rename("consumption_kwh")


# --------------------------------------------------------------------------
# 3. Supervised table: direct multi-horizon features + targets
# --------------------------------------------------------------------------
def build_supervised_table(hourly: pd.Series) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.DataFrame({"y": hourly})

    # --- Features: everything here only looks at index <= t ---
    for lag in LAGS:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for win in ROLL_WINDOWS:
        # shift(1) first so the rolling window covers (t-win, t], never t+.. itself
        shifted = df["y"].shift(1)
        df[f"rollmean_{win}"] = shifted.rolling(win).mean()
        df[f"rollstd_{win}"] = shifted.rolling(win).std()

    idx = df.index
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    
    # 1. Structural Interaction: Hour-of-Week
    df["hour_of_week"] = df["dow"] * 24 + df["hour"]
    df["how_sin"] = np.sin(2 * np.pi * df["hour_of_week"] / 168)
    df["how_cos"] = np.cos(2 * np.pi * df["hour_of_week"] / 168)

    # Standard cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # 2. Regional Context: Holiday Indicators (Dataset location: France)
    try:
        import holidays
        fr_holidays = holidays.France()
        df["is_holiday"] = df.index.to_series().apply(lambda x: 1 if x in fr_holidays else 0)
    except ImportError:
        df["is_holiday"] = 0

    # --- Targets: t+1 ... t+24 (strictly future, never used as features) ---
    target_cols = []
    for h in range(1, HORIZON + 1):
        col = f"target_h{h}"
        df[col] = df["y"].shift(-h)
        target_cols.append(col)

    feature_cols = [c for c in df.columns if c not in target_cols and c != "y"]

    df = df.dropna(subset=feature_cols + target_cols)
    return df, feature_cols, target_cols


# --------------------------------------------------------------------------
# 4. Chronological split
# --------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_test = int(n * test_fraction)
    train_df = df.iloc[: n - n_test]
    test_df = df.iloc[n - n_test :]
    print(
        f"Train: {len(train_df)} origins ({train_df.index.min()} -> {train_df.index.max()}) | "
        f"Test: {len(test_df)} origins ({test_df.index.min()} -> {test_df.index.max()})"
    )
    return train_df, test_df


# --------------------------------------------------------------------------
# 5. Model
# --------------------------------------------------------------------------
def train_model(train_df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]) -> MultiOutputRegressor:
    # Optimized base booster settings for tabular multi-horizon regression
    base_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=0,
    )
    # Using the MultiOutput Wrapper to train 24 completely specialized horizon models
    model = MultiOutputRegressor(base_model)
    
    t0 = time.time()
    model.fit(train_df[feature_cols].values, train_df[target_cols].values)
    print(f"XGBoost MultiOutput model trained in {time.time() - t0:.1f}s "
          f"({len(train_df)} rows x {len(feature_cols)} features -> {len(target_cols)} horizons)")
    return model


# --------------------------------------------------------------------------
# 6. Baselines + evaluation & plotting
# --------------------------------------------------------------------------
def _lag_matrix(hourly: pd.Series, origins: pd.DatetimeIndex, lag_hours: int) -> np.ndarray:
    """For every origin t and horizon h=1..24, look up y[t + h - lag_hours]."""
    preds = np.zeros((len(origins), HORIZON))
    for h in range(1, HORIZON + 1):
        shifted_times = origins + pd.Timedelta(hours=h - lag_hours)
        preds[:, h - 1] = hourly.reindex(shifted_times).values
    return preds


def naive_persistence_forecast(hourly: pd.Series, origins: pd.DatetimeIndex) -> np.ndarray:
    """'Tomorrow looks like today': forecast(t+h) = y(t+h-24)."""
    return _lag_matrix(hourly, origins, lag_hours=24)


def seasonal_naive_forecast(hourly: pd.Series, origins: pd.DatetimeIndex) -> np.ndarray:
    """'Same hour, last week': forecast(t+h) = y(t+h-168)."""
    return _lag_matrix(hourly, origins, lag_hours=168)


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.float64, np.float64]:
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    return mae, rmse


def generate_plots(
    results: dict, 
    model_maes: list[float], 
    seasonal_maes: list[float], 
    feature_cols: list[str], 
    model: MultiOutputRegressor, 
    test_df: pd.DataFrame, 
    target_cols: list[str], 
    y_pred_model: np.ndarray
):
    """Generates and saves summary visualisations to disk."""
    print("\nGenerating evaluation plots...")

    # 1. Overall MAE Comparison Bar Chart (sorted order)
    sorted_results = sorted(results.items(), key=lambda x: x[1][0])
    methods = [x[0] for x in sorted_results]
    maes = [x[1][0] for x in sorted_results]
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(methods, maes, color=['#2ca02c', '#1f77b4', '#ff7f0e'])
    ax.set_xlabel("Overall MAE (kW)")
    ax.set_title("Overall Mean Absolute Error Comparison (Lower is Better)")
    ax.bar_label(bars, fmt='%.4f', padding=5)
    plt.tight_layout()
    plt.savefig("overall_mae_comparison.png", dpi=150)
    plt.close()

    # 2. Per-Horizon Error Degradation
    fig, ax = plt.subplots(figsize=(10, 5))
    horizons = list(range(1, HORIZON + 1))
    ax.plot(horizons, model_maes, marker='o', label='XGBoost (MultiOutput)', color='#2ca02c', linewidth=2)
    ax.plot(horizons, seasonal_maes, marker='x', label='Seasonal Naive (Same hour last week)', color='#ff7f0e', linestyle='--')
    ax.set_xlabel("Forecast Horizon (Hours Ahead)")
    ax.set_ylabel("MAE (kW)")
    ax.set_title("Per-Horizon Error Breakdown")
    ax.set_xticks(horizons)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig("per_horizon_mae.png", dpi=150)
    plt.close()

    # 3. Feature Importance (Top 15 Features) - Aggregated from the multi-output ensemble
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    top_feat_imp = feat_imp.tail(15)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    top_feat_imp.plot(kind='barh', ax=ax, color='#1f77b4', edgecolor='none')
    ax.set_xlabel("Average Gain Importance")
    ax.set_title("Top 15 Most Influential Features (Averaged over Horizons)")
    ax.grid(True, axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

    # 4. Test Snapshot Comparison
    if len(test_df) > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        idx = 0 
        origin_time = test_df.index[idx]
        y_true_sample = test_df[target_cols].iloc[idx].values
        y_pred_sample = y_pred_model[idx]
        
        forecast_times = [origin_time + pd.Timedelta(hours=h) for h in range(1, HORIZON + 1)]
        
        ax.plot(forecast_times, y_true_sample, marker='o', label='Actual Consumption', color='#111111', linewidth=1.5)
        ax.plot(forecast_times, y_pred_sample, marker='s', label='XGBoost MultiOutput Forecast', color='#2ca02c', linewidth=1.5)
        ax.set_xlabel("Forecast Timeline")
        ax.set_ylabel("Consumption (kW)")
        ax.set_title(f"24-Hour Day-Ahead Forecast Snapshot (Origin: {origin_time})")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig("forecast_snapshot.png", dpi=150)
        plt.close()
    
    print("Visualisations exported successfully as PNG files.")


def evaluate_all(test_df: pd.DataFrame, target_cols: list[str], model: MultiOutputRegressor, feature_cols: list[str], hourly: pd.Series) -> dict[str, tuple[np.float64, np.float64]]:
    y_true = test_df[target_cols].values

    y_pred_model = model.predict(test_df[feature_cols].values)
    y_pred_persist = naive_persistence_forecast(hourly, test_df.index)
    y_pred_seasonal = seasonal_naive_forecast(hourly, test_df.index)

    results = {}
    for name, y_pred in [
        ("XGBoost (MultiOutput wrapper)", y_pred_model),
        ("Naive: yesterday same hour", y_pred_persist),
        ("Naive: same hour last week", y_pred_seasonal),
    ]:
        mae, rmse = mae_rmse(y_true, y_pred)
        results[name] = (mae, rmse)

    print("\n=== Day-ahead (24h) forecast error on held-out test period ===")
    print(f"{'Method':38s} {'MAE (kW)':>10s} {'RMSE (kW)':>10s}")
    for name, (mae, rmse) in results.items():
        print(f"{name:38s} {mae:10.4f} {rmse:10.4f}")

    # Per-horizon breakdown for the model vs the better baseline
    print("\nPer-horizon MAE (hours ahead), model vs seasonal-naive baseline:")
    print(f"{'h':>3s} {'model MAE':>10s} {'seasonal-naive MAE':>20s}")
    model_maes = []
    seasonal_maes = []
    for h in range(1, HORIZON + 1):
        m_mae = np.mean(np.abs(y_true[:, h - 1] - y_pred_model[:, h - 1]))
        s_mae = np.mean(np.abs(y_true[:, h - 1] - y_pred_seasonal[:, h - 1]))
        print(f"{h:3d} {m_mae:10.4f} {s_mae:20.4f}")
        model_maes.append(m_mae)
        seasonal_maes.append(s_mae)

    # Triggering image creation
    generate_plots(
        results=results,
        model_maes=model_maes,
        seasonal_maes=seasonal_maes,
        feature_cols=feature_cols,
        model=model,
        test_df=test_df,
        target_cols=target_cols,
        y_pred_model=y_pred_model
    )

    return results


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to a local household_power_consumption.txt (UCI raw format). "
        "If omitted, the script tries common local filenames, then ucimlrepo.",
    )
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    args, unknown = parser.parse_known_args()

    # Step 1: Data prep pipeline
    minute_df = load_raw_minute_data(args.data_path)
    hourly = resample_hourly(minute_df)
    table, feature_cols, target_cols = build_supervised_table(hourly)
    train_df, test_df = chronological_split(table, args.test_fraction)

    # Step 2: Train Model
    model = train_model(train_df, feature_cols, target_cols)

    # Step 3: Run Evaluation and generate figures
    results = evaluate_all(test_df, target_cols, model, feature_cols, hourly)

    # Step 4: Track with MLflow inside a single structured execution context
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("colsample_bytree", 0.8)

        # Log metrics
        mlflow.log_metric("MAE", results["XGBoost (MultiOutput wrapper)"][0])
        mlflow.log_metric("RMSE", results["XGBoost (MultiOutput wrapper)"][1])

        # Log the generated plot artifacts
        mlflow.log_artifact("overall_mae_comparison.png")
        mlflow.log_artifact("feature_importance.png")
        mlflow.log_artifact("forecast_snapshot.png")
        mlflow.log_artifact("per_horizon_mae.png")

        # Log the trained multioutput wrapper cleanly via MLflow Sklearn API
        mlflow.sklearn.log_model(
    model, 
    artifact_path="xgb_multioutput_model", 
    serialization_format="pickle"
)

    # Local fallback save 
    joblib.dump(model, "model.pkl")
    print("Pipeline complete. Model saved locally to 'model.pkl' and tracked in MLflow.")


if __name__ == "__main__":
    main()