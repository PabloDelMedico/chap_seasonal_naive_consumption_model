import json
import sys

import numpy as np
import pandas as pd

from train import demand_series

N_SAMPLES = 100
MIN_RESIDUALS = 3


def point_forecast(series: pd.Series, period: pd.Period, season: int, fallback_window: int) -> float:
    """Value of the same month in the most recent year that has one, else the recent mean."""
    source = period - season
    while source >= series.index.min():
        value = series.get(source, np.nan)
        if not np.isnan(value):
            return float(value)
        source -= season
    recent = series.dropna().tail(fallback_window)
    return float(recent.mean()) if len(recent) else 0.0


def seasonal_errors(series: pd.Series, season: int) -> np.ndarray:
    lagged = series.copy()
    lagged.index = lagged.index + season
    errors = (series - lagged.reindex(series.index)).dropna()
    return errors.to_numpy()


def predict(model_filename: str, historic_data_filename: str, future_data_filename: str, output_filename: str):
    with open(model_filename) as f:
        settings = json.load(f)
    rng = np.random.default_rng(0)
    history = demand_series(pd.read_csv(historic_data_filename))
    future_df = pd.read_csv(future_data_filename)

    rows = []
    for _, row in future_df.iterrows():
        loc = row["location"]
        series = history.get(loc, pd.Series(dtype=float))
        period = pd.Period(str(row["time_period"]), freq="M")
        point = point_forecast(series, period, settings["season"], settings["fallback_window"]) if len(series) else 0.0
        errors = seasonal_errors(series, settings["season"]) if len(series) else np.array([])
        if len(errors) >= MIN_RESIDUALS:
            samples = point + rng.choice(errors, N_SAMPLES)
        else:
            samples = rng.poisson(max(point, 0.0), N_SAMPLES).astype(float)
        samples = np.maximum(samples, 0.0)
        row_data = {"time_period": row["time_period"], "location": loc}
        row_data.update({f"sample_{i}": s for i, s in enumerate(samples)})
        rows.append(row_data)

    pd.DataFrame(rows).to_csv(output_filename, index=False)


if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
