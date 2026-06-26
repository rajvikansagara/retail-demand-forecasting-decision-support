"""
Retail demand forecasting thesis reproduction
=============================================

"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
STORE_ID = 1
TEST_START = pd.Timestamp("2015-06-20")
TEST_END = pd.Timestamp("2015-07-31")
SEQUENCE_LENGTH = 28
LEAD_TIME_DAYS = 7
SERVICE_FACTOR = 1.65

REPORTED_METRICS = pd.DataFrame(
    [
        (1, "XGBoost", 245.81, 317.28, 6.53),
        (2, "LSTM", 334.93, 476.36, 8.06),
        (3, "SARIMA", 566.30, 716.08, 14.85),
        (4, "Seasonal Naive", 981.45, 1162.32, 26.29),
    ],
    columns=["Rank", "Model", "MAE", "RMSE", "MAPE (%)"],
)

REPORTED_FEATURE_IMPORTANCE = pd.DataFrame(
    [
        (1, "Open", 1.441424, 0.201482, "Store opening status"),
        (2, "Promo", 0.091203, 0.015705, "Promotional activity"),
        (3, "SalesLag1", 0.029474, 0.009270, "Previous-day sales"),
        (4, "Day", 0.015358, 0.006533, "Day within month"),
        (5, "DayOfWeek", 0.010621, 0.004495, "Weekly pattern"),
        (6, "SalesLag14", 0.003777, 0.002087, "Two-week lagged sales"),
        (7, "RollingMean7", 0.000617, 0.001647, "Recent weekly average"),
        (8, "RollingStd14", 0.000237, 0.000455, "Two-week volatility"),
        (9, "RollingStd7", 0.000218, 0.001500, "Recent weekly volatility"),
        (10, "RollingMean28", 0.000189, 0.000474, "Monthly rolling average"),
    ],
    columns=["Rank", "Feature", "Importance mean", "Importance std.", "Interpretation"],
)

REPORTED_INVENTORY = pd.DataFrame(
    [
        ("SARIMA", 3094.26, 29109.37, 0, 0.00, 1663.09),
        ("XGBoost", 1385.06, 28904.98, 0, 0.00, 1462.47),
        ("LSTM", 1966.66, 27780.00, 0, 0.00, 1473.60),
        ("Seasonal Naive", 5064.07, 30977.41, 1, 2656.93, 2678.37),
    ],
    columns=[
        "Model",
        "Safety stock",
        "Average reorder point",
        "Stockout days",
        "Stockout units",
        "Average excess inventory",
    ],
)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    output: Path
    figures_generated: Path
    figures_reference: Path
    figures_report_exact: Path
    models: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        return cls(
            root=root,
            output=root / "outputs",
            figures_generated=root / "figures" / "generated",
            figures_reference=root / "figures" / "reference",
            figures_report_exact=root / "figures" / "report_exact",
            models=root / "models",
        )

    def ensure(self) -> None:
        for folder in (
            self.output,
            self.figures_generated,
            self.figures_reference,
            self.figures_report_exact,
            self.models,
        ):
            folder.mkdir(parents=True, exist_ok=True)


def set_reproducible_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], filename: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")


def load_datasets(paths: ProjectPaths) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = paths.root / "train.csv"
    test_path = paths.root / "test.csv"
    store_path = paths.root / "store.csv"

    for path in (train_path, test_path, store_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train = pd.read_csv(
        train_path,
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False,
    )
    test = pd.read_csv(
        test_path,
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False,
    )
    store = pd.read_csv(store_path)

    require_columns(
        train,
        ["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday"],
        "train.csv",
    )
    require_columns(
        test,
        ["Id", "Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"],
        "test.csv",
    )
    require_columns(
        store,
        [
            "Store",
            "StoreType",
            "Assortment",
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2",
            "Promo2SinceWeek",
            "Promo2SinceYear",
            "PromoInterval",
        ],
        "store.csv",
    )

    train["StateHoliday"] = train["StateHoliday"].astype(str).replace({"0.0": "0", "<NA>": "0"})
    test["StateHoliday"] = test["StateHoliday"].astype(str).replace({"0.0": "0", "<NA>": "0"})
    return train, test, store


def prepare_store_table(store: pd.DataFrame) -> pd.DataFrame:
    result = store.copy()
    result["CompetitionDistance"] = result["CompetitionDistance"].fillna(result["CompetitionDistance"].median())
    for column in (
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ):
        result[column] = result[column].fillna(0)
    result["PromoInterval"] = result["PromoInterval"].fillna("None")
    return result


def write_reported_outputs(paths: ProjectPaths) -> None:
    REPORTED_METRICS.to_csv(paths.output / "table_8_model_comparison_reported.csv", index=False)
    REPORTED_FEATURE_IMPORTANCE.to_csv(paths.output / "table_9_feature_importance_reported.csv", index=False)
    REPORTED_INVENTORY.to_csv(paths.output / "table_10_inventory_results_reported.csv", index=False)

    paths.figures_report_exact.mkdir(parents=True, exist_ok=True)
    for source in paths.figures_reference.glob("*.png"):
        shutil.copy2(source, paths.figures_report_exact / source.name)


def write_dataset_tables(
    train: pd.DataFrame,
    test: pd.DataFrame,
    store: pd.DataFrame,
    paths: ProjectPaths,
) -> None:
    dataset_structure = pd.DataFrame(
        [
            ("train.csv", len(train), train.shape[1], "Historical sales for model development", f"{train.Date.min().date()} to {train.Date.max().date()}"),
            ("test.csv", len(test), test.shape[1], "Future records without actual sales", f"{test.Date.min().date()} to {test.Date.max().date()}"),
            ("store.csv", len(store), store.shape[1], "Store-level characteristics", "Store-level data only"),
        ],
        columns=["File", "Rows", "Columns", "Main purpose", "Date coverage"],
    )
    dataset_structure.to_csv(paths.output / "table_1_dataset_structure.csv", index=False)

    stats = train[["Sales", "Customers", "Open", "Promo", "SchoolHoliday"]].describe().T
    stats = stats.rename(
        index={"SchoolHoliday": "School Holiday"},
        columns={"count": "Count", "mean": "Mean", "std": "Std.", "min": "Min", "50%": "Median", "max": "Max"},
    )[["Count", "Mean", "Std.", "Min", "Median", "Max"]]
    stats.to_csv(paths.output / "table_4_descriptive_statistics.csv")

    promo_table = (
        train.groupby("Promo")["Sales"]
        .agg(Count="count", **{"Mean sales": "mean", "Median sales": "median"})
        .reset_index()
    )
    promo_table.insert(0, "Category", "Promo")
    promo_table["Group"] = promo_table["Promo"].map({0: "No promotion", 1: "Promotion active"})
    promo_table = promo_table[["Category", "Group", "Count", "Mean sales", "Median sales"]]

    dow_table = (
        train.groupby("DayOfWeek")["Sales"]
        .agg(Count="count", **{"Mean sales": "mean", "Median sales": "median"})
        .reset_index()
    )
    dow_table.insert(0, "Category", "DayOfWeek")
    dow_table["Group"] = dow_table["DayOfWeek"].astype(str)
    dow_table = dow_table[["Category", "Group", "Count", "Mean sales", "Median sales"]]
    pd.concat([promo_table, dow_table], ignore_index=True).to_csv(
        paths.output / "table_6_sales_by_promotion_and_day_of_week.csv", index=False
    )


def plot_basic_eda(train: pd.DataFrame, paths: ProjectPaths) -> None:
    store_one = train.loc[train["Store"] == STORE_ID].sort_values("Date")
    train_part = store_one.loc[store_one["Date"] < TEST_START]
    test_part = store_one.loc[(store_one["Date"] >= TEST_START) & (store_one["Date"] <= TEST_END)]

    fig, ax = plt.subplots(figsize=(13.89, 4.90))
    ax.plot(train_part["Date"], train_part["Sales"], label="Train")
    ax.plot(test_part["Date"], test_part["Sales"], label="Test")
    ax.axvline(TEST_START, linestyle="--", label="Train/Test Split")
    ax.set_title("Time-Based Train-Test Split for Store 1")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(paths.figures_generated / "figure_3_time_based_train_test_split_store_1.png", dpi=100)
    plt.close(fig)

    total_daily = train.groupby("Date", as_index=False)["Sales"].sum().sort_values("Date")
    fig, ax = plt.subplots(figsize=(13.86, 4.90))
    ax.plot(total_daily["Date"], total_daily["Sales"])
    ax.set_title("Total Daily Sales Across All Stores")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(paths.figures_generated / "figure_4_total_daily_sales_all_stores.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.87, 4.90))
    ax.plot(store_one["Date"], store_one["Sales"])
    ax.set_title("Daily Sales for Store 1")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(paths.figures_generated / "figure_5_daily_sales_store_1.png", dpi=100)
    plt.close(fig)


def fixed_category_dummies(frame: pd.DataFrame) -> pd.DataFrame:
    category_levels: Mapping[str, Sequence[str]] = {
        "StateHoliday": ["0", "a", "b", "c"],
        "StoreType": ["a", "b", "c", "d"],
        "Assortment": ["a", "b", "c"],
        "PromoInterval": ["Feb,May,Aug,Nov", "Jan,Apr,Jul,Oct", "Mar,Jun,Sept,Dec", "None"],
    }
    categorical = pd.DataFrame(index=frame.index)
    for column, levels in category_levels.items():
        categorical[column] = pd.Categorical(frame[column].astype(str), categories=levels)
    return pd.get_dummies(categorical, drop_first=True, dtype=int)


def build_store_one_features(
    train: pd.DataFrame,
    store: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    cleaned_store = prepare_store_table(store)
    merged = train.merge(cleaned_store, on="Store", how="left")
    selected = merged.loc[merged["Store"] == STORE_ID].sort_values("Date").reset_index(drop=True)

    selected["Year"] = selected["Date"].dt.year
    selected["Month"] = selected["Date"].dt.month
    selected["Day"] = selected["Date"].dt.day
    selected["WeekOfYear"] = selected["Date"].dt.isocalendar().week.astype(int)
    selected["Quarter"] = selected["Date"].dt.quarter
    selected["IsMonthStart"] = selected["Date"].dt.is_month_start.astype(int)
    selected["IsWeekend"] = (selected["DayOfWeek"] >= 6).astype(int)

    selected["CompetitionOpenMonths"] = (
        (selected["Year"] - selected["CompetitionOpenSinceYear"]) * 12
        + (selected["Month"] - selected["CompetitionOpenSinceMonth"])
    ).clip(lower=0)
    selected.loc[selected["CompetitionOpenSinceYear"] <= 0, "CompetitionOpenMonths"] = 0

    selected["Promo2AgeWeeks"] = (
        (selected["Year"] - selected["Promo2SinceYear"]) * 52
        + (selected["WeekOfYear"] - selected["Promo2SinceWeek"])
    ).clip(lower=0)
    selected.loc[selected["Promo2SinceYear"] <= 0, "Promo2AgeWeeks"] = 0

    for lag in (1, 7, 14):
        selected[f"SalesLag{lag}"] = selected["Sales"].shift(lag)

    shifted_sales = selected["Sales"].shift(1)
    for window in (7, 14, 28):
        selected[f"RollingMean{window}"] = shifted_sales.rolling(window).mean()
        selected[f"RollingStd{window}"] = shifted_sales.rolling(window).std()

    numeric_features = [
        "Store",
        "DayOfWeek",
        "Open",
        "Promo",
        "SchoolHoliday",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "Year",
        "Month",
        "Day",
        "WeekOfYear",
        "Quarter",
        "IsMonthStart",
        "IsWeekend",
        "CompetitionOpenMonths",
        "Promo2AgeWeeks",
        "SalesLag1",
        "SalesLag7",
        "SalesLag14",
        "RollingMean7",
        "RollingStd7",
        "RollingMean14",
        "RollingStd14",
        "RollingMean28",
        "RollingStd28",
    ]

    encoded = fixed_category_dummies(selected)
    feature_matrix = pd.concat([selected[numeric_features], encoded], axis=1)
    valid = feature_matrix.notna().all(axis=1)
    feature_matrix = feature_matrix.loc[valid].astype(float).reset_index(drop=True)
    target = selected.loc[valid, "Sales"].astype(float).reset_index(drop=True)
    dates = selected.loc[valid, "Date"].reset_index(drop=True)
    modelling_frame = selected.loc[valid].reset_index(drop=True)

    if feature_matrix.shape[1] != 40:
        raise RuntimeError(f"Expected 40 XGBoost features, found {feature_matrix.shape[1]}.")

    return feature_matrix, target, dates, modelling_frame


def write_split_table(dates: pd.Series, paths: ProjectPaths) -> None:
    train_dates = dates.loc[dates < TEST_START]
    test_dates = dates.loc[(dates >= TEST_START) & (dates <= TEST_END)]
    table = pd.DataFrame(
        [
            ("Selected store", f"Store {STORE_ID}"),
            ("Raw selected-store records", "942 days"),
            ("Selected-store date range", "2013-01-01 to 2015-07-31"),
            ("Rows after feature engineering", str(len(dates))),
            ("Training period", f"{train_dates.min().date()} to {train_dates.max().date()}"),
            ("Training rows", str(len(train_dates))),
            ("Testing period", f"{test_dates.min().date()} to {test_dates.max().date()}"),
            ("Testing rows", str(len(test_dates))),
        ],
        columns=["Item", "Value"],
    )
    table.to_csv(paths.output / "table_2_selected_store_and_time_split.csv", index=False)
    table.to_csv(paths.output / "table_3_train_test_split_store_1.csv", index=False)


def mape_excluding_zero(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def metric_row(model_name: str, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float | str]:
    predicted = np.maximum(0, np.asarray(predicted, dtype=float))
    return {
        "Model": model_name,
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(math.sqrt(mean_squared_error(actual, predicted))),
        "MAPE (%)": mape_excluding_zero(actual, predicted),
    }


def fit_seasonal_naive(raw_store: pd.DataFrame) -> np.ndarray:
    series = raw_store.sort_values("Date").copy()
    series["SeasonalNaive"] = series["Sales"].shift(7)
    return series.loc[(series["Date"] >= TEST_START) & (series["Date"] <= TEST_END), "SeasonalNaive"].to_numpy(dtype=float)


def fit_sarima(raw_store: pd.DataFrame, paths: ProjectPaths) -> np.ndarray:
    ordered = raw_store.sort_values("Date")
    training = ordered.loc[ordered["Date"] < TEST_START, "Sales"].astype(float)
    model = SARIMAX(
        training,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=100)
    result.save(paths.models / "sarima_store_1.pkl")
    return np.maximum(0, np.asarray(result.forecast(42), dtype=float))


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    paths: ProjectPaths,
) -> Tuple[np.ndarray, XGBRegressor, pd.DataFrame]:
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.10,
        max_depth=4,
        min_child_weight=1,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    predictions = np.maximum(0, model.predict(X_test))
    joblib.dump(model, paths.models / "xgboost_store_1.joblib")

    return predictions, model, X_test


def calculate_permutation_importance(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="r2",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance mean": result.importances_mean,
            "Importance std.": result.importances_std,
        }
    ).sort_values("Importance mean", ascending=False, ignore_index=True)
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    return table


def fit_lstm(
    raw_store: pd.DataFrame,
    paths: ProjectPaths,
    epochs: int = 40,
) -> Tuple[np.ndarray, List[float], List[float]]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the LSTM. Install requirements.txt first.") from exc

    set_reproducible_seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(1)

    data = raw_store.sort_values("Date").reset_index(drop=True).copy()
    data["Day"] = data["Date"].dt.day
    data["Month"] = data["Date"].dt.month
    lstm_features = ["Sales", "Open", "Promo", "DayOfWeek", "Day", "Month"]

    raw_values = data[lstm_features].astype(float).to_numpy()
    train_raw_count = int((data["Date"] < TEST_START).sum())
    feature_scaler = MinMaxScaler().fit(raw_values[:train_raw_count])
    feature_values = feature_scaler.transform(raw_values)

    target_values = data[["Sales"]].astype(float).to_numpy()
    target_scaler = MinMaxScaler().fit(target_values[:train_raw_count])
    scaled_target = target_scaler.transform(target_values).ravel()

    sequences: List[np.ndarray] = []
    targets: List[float] = []
    target_dates: List[pd.Timestamp] = []
    for index in range(SEQUENCE_LENGTH, len(data)):
        sequences.append(feature_values[index - SEQUENCE_LENGTH : index])
        targets.append(scaled_target[index])
        target_dates.append(data.loc[index, "Date"])

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    dates = pd.Series(target_dates)
    training_mask = (dates < TEST_START).to_numpy()
    testing_mask = ((dates >= TEST_START) & (dates <= TEST_END)).to_numpy()

    X_all_train, y_all_train = X[training_mask], y[training_mask]
    X_test = X[testing_mask]
    validation_start = int(len(X_all_train) * 0.85)
    X_train, y_train = X_all_train[:validation_start], y_all_train[:validation_start]
    X_validation, y_validation = X_all_train[validation_start:], y_all_train[validation_start:]

    class SalesLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, 1)

        def forward(self, inputs):
            sequence_output, _ = self.lstm(inputs)
            return self.output(sequence_output[:, -1, :]).squeeze(-1)

    model = SalesLSTM(input_size=X.shape[2], hidden_size=64)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    batch_size = 32
    training_loss: List[float] = []
    validation_loss: List[float] = []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for start in range(0, len(X_train), batch_size):
            end = start + batch_size
            batch_X = torch.from_numpy(X_train[start:end])
            batch_y = torch.from_numpy(y_train[start:end])
            optimiser.zero_grad()
            loss = loss_function(model(batch_X), batch_y)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item()) * len(batch_X)
        training_loss.append(epoch_loss / len(X_train))

        model.eval()
        with torch.no_grad():
            val_loss = loss_function(
                model(torch.from_numpy(X_validation)),
                torch.from_numpy(y_validation),
            )
        validation_loss.append(float(val_loss.item()))

    model.eval()
    with torch.no_grad():
        scaled_predictions = model(torch.from_numpy(X_test)).numpy()
    predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).ravel()
    predictions = np.maximum(0, predictions)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": lstm_features,
            "sequence_length": SEQUENCE_LENGTH,
            "feature_scaler_min": feature_scaler.min_,
            "feature_scaler_scale": feature_scaler.scale_,
            "target_scaler_min": target_scaler.min_,
            "target_scaler_scale": target_scaler.scale_,
        },
        paths.models / "lstm_store_1.pt",
    )
    return predictions, training_loss, validation_loss


def inventory_decision_support(
    actual: np.ndarray,
    forecast: np.ndarray,
    model_name: str,
) -> Dict[str, float | int | str]:
    actual = np.asarray(actual, dtype=float)
    forecast = np.maximum(0, np.asarray(forecast, dtype=float))
    errors = actual - forecast
    safety_stock = SERVICE_FACTOR * float(np.std(errors, ddof=0)) * math.sqrt(LEAD_TIME_DAYS)
    average_reorder_point = float(np.mean(forecast * LEAD_TIME_DAYS + safety_stock))

    # Simple day-level decision-support simulation. It is deliberately a
    # transparent illustration rather than a complete inventory optimiser.
    on_hand = average_reorder_point
    arrivals: Dict[int, float] = {}
    stockout_days = 0
    stockout_units = 0.0
    excess_levels: List[float] = []

    for day, demand in enumerate(actual):
        on_hand += arrivals.pop(day, 0.0)
        fulfilled = min(on_hand, demand)
        shortage = max(demand - on_hand, 0.0)
        if shortage > 0:
            stockout_days += 1
            stockout_units += shortage
        on_hand -= fulfilled

        expected_next_week = float(forecast[day : day + LEAD_TIME_DAYS].sum())
        reorder_point = expected_next_week + safety_stock
        if on_hand <= reorder_point:
            order_quantity = max(reorder_point - on_hand, 0.0)
            arrivals[day + LEAD_TIME_DAYS] = arrivals.get(day + LEAD_TIME_DAYS, 0.0) + order_quantity

        excess_levels.append(max(on_hand - forecast[day], 0.0))

    return {
        "Model": model_name,
        "Safety stock": safety_stock,
        "Average reorder point": average_reorder_point,
        "Stockout days": stockout_days,
        "Stockout units": stockout_units,
        "Average excess inventory": float(np.mean(excess_levels)),
    }


def plot_retrained_results(
    test_dates: pd.Series,
    actual: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    inventory: pd.DataFrame,
    training_loss: Sequence[float],
    validation_loss: Sequence[float],
    paths: ProjectPaths,
) -> None:
    fig, ax = plt.subplots(figsize=(7.90, 3.90))
    ax.plot(range(len(training_loss)), training_loss, label="Training Loss")
    ax.plot(range(len(validation_loss)), validation_loss, label="Validation Loss")
    ax.set_title("LSTM Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures_generated / "figure_6_lstm_training_validation_loss.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.85, 5.90))
    ax.plot(test_dates, actual, label="Actual Sales")
    for name in ("Seasonal Naive", "SARIMA", "XGBoost", "LSTM"):
        ax.plot(test_dates, predictions[name], label=name)
    ax.set_title("Actual vs Forecasted Sales for Store 1")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(paths.figures_generated / "figure_7_actual_vs_forecasted_sales_store_1.png", dpi=100)
    plt.close(fig)

    order = ["XGBoost", "LSTM", "SARIMA", "Seasonal Naive"]
    metric_plot = metrics.set_index("Model").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(9.89, 4.90))
    ax.bar(metric_plot["Model"], metric_plot["RMSE"])
    ax.set_title("Model Comparison by RMSE")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(paths.figures_generated / "figure_8_model_comparison_rmse.png", dpi=100)
    plt.close(fig)

    top = feature_importance.head(20).sort_values("Importance mean")
    fig, ax = plt.subplots(figsize=(9.90, 5.90))
    ax.barh(top["Feature"], top["Importance mean"])
    ax.set_title("Top Demand Drivers from Explainability Analysis")
    ax.set_xlabel("ImportanceMean")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(paths.figures_generated / "figure_9_top_demand_drivers.png", dpi=100)
    plt.close(fig)

    inventory_order = ["Seasonal Naive", "SARIMA", "XGBoost", "LSTM"]
    inventory_plot = inventory.set_index("Model").loc[inventory_order].reset_index()
    fig, ax = plt.subplots(figsize=(9.89, 4.90))
    ax.bar(inventory_plot["Model"], inventory_plot["Stockout units"])
    ax.set_title("Inventory Simulation: Stockout Units by Forecasting Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Stockout Units")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(paths.figures_generated / "figure_10_stockout_units_by_model.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.89, 4.90))
    ax.bar(inventory_plot["Model"], inventory_plot["Average excess inventory"])
    ax.set_title("Inventory Simulation: Average Excess Inventory by Forecasting Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Average Excess Inventory")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(paths.figures_generated / "figure_11_average_excess_inventory_by_model.png", dpi=100)
    plt.close(fig)


def retrain_workflow(
    train: pd.DataFrame,
    test: pd.DataFrame,
    store: pd.DataFrame,
    paths: ProjectPaths,
) -> None:
    print("[1/7] Writing dataset tables and exploratory figures...")
    write_dataset_tables(train, test, store, paths)
    plot_basic_eda(train, paths)

    print("[2/7] Engineering 40 features for Store 1...")
    X, y, dates, modelling_frame = build_store_one_features(train, store)
    write_split_table(dates, paths)

    training_mask = dates < TEST_START
    testing_mask = (dates >= TEST_START) & (dates <= TEST_END)
    X_train, X_test = X.loc[training_mask], X.loc[testing_mask]
    y_train, y_test = y.loc[training_mask], y.loc[testing_mask]
    test_dates = dates.loc[testing_mask].reset_index(drop=True)
    actual = y_test.to_numpy(dtype=float)

    raw_store = train.loc[train["Store"] == STORE_ID].sort_values("Date").reset_index(drop=True)

    print("[3/7] Fitting seasonal-naive and SARIMA models...")
    seasonal_naive = fit_seasonal_naive(raw_store)
    sarima = fit_sarima(raw_store, paths)

    print("[4/7] Fitting XGBoost model...")
    xgboost_predictions, xgboost_model, _ = fit_xgboost(X_train, y_train, X_test, paths)

    print("[5/7] Fitting LSTM model...")
    lstm_predictions, training_loss, validation_loss = fit_lstm(raw_store, paths, epochs=40)

    predictions: Dict[str, np.ndarray] = {
        "Seasonal Naive": seasonal_naive,
        "SARIMA": sarima,
        "XGBoost": xgboost_predictions,
        "LSTM": lstm_predictions,
    }

    print("[6/7] Calculating metrics, feature importance and inventory outputs...")
    metric_rows = [metric_row(name, actual, prediction) for name, prediction in predictions.items()]
    metrics = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)
    metrics.insert(0, "Rank", np.arange(1, len(metrics) + 1))
    metrics.to_csv(paths.output / "model_comparison_retrained.csv", index=False)

    feature_importance = calculate_permutation_importance(xgboost_model, X_test, y_test)
    feature_importance.to_csv(paths.output / "feature_importance_retrained.csv", index=False)

    inventory_rows = [
        inventory_decision_support(actual, prediction, name)
        for name, prediction in predictions.items()
    ]
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(paths.output / "inventory_results_retrained.csv", index=False)

    forecast_table = pd.DataFrame({"Date": test_dates, "Actual Sales": actual})
    for name, prediction in predictions.items():
        forecast_table[name] = prediction
    forecast_table.to_csv(paths.output / "forecast_predictions_retrained.csv", index=False)

    metadata = {
        "selected_store": STORE_ID,
        "raw_store_rows": int(len(raw_store)),
        "rows_after_feature_engineering": int(len(X)),
        "training_rows": int(training_mask.sum()),
        "testing_rows": int(testing_mask.sum()),
        "training_start": str(dates.loc[training_mask].min().date()),
        "training_end": str(dates.loc[training_mask].max().date()),
        "testing_start": str(test_dates.min().date()),
        "testing_end": str(test_dates.max().date()),
        "xgboost_features": int(X.shape[1]),
        "mape_zero_sales_rule": "Zero-sales days excluded from MAPE denominator.",
    }
    (paths.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[7/7] Saving regenerated figures...")
    plot_retrained_results(
        test_dates,
        actual,
        predictions,
        metrics,
        feature_importance,
        inventory,
        training_loss,
        validation_loss,
        paths,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the retail demand forecasting thesis analysis.")
    parser.add_argument(
        "--mode",
        choices=("reported", "retrain", "both"),
        default="both",
        help="reported writes exact thesis outputs; retrain refits the models; both does both.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing train.csv, test.csv and store.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_root(args.project_root.resolve())
    paths.ensure()
    set_reproducible_seed()

    try:
        train, test, store = load_datasets(paths)
        write_dataset_tables(train, test, store, paths)

        if args.mode in {"reported", "both"}:
            print("Writing exact Chapter 4 reported tables and reference figures...")
            write_reported_outputs(paths)

        if args.mode in {"retrain", "both"}:
            retrain_workflow(train, test, store, paths)

        print("\nCompleted successfully.")
        print(f"Tables and CSV outputs: {paths.output}")
        print(f"Exact thesis figures:     {paths.figures_report_exact}")
        print(f"Regenerated figures:      {paths.figures_generated}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
