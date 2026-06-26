# Retail Demand Forecasting Thesis Code

This project reproduces the methodology and Chapter 4 workflow for the thesis **Comparative Analysis of Machine Learning and Deep Learning Models for Retail Demand Forecasting and Inventory Optimization**.

## Project structure

```text
retail-demand-forecasting-thesis/
├── train.csv
├── test.csv
├── store.csv
├── main.py
├── requirements.txt
├── README.md
├── models/
├── notebooks/
├── outputs/
└── figures/
    ├── reference/
    ├── report_exact/
    └── generated/
```

## What the code does

The script follows the thesis methodology:

1. Loads and validates `train.csv`, `test.csv`, and `store.csv`.
2. Audits the Rossmann dataset and creates the descriptive tables used in Chapter 4.
3. Selects **Store 1** for the controlled comparison.
4. Creates a chronological test period of **42 days**, from **20 June 2015 to 31 July 2015**.
5. Creates **40 forecasting features**, including date variables, promotion and holiday indicators, store characteristics, lagged sales, and rolling statistics.
6. Fits the following models:
   - Seasonal Naive
   - SARIMA `(1,1,1)(1,1,1,7)`
   - XGBoost
   - LSTM
7. Evaluates all models using MAE, RMSE, and MAPE.
8. Produces permutation feature importance for XGBoost.
9. Runs a simplified inventory decision-support simulation.
10. Saves models, tables, predictions, and figures.

`Customers` is excluded from the forecasting feature matrix because same-day customer counts would not normally be known at the time a future sales forecast is produced.

## Installation

Open a terminal inside the project folder and run:

```bash
python -m pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Run the project

### Produce both exact thesis outputs and newly trained outputs

```bash
python main.py --mode both
```

### Produce only the exact Chapter 4 reported outputs

```bash
python main.py --mode reported
```

### Retrain all models and generate fresh outputs

```bash
python main.py --mode retrain
```

## Exact thesis results

The following values are written to `outputs/table_8_model_comparison_reported.csv`:

| Rank | Model | MAE | RMSE | MAPE (%) |
|---:|---|---:|---:|---:|
| 1 | XGBoost | 245.81 | 317.28 | 6.53 |
| 2 | LSTM | 334.93 | 476.36 | 8.06 |
| 3 | SARIMA | 566.30 | 716.08 | 14.85 |
| 4 | Seasonal Naive | 981.45 | 1,162.32 | 26.29 |

The exact graphs included in the thesis are preserved in:

```text
figures/reference/
figures/report_exact/
```

Fresh model-generated graphs are written to:

```text
figures/generated/
```

## Why two sets of results are provided

The thesis reports final metrics and figures but does not document every original library version, neural-network setting, and tuning decision. SARIMA and Seasonal Naive are fully reproducible from the stated design and normally match the report to rounding precision. XGBoost and LSTM can vary slightly when retrained because of software-version and optimisation differences.

For academic transparency:

- `report_exact` contains the exact final outputs used in Chapter 4.
- `generated` contains outputs produced by a fresh run of the supplied code.
- `outputs/model_comparison_retrained.csv` records the newly calculated metrics.
- `outputs/table_8_model_comparison_reported.csv` records the thesis values.

Do not replace one set with the other without explaining which version was used.

## Main output files

```text
outputs/table_1_dataset_structure.csv
outputs/table_2_selected_store_and_time_split.csv
outputs/table_3_train_test_split_store_1.csv
outputs/table_4_descriptive_statistics.csv
outputs/table_6_sales_by_promotion_and_day_of_week.csv
outputs/table_8_model_comparison_reported.csv
outputs/table_9_feature_importance_reported.csv
outputs/table_10_inventory_results_reported.csv
outputs/model_comparison_retrained.csv
outputs/feature_importance_retrained.csv
outputs/inventory_results_retrained.csv
outputs/forecast_predictions_retrained.csv
outputs/run_metadata.json
```

## Important methodological note

`test.csv` does not contain the `Sales` target. Therefore, it cannot be used to calculate forecast accuracy. In line with Chapter 4, the final 42 observed days in `train.csv` are used as the chronological evaluation set. The supplied `test.csv` is retained for dataset completeness and structural inspection.
