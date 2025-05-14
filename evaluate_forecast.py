import pandas as pd
import numpy as np

def calculate_forecast_metrics(forecast, actual, columns=None):
    """
    Calculate forecast accuracy metrics for each variable in the forecasted and actual data.

    Parameters:
    - forecast: pandas DataFrame or NumPy array, forecasted values with shape (n, k)
    - actual: pandas DataFrame or NumPy array, actual values with the same shape as forecast
    - columns: list, optional, names of the variables (required if inputs are NumPy arrays)

    Returns:
    - dict: Dictionary with metrics (ME, MAE, MPE, MAPE, RMSE) for each variable
    """
    if isinstance(forecast, np.ndarray):
        if columns is None:
            raise ValueError("Column names must be provided if forecast is a NumPy array")
        forecast = pd.DataFrame(forecast, columns=columns)
    if isinstance(actual, np.ndarray):
        if columns is None:
            raise ValueError("Column names must be provided if actual is a NumPy array")
        actual = pd.DataFrame(actual, columns=columns)

    actual = actual[forecast.columns].copy()
    forecast_values = forecast.values
    actual_values = actual.values
    actual_values[actual_values == 0] = 0.0001

    me = np.mean(forecast_values - actual_values, axis=0)
    mae = np.mean(np.abs(forecast_values - actual_values), axis=0)
    mpe = np.mean((forecast_values - actual_values) / actual_values, axis=0)
    mape = np.mean(np.abs(forecast_values - actual_values) / np.abs(actual_values), axis=0)
    rmse = np.sqrt(np.mean((forecast_values - actual_values) ** 2, axis=0))

    metrics = {}
    for i, col in enumerate(forecast.columns):
        metrics[col] = {
            'ME': me[i],
            'MAE': mae[i],
            'MPE': mpe[i],
            'MAPE': mape[i],
            'RMSE': rmse[i]
        }
    return metrics

def evaluate_forecast(train, test, df_fc):
    """
    Evaluates the forecast using metrics and saves the results.
    Returns the input DataFrames unchanged.
    """
    # Task 19: Evaluate the forecast
    fc_subset = df_fc[['unempgr', 'dfedrate', 'inflat']]
    test_subset = test[['unempgr', 'dfedrate', 'inflat']]
    metrics = calculate_forecast_metrics(fc_subset, test_subset)

    # Save forecast metrics (overwrite the previous file to update with final model metrics)
    with open("results/analysis/forecast_metrics.txt", "w") as f:
        f.write("Forecast Accuracy Metrics (Final Model with 3 Lags):\n\n")
        for var, values in metrics.items():
            f.write(f"{var}:\n")
            f.write(f"  ME: {values['ME']:.4f}\n")
            f.write(f"  MAE: {values['MAE']:.4f}\n")
            f.write(f"  MPE: {values['MPE']:.4f}\n")
            f.write(f"  MAPE: {values['MAPE']:.4f}\n")
            f.write(f"  RMSE: {values['RMSE']:.4f}\n\n")

    return train, test, df_fc

if __name__ == "__main__":
    from import_data import import_data
    from merge_data import merge_data
    from eda import perform_eda
    from stationarity_test import test_stationarity
    from feature_engineering import feature_engineering
    from stationarity_check import check_stationarity
    from explore_var_lags import explore_var_lags
    from compare_var_lags import compare_var_lags
    from pacf_analysis import pacf_analysis
    from split_and_model import split_and_model
    from forecast_visualizations import forecast_visualizations
    from invert_transformation import invert_transformation
    unemp_q, inflat_q, fedfund_q = import_data()
    df = merge_data(unemp_q, inflat_q, fedfund_q)
    df = perform_eda(df)
    df = test_stationarity(df)
    df = feature_engineering(df)
    stationary_df = check_stationarity(df)
    vardf = explore_var_lags(stationary_df)
    final_results = compare_var_lags(vardf, lags_list=[3, 4, 5, 6, 8])
    vardf = pacf_analysis(vardf)
    train, test, df_fc = split_and_model(vardf)
    train, test, df_fc = forecast_visualizations(train, test, df_fc)
    train, test, df_fc = invert_transformation(train, test, df_fc, df)
    train, test, df_fc = evaluate_forecast(train, test, df_fc)
    print("Forecast evaluation completed")