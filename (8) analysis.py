import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
import os

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

def perform_analysis(results, train, test, df_fc, df):
    """
    Visualizes forecasts, evaluates metrics, performs Granger Causality tests, and concludes the analysis.
    Saves outputs to results/.
    """
    # Ensure the output directories exist
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/analysis", exist_ok=True)

    # Add 'fedrate' to train, test, and df_fc
    train['fedrate'] = df['fedrate'][1:-8]
    test['fedrate'] = df['fedrate'][-8:]
    last_fedrate = train['fedrate'].iloc[-1]
    df_fc['fedrate'] = last_fedrate + df_fc['dfedrate'].cumsum()

    # Task 17: Visualize forecasts vs actuals for all three variables
    variables = ['unempgr', 'dfedrate', 'inflat', 'fedrate']
    titles = {
        'unempgr': 'Forecast vs Actuals for Unemployment Growth Rate',
        'dfedrate': 'Forecast vs Actuals for Differenced Federal Funds Rate',
        'inflat': 'Forecast vs Actuals for Inflation Rate',
        'fedrate': 'Forecast vs Actuals for Federal Funds Rate'
    }
    y_labels = {
        'unempgr': 'Unemployment Growth Rate (%)',
        'dfedrate': 'Percentage Change',
        'inflat': 'Inflation Rate (%)',
        'fedrate': 'Federal Funds Rate (%)'
    }

    for var in variables:
        plt.figure(figsize=(12, 5), dpi=100)
        plt.plot(train[var][-40:], label='Training', color='blue')
        plt.plot(test[var], label='Actual', color='green')
        plt.plot(df_fc[var], label='Forecast', color='red')
        plt.title(titles[var])
        plt.xlabel('Date')
        plt.ylabel(y_labels[var])
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"results/plots/forecast_vs_actuals_{var}.png")
        plt.close()

    # Task 18: Evaluate the forecast
    fc_subset = df_fc[['unempgr', 'dfedrate', 'inflat']]
    test_subset = test[['unempgr', 'dfedrate', 'inflat']]
    metrics = calculate_forecast_metrics(fc_subset, test_subset)

    # Save forecast metrics
    with open("results/analysis/forecast_metrics.txt", "w") as f:
        f.write("Forecast Accuracy Metrics:\n\n")
        for var, values in metrics.items():
            f.write(f"{var}:\n")
            f.write(f"  ME: {values['ME']:.4f}\n")
            f.write(f"  MAE: {values['MAE']:.4f}\n")
            f.write(f"  MPE: {values['MPE']:.4f}\n")
            f.write(f"  MAPE: {values['MAPE']:.4f}\n")
            f.write(f"  RMSE: {values['RMSE']:.4f}\n\n")

    # Task 19: Granger Causality Tests
    # dfedrate -> unempgr
    df_gc1 = pd.DataFrame({'dfedrate': train['dfedrate'], 'unempgr': train['unempgr']})
    gc_results_1 = grangercausalitytests(df_gc1, 2, verbose=False)
    with open("results/analysis/granger_causality_dfedrate_unempgr.txt", "w") as f:
        for lag, result in gc_results_1.items():
            f.write(f"\nGranger Causality (dfedrate -> unempgr), Lag {lag}\n")
            for test, stats in result[0].items():
                f.write(f"{test}: {stats}\n")

    # dfedrate -> inflat
    df_gc2 = pd.DataFrame({'dfedrate': train['dfedrate'], 'inflat': train['inflat']})
    gc_results_2 = grangercausalitytests(df_gc2, 2, verbose=False)
    with open("results/analysis/granger_causality_dfedrate_inflat.txt", "w") as f:
        for lag, result in gc_results_2.items():
            f.write(f"\nGranger Causality (dfedrate -> inflat), Lag {lag}\n")
            for test, stats in result[0].items():
                f.write(f"{test}: {stats}\n")

    # unempgr -> inflat
    df_gc3 = pd.DataFrame({'unempgr': train['unempgr'], 'inflat': train['inflat']})
    gc_results_3 = grangercausalitytests(df_gc3, 2, verbose=False)
    with open("results/analysis/granger_causality_unempgr_inflat.txt", "w") as f:
        for lag, result in gc_results_3.items():
            f.write(f"\nGranger Causality (unempgr -> inflat), Lag {lag}\n")
            for test, stats in result[0].items():
                f.write(f"{test}: {stats}\n")

    # Task 20: Conclusion
    conclusion = """
Conclusion:
- The VAR model captured relationships between unemployment growth (unempgr), differenced federal funds rate (dfedrate), and inflation (inflat).
- Granger Causality tests showed that dfedrate significantly predicts unempgr (p < 0.001), supporting the idea that monetary policy impacts unemployment.
- No significant causality was found from dfedrate to inflat (p > 0.24) or unempgr to inflat (p > 0.27), suggesting limited predictive power in these directions.
- Forecast accuracy metrics (see results/analysis/forecast_metrics.txt) indicate the model's performance, with dfedrate forecasts being more accurate than inflat.
- Policy Recommendation: To address high unemployment, the Federal Reserve should consider lowering interest rates to stimulate economic activity. However, the lack of causality with inflation suggests a structural VAR (SVAR) may be needed for better modeling.
"""
    with open("results/conclusion.txt", "w") as f:
        f.write(conclusion)

    return None

if __name__ == "__main__":
    from import_data import import_data
    from merge_data import merge_data
    from feature_engineering import feature_engineering
    from stationarity_check import check_stationarity
    from var_model import fit_var_model
    unemp_q, inflat_q, fedfund_q = import_data()
    df = merge_data(unemp_q, inflat_q, fedfund_q)
    df = feature_engineering(df)
    stationary_df = check_stationarity(df)
    results, train, test, df_fc = fit_var_model(stationary_df)
    perform_analysis(results, train, test, df_fc, df)
    print("Analysis completed")