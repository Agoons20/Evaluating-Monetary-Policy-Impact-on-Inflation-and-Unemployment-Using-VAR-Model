import pandas as pd
import statsmodels.tsa.stattools as sts
import os

def test_stationarity(df):
    """
    Tests stationarity for unempgr, fedrate, and inflat using ADF tests.
    Saves results to results/analysis/stationarity_results.txt.
    Returns the DataFrame unchanged.
    """
    # Ensure the analysis directory exists
    os.makedirs("results/analysis", exist_ok=True)

    # Task 9: Test stationarity using specified ADF commands
    variables = {
        'unempgr': df['unempgr'][1:],
        'fedrate': df['fedrate'][1:],
        'inflat': df['inflat'][1:]
    }
    with open("results/analysis/stationarity_results.txt", "w") as f:
        f.write("Initial Stationarity Check (ADF Test Results):\n\n")
        for name, series in variables.items():
            result = sts.adfuller(series)  
            f.write(f"ADF Test for {name}:\n")
            f.write(f"ADF Statistic: {result[0]:.4f}\n")
            f.write(f"p-value: {result[1]:.4f}\n")
            f.write(f"Critical Values: {result[4]}\n")
            if result[1] < 0.05:
                f.write(f"{name} is stationary (p < 0.05)\n\n")
            else:
                f.write(f"{name} is not stationary (p >= 0.05)\n\n")

    return df

if __name__ == "__main__":
    from import_data import import_data
    from merge_data import merge_data
    from eda import perform_eda
    unemp_q, inflat_q, fedfund_q = import_data()
    df = merge_data(unemp_q, inflat_q, fedfund_q)
    df = perform_eda(df)
    df = test_stationarity(df)
    print("Initial stationarity test completed")
