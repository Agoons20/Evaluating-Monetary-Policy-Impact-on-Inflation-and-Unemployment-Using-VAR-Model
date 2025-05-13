import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import os

def check_stationarity(df):
    """
    Tests stationarity for each variable, visualizes fedrate vs dfedrate, and returns a DataFrame with stationary variables.
    Saves stationarity results and plots to results/.
    """
    # Ensure the plots directory exists
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/analysis", exist_ok=True)

    # Task 9: Test stationarity for each variable (fedrate, unempgr, inflat)
    variables = {'unempgr': df['unempgr'], 'fedrate': df['fedrate'], 'inflat': df['inflat']}
    stationary_vars = {}
    with open("results/analysis/stationarity_results.txt", "w") as f:
        f.write("Stationarity Check (ADF Test Results):\n\n")
        for name, series in variables.items():
            result = adfuller(series.dropna())
            f.write(f"ADF Test for {name}:\n")
            f.write(f"ADF Statistic: {result[0]:.4f}\n")
            f.write(f"p-value: {result[1]:.4f}\n")
            f.write(f"Critical Values: {result[4]}\n")
            if result[1] < 0.05:
                f.write(f"{name} is stationary (p < 0.05)\n\n")
                stationary_vars[name] = series
            else:
                f.write(f"{name} is not stationary (p >= 0.05)\n\n")
                if name == 'fedrate':
                    # Task 11: Test dfedrate for stationarity
                    dfedrate = df['dfedrate']
                    result_dfed = adfuller(dfedrate.dropna())
                    f.write(f"ADF Test for dfedrate (differenced fedrate):\n")
                    f.write(f"ADF Statistic: {result_dfed[0]:.4f}\n")
                    f.write(f"p-value: {result_dfed[1]:.4f}\n")
                    f.write(f"Critical Values: {result_dfed[4]}\n")
                    if result_dfed[1] < 0.05:
                        f.write("dfedrate is stationary (p < 0.05)\n\n")
                        stationary_vars['dfedrate'] = dfedrate
                    else:
                        f.write("dfedrate is not stationary (p >= 0.05)\n\n")

    # Task 12: Visualize fedrate and dfedrate to confirm trend removal
    plt.figure(figsize=(12, 5), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(df['fedrate'], label='Federal Funds Rate (%)', color='green')
    plt.title('Federal Funds Rate (Before Differencing)')
    plt.xlabel('Date')
    plt.ylabel('Percentage')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(1, 2, 2)
    plt.plot(df['dfedrate'], label='Differenced Federal Funds Rate (%)', color='purple')
    plt.title('Differenced Federal Funds Rate')
    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/plots/fedrate_comparison.png")
    plt.close()

    # Create a new DataFrame with stationary variables
    stationary_df = pd.DataFrame(stationary_vars, index=df.index)
    stationary_df = stationary_df.dropna()

    return stationary_df

if __name__ == "__main__":
    from import_data import import_data
    from merge_data import merge_data
    from feature_engineering import feature_engineering
    unemp_q, inflat_q, fedfund_q = import_data()
    df = merge_data(unemp_q, inflat_q, fedfund_q)
    df = feature_engineering(df)
    stationary_df = check_stationarity(df)
    print("Stationarity check completed")