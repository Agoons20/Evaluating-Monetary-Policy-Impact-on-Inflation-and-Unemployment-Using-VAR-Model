import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import os

def perform_eda(df):
    """
    Visualizes the time series and generates PACF plots for the variables.
    Saves plots to results/plots/.
    Returns the DataFrame unchanged.
    """
    # Ensure the plots directory exists
    os.makedirs("results/plots", exist_ok=True)

    # Task 8: Generate summary statistics and time series plot
    summary_stats = df.describe()
    with open("results/summary_stats.txt", "w") as f:
        f.write(summary_stats.to_string())

    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(df['unempgr'], label='Unemployment Growth Rate (%)', color='blue')
    plt.plot(df['fedrate'], label='Federal Funds Rate (%)', color='green')
    plt.plot(df['inflat'], label='Inflation Rate (%)', color='red')
    plt.title('Time Series of Unemployment Growth, Federal Funds, and Inflation Rates (1970–2019)')
    plt.xlabel('Date')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("results/plots/time_series_plot.png")
    plt.close()

    # Task 15: Plot PACF for each variable
    for column in ['unempgr', 'fedrate', 'inflat']:
        plt.figure(figsize=(10, 4), dpi=100)
        sgt.plot_pacf(df[column].dropna(), lags=10, method='ywm')
        plt.title(f'PACF of {column}')
        plt.savefig(f"results/plots/pacf_{column}.png")
        plt.close()

    return df

if __name__ == "__main__":
    from import_data import import_data
    from merge_data import merge_data
    unemp_q, inflat_q, fedfund_q = import_data()
    df = merge_data(unemp_q, inflat_q, fedfund_q)
    perform_eda(df)
    print("EDA completed")