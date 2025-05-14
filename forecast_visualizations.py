import pandas as pd
import matplotlib.pyplot as plt

def forecast_visualizations(train, test, df_fc):
    """
    Visualizes forecasts vs actuals for dfedrate, inflat, and unempgr.
    Saves plots to results/plots/.
    Returns the input DataFrames unchanged.
    """
    # Task 17: Visualize forecasts vs actuals for dfedrate
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train.dfedrate[-40:], label='training')
    plt.plot(test.dfedrate, label='actual')
    plt.plot(df_fc.dfedrate, label='forecast')
    plt.title('Forecast vs Actuals for Differenced Federal Funds Rate')
    plt.legend(loc='upper left', fontsize=10)
    plt.savefig("results/plots/forecast_vs_actuals_dfedrate.png")
    plt.close()

    # Visualize forecasts vs actuals for inflat
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train.inflat[-40:], label='training')
    plt.plot(test.inflat, label='actual')
    plt.plot(df_fc.inflat, label='forecast')
    plt.title('Forecast vs Actuals for Inflation')
    plt.legend(loc='upper left', fontsize=10)
    plt.savefig("results/plots/forecast_vs_actuals_inflat.png")
    plt.close()

    # Visualize forecasts vs actuals for unempgr
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train.unempgr[-40:], label='training')
    plt.plot(test.unempgr, label='actual')
    plt.plot(df_fc.unempgr, label='forecast')
    plt.title('Forecast vs Actuals for Unemployment Growth Rate')
    plt.legend(loc='upper left', fontsize=10)
    plt.savefig("results/plots/forecast_vs_actuals_unempgr.png")
    plt.close()

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
    print("Forecast visualizations completed")