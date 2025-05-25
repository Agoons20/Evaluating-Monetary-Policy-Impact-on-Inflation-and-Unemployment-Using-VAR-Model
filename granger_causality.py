from statsmodels.tsa.stattools import grangercausalitytests
import os

def granger_causality(df):
    """
    Performs Granger Causality tests for dfedrate -> unempgr, dfedrate -> inflat, and unempgr -> inflat on the full dataset.
    Saves results to results/analysis/.
    Returns the DataFrame unchanged
    """
    # Ensure the analysis directory exists
    os.makedirs("results/analysis", exist_ok=True)

    # Test if dfedrate Granger-causes unempgr
    df_gc = pd.DataFrame({'dfedrate': df.dfedrate[1:], 'unempgr': df.unempgr[1:]})
    gc_results = grangercausalitytests(df_gc, 3)
    with open("results/analysis/granger_causality_dfedrate_unempgr.txt", "w") as f:
        for lag, result in gc_results.items():
            f.write(f"\nGranger Causality (dfedrate -> unempgr), Lag {lag}\n")
            for test, stats in result[0].items():
                f.write(f"{test}: {stats}\n")

    # Test if dfedrate Granger-causes inflat
    df_gc2 = pd.DataFrame({'dfedrate': df.dfedrate[1:], 'inflat': df.inflat[1:]})
    gc_results = grangercausalitytests(df_gc2, 3)
    with open("results/analysis/granger_causality_dfedrate_inflat.txt", "w") as f:
        for lag, result in gc_results.items():
            f.write(f"\nGranger Causality (dfedrate -> inflat), Lag {lag}\n")
            for test, stats in result[0].items():
                f.write(f"{test}: {stats}\n")

    # Test if unemployment growth rate Granger-causes inflation
    df_gc3 = pd.DataFrame({'unempgr': df.unempgr[1:], 'inflat': df.inflat[1:]})
    gc_results = grangercausalitytests(df_gc3, 3)
    with open("results/analysis/granger_causality_unempgr_inflat.txt", "w") as f:
        for lag, result in gc_results.items():
            f.write(f"\nGranger Causality (unempgr -> inflat), Lag {lag}\n")
            for test, stats in result[0].items():
                f.write(f"{test}: {stats}\n")

    return df

if __name__ == "__main__":
    import pandas as pd
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
    from evaluate_forecast import evaluate_forecast
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
    df = granger_causality(df)
    print("Granger Causality tests completed")