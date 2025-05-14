def write_conclusion():
    """
    Writes the final conclusion based on the Granger Causality results.
    Saves the conclusion to results/conclusion.txt.
    """
    conclusion = """
### The results showed:

1) 'dfedrate' Granger-causes 'unempgr' (p-value < 0.001 for the 3 previous quarters), meaning past changes in the federal funds rate are useful for predicting unemployment growth rate. This supports the idea that monetary policy impacts unemployment. 

dfedrate Granger-causes unempgr (p = 0.0002 for lag 1, p = 0.0008 for lag 2, p = 0.0033 for lag 3), indicating that past changes in the federal funds rate help predict unemployment growth.

2) dfedrate does not Granger-cause inflat (p = 0.2425 for lag 1, p = 0.3878 for lag 2).
3) unempgr does not Granger-cause inflat (p = 0.2738 for lag 1, p = 0.3549 for lag 2).

### Conclusion 
The VAR model captured bidirectional relationships (e.g., differenced fedrate lending rates affects unemployment growth rate, and unemployment growth rate affects differenced fedrate lending rates), but it doesn’t explicitly test which direction is statistically significant. Granger Causality tests provide this directional insight showing that differenced fedrate lending rates Granger-causes unemployment growth rate (p < 0.001), meaning past changes in the federal funds rate are useful for predicting unemployment growth. This supports the idea that monetary policy impacts unemployment. Thus, to address high levels of unemployment, the Federal Reserve should consider lowering interest rates or increasing the money supply to stimulate spending and encourage hiring.

The lack of Granger Causality from differenced fedrate lending rates to inflation challenges the recommendation to raise rates to combat inflation, as the model suggests limited predictive power in this direction. This finding aligns with the VAR results, where differenced federal lending rate unexpectedly increased inflation, prompting the need for a structural VAR (SVAR) to capture contemporaneous effects.
"""
    with open("results/conclusion.txt", "w") as f:
        f.write(conclusion)

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
    from evaluate_forecast import evaluate_forecast
    from granger_causality import granger_causality
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
    train = granger_causality(train)
    write_conclusion()
    print("Conclusion written")