from scripts.import_data import import_data
from scripts.merge_data import merge_data
from scripts.eda import perform_eda
from scripts.stationarity_test import test_stationarity
from scripts.feature_engineering import feature_engineering
from scripts.stationarity_check import check_stationarity
from scripts.explore_var_lags import explore_var_lags
from scripts.compare_var_lags import compare_var_lags
from scripts.pacf_analysis import pacf_analysis
from scripts.split_and_model import split_and_model
from scripts.forecast_visualizations import forecast_visualizations
from scripts.invert_transformation import invert_transformation
from scripts.evaluate_forecast import evaluate_forecast
from scripts.granger_causality import granger_causality
from scripts.conclusion import write_conclusion

def main():
    """
    Orchestrates the entire pipeline in memory, saving only final outputs to results/.
    """
    # Task 1-4: Import data
    unemp_q, inflat_q, fedfund_q = import_data()

    # Task 5-7: Merge data and compute unempgr
    df = merge_data(unemp_q, inflat_q, fedfund_q)

    # Task 8: EDA (visualizations saved to disk)
    df = perform_eda(df)

    # Task 9: Test stationarity of unempgr, fedrate, inflat (results saved to disk)
    df = test_stationarity(df)

    # Task 10: Feature engineering (compute dfedrate)
    df = feature_engineering(df)

    # Task 11-12: Stationarity check for dfedrate, visualization (stationarity results and plots saved to disk)
    stationary_df = check_stationarity(df)

    # Task 13-14: Explore optimal lags, build VAR with 5 lags (outputs saved to disk)
    vardf = explore_var_lags(stationary_df)

    # Task 14: Compare VAR models with different lags, build final VAR with 3 lags (outputs saved to disk)
    final_results = compare_var_lags(vardf, lags_list=[3, 4, 5, 6, 8])

    # Task 15: Generate PACF plots, conclude on 3 lags (plots and comments saved to disk)
    vardf = pacf_analysis(vardf)

    # Task 16: Split data, run VAR model with 3 lags, forecast
    train, test, df_fc = split_and_model(vardf)

    # Task 17: Visualize forecasts vs actuals (plots saved to disk)
    train, test, df_fc = forecast_visualizations(train, test, df_fc)

    # Task 18: Invert differencing to compute forecasted fedrate
    train, test, df_fc = invert_transformation(train, test, df_fc, df)

    # Task 19: Evaluate the forecast (metrics saved to disk)
    train, test, df_fc = evaluate_forecast(train, test, df_fc)

    # Task 20: Perform Granger Causality tests (results saved to disk)
    train = granger_causality(train)

    # Task 21: Write conclusion (saved to disk)
    write_conclusion()

    print("Pipeline completed. All outputs saved in results/")

if __name__ == "__main__":
    main()