from scripts.import_data import import_data
from scripts.merge_data import merge_data
from scripts.eda import perform_eda
from scripts.feature_engineering import feature_engineering
from scripts.stationarity_check import check_stationarity
from scripts.var_model import fit_var_model
from scripts.compare_var_lags import compare_var_lags
from scripts.analysis import perform_analysis

def main():
    """
    Orchestrates the entire pipeline in memory, saving only final outputs to results/.
    """
    # Task 1-4: Import data
    unemp_q, inflat_q, fedfund_q = import_data()

    # Task 5-7: Merge data
    df = merge_data(unemp_q, inflat_q, fedfund_q)

    # Task 8, 15: EDA (visualizations saved to disk)
    df = perform_eda(df)

    # Task 10: Feature engineering
    df = feature_engineering(df)

    # Task 9, 11, 12: Stationarity check (stationarity results and plots saved to disk)
    stationary_df = check_stationarity(df)

    # Task 13, 14, 16: Fit VAR model
    results, train, test, df_fc = fit_var_model(stationary_df)

    # Task 14: Compare VAR models with different lags (results saved to disk)
    compare_var_lags(train, lags_list=[3, 4, 5, 6, 8])

    # Task 17-20: Analysis (forecast visualizations, metrics, Granger Causality, conclusion saved to disk)
    perform_analysis(results, train, test, df_fc, df)

    print("Pipeline completed. All outputs saved in results/")

if __name__ == "__main__":
    main()
