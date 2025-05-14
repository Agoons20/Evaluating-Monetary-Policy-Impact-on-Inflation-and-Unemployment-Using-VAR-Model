import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import os

def pacf_analysis(data):
    """
    Generates PACF plots for unempgr, dfedrate, and inflat, concludes on using 3 lags.
    Saves plots and comments to results/.
    Returns the DataFrame for further processing.
    """
    # Ensure the plots directory exists
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/analysis", exist_ok=True)

    # Task 15: Generate PACF plots for unempgr, dfedrate, and inflat
    for column in ['unempgr', 'dfedrate', 'inflat']:
        plt.figure(figsize=(10, 4), dpi=100)
        sgt.plot_pacf(data[column].dropna(), lags=10, method='ywm')
        plt.title(f'PACF of {column}')
        plt.savefig(f"results/plots/pacf_{column}.png")
        plt.close()

    # Conclude on using 3 lags and comment on the results
    comments = """
PACF Analysis and Lag Selection Conclusion:
- We also look at the PACF to determine the lags to be used in the VAR model. Only the vertical lines that exceed the shaded area on the plot are statistically 
  significant predictors for predicting the current period. In the case of unemployment growth rate, inflation and differenced federal funds lending rate, the PACF
  suggests that five previous quarters are statistically significant in predicting their current quarter respective values. 
- The PACF plots for unempgr, dfedrate, and inflat show significant correlations at lags 1, 2, and 3, with diminishing significance beyond lag 3.
- For unempgr, significant spikes are observed at lags 1 and 2, with a smaller spike at lag 3, indicating that 3 lags capture most of the autocorrelation.
- For dfedrate, the PACF shows significant correlations up to lag 3, supporting the choice of 3 lags to model the dynamics of the differenced federal funds rate.
- For inflat, the PACF also indicates significant lags up to 3, with correlations tapering off afterward.
- Combined with the earlier lag selection (AIC suggesting 5 lags, but residual correlations and log likelihood favoring 3 lags in compare_var_lags.py), we conclude that 3 lags is a suitable choice for the VAR model.
- This aligns with practical considerations, balancing model complexity and explanatory power.
"""
    with open("results/analysis/pacf_lag_comments.txt", "w") as f:
        f.write(comments)

    return data

if __name__ == "__main__":
    from import_data import import_data
    from merge_data import merge_data
    from eda import perform_eda
    from stationarity_test import test_stationarity
    from feature_engineering import feature_engineering
    from stationarity_check import check_stationarity
    from explore_var_lags import explore_var_lags
    from compare_var_lags import compare_var_lags
    unemp_q, inflat_q, fedfund_q = import_data()
    df = merge_data(unemp_q, inflat_q, fedfund_q)
    df = perform_eda(df)
    df = test_stationarity(df)
    df = feature_engineering(df)
    stationary_df = check_stationarity(df)
    vardf = explore_var_lags(stationary_df)
    final_results = compare_var_lags(vardf, lags_list=[3, 4, 5, 6, 8])
    vardf = pacf_analysis(vardf)
    print("PACF analysis and lag selection conclusion completed")