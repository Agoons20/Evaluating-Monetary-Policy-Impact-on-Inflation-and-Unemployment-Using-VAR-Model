import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
import os

def compare_var_lags(data, lags_list=[3, 4, 5, 6, 8]):
    """
    Compares VAR models with different lags, builds a final VAR model with 3 lags, saves results to file.
    Returns the final VAR model results.
    """
    # Ensure the models directory exists
    os.makedirs("results/models", exist_ok=True)

    # Task 14: Compare VAR models with different lags
    results = {}
    for lags in lags_list:
        try:
            model = VAR(data).fit(maxlags=lags)
            log_likelihood = model.llf
            residual_corr = pd.DataFrame(model.resid_corr, index=data.columns, columns=data.columns)
            corr_values = residual_corr.where(np.triu(np.ones(residual_corr.shape), k=1).astype(bool)).stack()
            results[lags] = {
                'log_likelihood': log_likelihood,
                'residual_correlations': corr_values.to_dict()
            }
        except Exception as e:
            print(f"Error fitting VAR with maxlags={lags}: {e}")
            continue

    # Write comparison results to file
    with open("results/models/var_lags_comparison.txt", "w") as f:
        f.write("VAR Model Comparison Across Different maxlags Values\n")
        f.write("=" * 50 + "\n\n")
        for lags, metrics in results.items():
            f.write(f"maxlags = {lags}\n")
            f.write(f"Log Likelihood: {metrics['log_likelihood']}\n")
            f.write("Correlation of Residuals (Upper Triangle):\n")
            for (var1, var2), corr in metrics['residual_correlations'].items():
                f.write(f"  {var1} - {var2}: {corr:.6f}\n")
            f.write("\n")

    # Build final VAR model with 3 lags
    final_model = VAR(data)
    final_results = final_model.fit(maxlags=3)
    with open("results/models/var_model_summary_3_lags.txt", "w") as f:
        f.write(str(final_results.summary()))
    with open("results/models/var_model_interpretation_3_lags.txt", "w") as f:
        f.write("VAR Model Interpretation (3 lags):\n\n")
        for var in final_results.names:
            f.write(f"Equation for {var}:\n")
            for lag in range(1, 4):
                for exog_var in final_results.names:
                    coef = final_results.params[f"L{lag}.{exog_var}"][var]
                    pval = final_results.pvalues[f"L{lag}.{exog_var}"][var]
                    f.write(f"  L{lag}.{exog_var}: Coefficient = {coef:.4f}, p-value = {pval:.4f}\n")
            f.write("\n")

    return final_results

if __name__ == "__main__":
    from import_data import import_data
    from merge_data import merge_data
    from eda import perform_eda
    from stationarity_test import test_stationarity
    from feature_engineering import feature_engineering
    from stationarity_check import check_stationarity
    from explore_var_lags import explore_var_lags
    unemp_q, inflat_q, fedfund_q = import_data()
    df = merge_data(unemp_q, inflat_q, fedfund_q)
    df = perform_eda(df)
    df = test_stationarity(df)
    df = feature_engineering(df)
    stationary_df = check_stationarity(df)
    vardf = explore_var_lags(stationary_df)
    final_results = compare_var_lags(vardf, lags_list=[3, 4, 5, 6, 8])
    print("Lag comparison and final VAR model (3 lags) completed")