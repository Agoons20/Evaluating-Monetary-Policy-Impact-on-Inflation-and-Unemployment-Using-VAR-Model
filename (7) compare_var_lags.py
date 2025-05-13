import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
import os

def compare_var_lags(data, lags_list=[3, 4, 5, 6, 8]):
    """
    Compares VAR models with different lags, saves results to file.
    Returns None (output is saved to file).
    """
    # Ensure the models directory exists
    os.makedirs("results/models", exist_ok=True)

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

    # Write results to file
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

    return None

if __name__ == "__main__":
    from import_data import import_data
    from merge_data import merge_data
    from feature_engineering import feature_engineering
    from stationarity_check import check_stationarity
    unemp_q, inflat_q, fedfund_q = import_data()
    df = merge_data(unemp_q, inflat_q, fedfund_q)
    df = feature_engineering(df)
    stationary_df = check_stationarity(df)
    compare_var_lags(stationary_df, lags_list=[3, 4, 5, 6, 8])
    print("Lag comparison completed")