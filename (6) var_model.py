import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
import os

def fit_var_model(data):
    """
    Builds a VAR model, selects lags, splits data into train/test sets.
    Returns the fitted VAR model, training data, test data, and forecast DataFrame.
    Saves model outputs to results/models/.
    """
    # Ensure the models directory exists
    os.makedirs("results/models", exist_ok=True)

    # Task 13: Create DataFrame with unempgr, dfedrate, and inflat for VAR model
    var_data = data[['unempgr', 'dfedrate', 'inflat']]

    # Task 14: Function to select optimal lag length
    def select_var_lags(data, maxlags=8, ic='aic'):
        ic_values = {}
        for lag in range(1, maxlags + 1):
            try:
                model = VAR(data).fit(lag)
                ic_values[lag] = getattr(model, ic)
            except Exception as e:
                print(f"Error fitting VAR with {lag} lags: {e}")
                continue
        optimal_lag = min(ic_values, key=ic_values.get)
        return optimal_lag

    # Task 16: Split data into training and test sets (96% train, 4% test)
    total_rows = len(var_data)
    test_size = int(total_rows * 0.04)  # 4% for test (≈8 quarters)
    train = var_data.iloc[:-test_size]
    test = var_data.iloc[-test_size:]

    # Task 14: Select optimal lag length using AIC
    optimal_lag_aic = select_var_lags(train, maxlags=8, ic='aic')
    with open("results/models/lag_selection.txt", "w") as f:
        f.write("Lag Selection Results:\n")
        f.write(f"Optimal lag length based on AIC: {optimal_lag_aic}\n")
        f.write("Note: compare_var_lags.py suggested 5 lags, but we chose 3 lags based on highest log likelihood and lowest residual correlation.\n")

    # Fit VAR model with chosen lags (3 lags based on task 14)
    model = VAR(train)
    results = model.fit(maxlags=3)

    # Task 13: Save VAR model summary and interpretation
    with open("results/models/var_model_summary.txt", "w") as f:
        f.write(str(results.summary()))
    with open("results/models/var_model_interpretation.txt", "w") as f:
        f.write("VAR Model Interpretation:\n\n")
        for var in results.names:
            f.write(f"Equation for {var}:\n")
            for lag in range(1, 4):
                for exog_var in results.names:
                    coef = results.params[f"L{lag}.{exog_var}"][var]
                    pval = results.pvalues[f"L{lag}.{exog_var}"][var]
                    f.write(f"  L{lag}.{exog_var}: Coefficient = {coef:.4f}, p-value = {pval:.4f}\n")
            f.write("\n")

    # Generate forecasts for the test period
    fcinput = train.values[-3:]  # Last 3 rows for lags=3
    forecast = results.forecast(fcinput, steps=len(test))
    df_fc = pd.DataFrame(forecast, index=test.index, columns=train.columns)

    return results, train, test, df_fc

if __name__ == "__main__":
    from import_data import import_data
    from merge_data import merge_data
    from feature_engineering import feature_engineering
    from stationarity_check import check_stationarity
    unemp_q, inflat_q, fedfund_q = import_data()
    df = merge_data(unemp_q, inflat_q, fedfund_q)
    df = feature_engineering(df)
    stationary_df = check_stationarity(df)
    results, train, test, df_fc = fit_var_model(stationary_df)
    print("VAR model fitted successfully")