import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
import random

def split_and_model(vardf):
    """
    Splits the data into training and test sets, runs a VAR model with 3 lags, and generates forecasts.
    Returns the training DataFrame, test DataFrame, and forecast DataFrame.
    """
    # Task 16: Split the data
    # ':-8' selects all rows except the last 8 quarters (approximately 2 years) for training
    train = vardf[:-8]

    # '-8:' selects the last 8 quarters as the test set to evaluate the model's forecasting performance
    test = vardf[-8:]

    # Initialize and fit the VAR model
    random.seed(1)

    # Initialize a VAR model using the training data
    model = VAR(train)

    # Define the maximum number of lags to include in the VAR model
    lags = 3

    # Fit the VAR model with the specified number of lags
    results = model.fit(maxlags=lags)

    # Generate forecasts
    # Use the last 3 quarters to predict the current quarter
    fcinput = train.values[-1 * lags:]
    fc = results.forecast(y=fcinput, steps=8)

    # Convert the forecast array into a pandas DataFrame
    df_fc = pd.DataFrame(fc, index=test.index, columns=test.columns)

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
    print("Data splitting and VAR modeling completed")