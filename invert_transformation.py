import pandas as pd

def invert_transformation(train, test, df_fc, df):
    """
    Inverts the differencing transformation to compute the forecasted fedrate.
    Returns the updated train, test, and df_fc DataFrames.
    """
    # Task 18: Invert the transformation (differencing) to get the real forecast
    # Add the original federal funds rate ('fedrate') to the training set
    train['fedrate'] = df.fedrate[1:-8]

    # Add the original federal funds rate ('fedrate') to the test set
    test['fedrate'] = df.fedrate[-8:]

    # Convert the forecasted differenced federal funds rate ('dfedrate') back to the actual federal funds rate ('fedrate')
    df_fc['fedrate'] = train.fedrate.iloc[-1] + df_fc.dfedrate.cumsum()

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
    from forecast_visualizations import forecast_visualizations
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
    print("Transformation inversion completed")