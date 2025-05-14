import pandas as pd

def feature_engineering(df):
    """
    Computes the differenced federal funds rate ('dfedrate') to remove the trend in the series.
    Returns the updated DataFrame.
    """
    # Task 10: Compute differenced federal funds rate
    df['dfedrate'] = df['fedrate'].diff()

    return df

if __name__ == "__main__":
    from import_data import import_data
    from merge_data import merge_data
    from eda import perform_eda
    from stationarity_test import test_stationarity
    unemp_q, inflat_q, fedfund_q = import_data()
    df = merge_data(unemp_q, inflat_q, fedfund_q)
    df = perform_eda(df)
    df = test_stationarity(df)
    df = feature_engineering(df)
    print("Feature engineering completed")