import pandas as pd

def feature_engineering(df):
    """
    Computes unemployment growth rate and differenced federal funds rate.
    Returns the updated DataFrame.
    """
    # Task 10: Compute unemployment growth rate and differenced federal funds rate
    df['unempgr'] = df['UNRATE'].pct_change() * 100  # Already computed in import_data.py, but ensure consistency
    df['dfedrate'] = df['fedrate'].diff()

    return df

if __name__ == "__main__":
    from import_data import import_data
    from merge_data import merge_data
    unemp_q, inflat_q, fedfund_q = import_data()
    df = merge_data(unemp_q, inflat_q, fedfund_q)
    df = feature_engineering(df)
    print("Feature engineering completed")