import pandas as pd
from pandas_datareader import data
from datetime import datetime

def import_data():
    """
    Fetches data from the federal reserve economic Data (FRED) and resamples to quarterly frequency.
    Returns three DataFrames: unempdata, inflatdata, fedfunddata.
    ".resample('3M', axis=0)": Groups the data into 3-month (quarterly) intervals along the rows (axis=0).
    ".last()": Takes the last value in each 3-month period as the quarterly value

    """
    # Define date range
    start = datetime(1970, 1, 1)
    end = datetime(2019, 12, 31)

    # Task 1-4: Download data from FRED
    # Unemployment data (UNRATE)
    unempdata = data.DataReader('UNRATE', data_source='fred', start=start, end=end)
    unemp_q = unempdata.resample('3ME').last()

    # Inflation data (FLEXCPIM679SFRBATL)
    inflatdata = data.DataReader('FLEXCPIM679SFRBATL', data_source='fred', start=start, end=end)
    inflat_q = inflatdata.resample('3ME').last()

    # Federal funds rate data (FEDFUNDS)
    fedfunddata = data.DataReader('FEDFUNDS', data_source='fred', start=start, end=end)
    fedfund_q = fedfunddata.resample('3ME').last()

    return unemp_q, inflat_q, fedfund_q

if __name__ == "__main__":
    unemp_q, inflat_q, fedfund_q = import_data()
    print("Data imported successfully")