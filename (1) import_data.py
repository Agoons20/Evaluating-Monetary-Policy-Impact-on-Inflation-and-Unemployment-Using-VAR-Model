import pandas as pd
from pandas_datareader import data
from datetime import datetime
import os

# Ensure the raw data directory exists
os.makedirs("data/raw", exist_ok=True)

# Define date range
start = datetime(1970, 1, 1)
end = datetime(2019, 12, 31)

# Import unemployment data
unempdata = data.DataReader('UNRATE', data_source='fred', start=start, end=end)
unempdata.to_csv("data/raw/unempdata.csv")

# Import inflation data
inflatdata = data.DataReader('FLEXCPIM679SFRBATL', data_source='fred', start=start, end=end)
inflatdata.to_csv("data/raw/inflatdata.csv")

# Import federal funds rate data
fedfunddata = data.DataReader('FEDFUNDS', data_source='fred', start=start, end=end)
fedfunddata.to_csv("data/raw/fedfunddata.csv")

print("Data imported and saved to data/raw/")