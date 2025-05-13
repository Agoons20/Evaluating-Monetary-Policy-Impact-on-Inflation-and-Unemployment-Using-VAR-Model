import pandas as pd
import os

def merge_data(unemp_q, inflat_q, fedfund_q):
    """
    Merges the datasets, checks for missing values and duplicates.
    Returns the merged DataFrame.
    """
    # Ensure the analysis directory exists for outputs
    os.makedirs("results/analysis", exist_ok=True)

    # Standardize index to month-end timestamps
    unemp_q.index = unemp_q.index.to_period('M').to_timestamp('M')

    # Merge datasets
    df = unemp_q.copy()
    df['fedrate'] = fedfund_q['FEDFUNDS']
    df['inflat'] = inflat_q['FLEXCPIM679SFRBATL']

    # Task 5: Check for missing values
    missing_values = df.isna().sum()
    with open("results/analysis/missing_values.txt", "w") as f:
        f.write("Missing Values:\n")
        for col, count in missing_values.items():
            f.write(f"{col}: {count}\n")

    # Task 6: Check for duplicate entries
    duplicates = df.index.duplicated().sum()
    with open("results/analysis/duplicates.txt", "w") as f:
        f.write(f"Number of Duplicate Entries: {duplicates}\n")

    return df

if __name__ == "__main__":
    from import_data import import_data
    unemp_q, inflat_q, fedfund_q = import_data()
    df = merge_data(unemp_q, inflat_q, fedfund_q)
    print("Data merged successfully")