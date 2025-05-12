{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c39765c0-d23f-418e-85ec-123f9899a9f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import statsmodels.graphics.tsaplots as sgt\n",
    "import os\n",
    "\n",
    "def perform_eda(df):\n",
    "    \"\"\"\n",
    "    Visualizes the time series and generates PACF plots for the variables.\n",
    "    Saves plots to results/plots/.\n",
    "    Returns the DataFrame unchanged.\n",
    "    \"\"\"\n",
    "    # Ensure the plots directory exists\n",
    "    os.makedirs(\"results/plots\", exist_ok=True)\n",
    "\n",
    "    # Task 8: Generate summary statistics and time series plot\n",
    "    summary_stats = df.describe()\n",
    "    with open(\"results/summary_stats.txt\", \"w\") as f:\n",
    "        f.write(summary_stats.to_string())\n",
    "\n",
    "    plt.figure(figsize=(12, 5), dpi=100)\n",
    "    plt.plot(df['unempgr'], label='Unemployment Growth Rate (%)', color='blue')\n",
    "    plt.plot(df['fedrate'], label='Federal Funds Rate (%)', color='green')\n",
    "    plt.plot(df['inflat'], label='Inflation Rate (%)', color='red')\n",
    "    plt.title('Time Series of Unemployment Growth, Federal Funds, and Inflation Rates (1970–2019)')\n",
    "    plt.xlabel('Date')\n",
    "    plt.ylabel('Percentage')\n",
    "    plt.legend()\n",
    "    plt.grid(True, linestyle='--', alpha=0.7)\n",
    "    plt.savefig(\"results/plots/time_series_plot.png\")\n",
    "    plt.close()\n",
    "\n",
    "    # Task 15: Plot PACF for each variable\n",
    "    for column in ['unempgr', 'fedrate', 'inflat']:\n",
    "        plt.figure(figsize=(10, 4), dpi=100)\n",
    "        sgt.plot_pacf(df[column].dropna(), lags=10, method='ywm')\n",
    "        plt.title(f'PACF of {column}')\n",
    "        plt.savefig(f\"results/plots/pacf_{column}.png\")\n",
    "        plt.close()\n",
    "\n",
    "    return df\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    from import_data import import_data\n",
    "    from merge_data import merge_data\n",
    "    unemp_q, inflat_q, fedfund_q = import_data()\n",
    "    df = merge_data(unemp_q, inflat_q, fedfund_q)\n",
    "    perform_eda(df)\n",
    "    print(\"EDA completed\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
