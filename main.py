{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5f54d0a-9212-44a3-87b3-c665727d4dba",
   "metadata": {},
   "outputs": [],
   "source": [
    "from scripts.import_data import import_data\n",
    "from scripts.merge_data import merge_data\n",
    "from scripts.eda import perform_eda\n",
    "from scripts.feature_engineering import feature_engineering\n",
    "from scripts.stationarity_check import check_stationarity\n",
    "from scripts.var_model import fit_var_model\n",
    "from scripts.compare_var_lags import compare_var_lags\n",
    "from scripts.analysis import perform_analysis\n",
    "\n",
    "def main():\n",
    "    \"\"\"\n",
    "    Orchestrates the entire pipeline in memory, saving only final outputs to results/.\n",
    "    \"\"\"\n",
    "    # Task 1-4: Import data\n",
    "    unemp_q, inflat_q, fedfund_q = import_data()\n",
    "\n",
    "    # Task 5-7: Merge data\n",
    "    df = merge_data(unemp_q, inflat_q, fedfund_q)\n",
    "\n",
    "    # Task 8, 15: EDA (visualizations saved to disk)\n",
    "    df = perform_eda(df)\n",
    "\n",
    "    # Task 10: Feature engineering\n",
    "    df = feature_engineering(df)\n",
    "\n",
    "    # Task 9, 11, 12: Stationarity check (stationarity results and plots saved to disk)\n",
    "    stationary_df = check_stationarity(df)\n",
    "\n",
    "    # Task 13, 14, 16: Fit VAR model\n",
    "    results, train, test, df_fc = fit_var_model(stationary_df)\n",
    "\n",
    "    # Task 14: Compare VAR models with different lags (results saved to disk)\n",
    "    compare_var_lags(train, lags_list=[3, 4, 5, 6, 8])\n",
    "\n",
    "    # Task 17-20: Analysis (forecast visualizations, metrics, Granger Causality, conclusion saved to disk)\n",
    "    perform_analysis(results, train, test, df_fc, df)\n",
    "\n",
    "    print(\"Pipeline completed. All outputs saved in results/\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
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
