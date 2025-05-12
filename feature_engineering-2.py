{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "28733ace-f73b-4c2a-9f2b-91d900312de6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import os\n",
    "\n",
    "def merge_data(unemp_q, inflat_q, fedfund_q):\n",
    "    \"\"\"\n",
    "    Merges the datasets, checks for missing values and duplicates.\n",
    "    Returns the merged DataFrame.\n",
    "    \"\"\"\n",
    "    # Ensure the analysis directory exists for outputs\n",
    "    os.makedirs(\"results/analysis\", exist_ok=True)\n",
    "\n",
    "    # Standardize index to month-end timestamps\n",
    "    unemp_q.index = unemp_q.index.to_period('M').to_timestamp('M')\n",
    "\n",
    "    # Merge datasets\n",
    "    df = unemp_q.copy()\n",
    "    df['fedrate'] = fedfund_q['FEDFUNDS']\n",
    "    df['inflat'] = inflat_q['FLEXCPIM679SFRBATL']\n",
    "\n",
    "    # Task 5: Check for missing values\n",
    "    missing_values = df.isna().sum()\n",
    "    with open(\"results/analysis/missing_values.txt\", \"w\") as f:\n",
    "        f.write(\"Missing Values:\\n\")\n",
    "        for col, count in missing_values.items():\n",
    "            f.write(f\"{col}: {count}\\n\")\n",
    "\n",
    "    # Task 6: Check for duplicate entries\n",
    "    duplicates = df.index.duplicated().sum()\n",
    "    with open(\"results/analysis/duplicates.txt\", \"w\") as f:\n",
    "        f.write(f\"Number of Duplicate Entries: {duplicates}\\n\")\n",
    "\n",
    "    # Drop rows with NaN values (if any)\n",
    "    df = df.dropna()\n",
    "\n",
    "    return df\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    from import_data import import_data\n",
    "    unemp_q, inflat_q, fedfund_q = import_data()\n",
    "    df = merge_data(unemp_q, inflat_q, fedfund_q)\n",
    "    print(\"Data merged successfully\")"
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
