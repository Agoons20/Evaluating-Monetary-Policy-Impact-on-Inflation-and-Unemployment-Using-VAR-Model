## Evaluating Monetary Policy Impact on Inflation and Unemployment Using Vector Autoregression Model

### Situation 

The Federal Reserve (Feds) sets the federal funds rate during its regular Board of Governors’ meetings (Upcoming Fed Meeting Scheduled for June 2025), a key component of the government’s monetary policy that determines the interbank lending interest rate. This rate plays a critical role in shaping economic conditions across the United States. **How does monetary policy influence two vital economic indicators: inflation and unemployment rates?** Based on the analysis, make policy recommendations.


### Task
Construct and analyze a Vector Auto Regression model to assess the effects of monetary policy on inflation and unemployment rates and recommend a Federal Reserve policy to address the high levels of inflation and unemployment. Utilize unemployment data, inflation ad lending federal rate from Federal Reserve Economic Data (FRED) from 1970 to 2019 to evaluate these relationships (excluding post-2019 data to avoid distortions from the COVID-19 pandemic). 


### Action 
To understand how monetary policy (interest rate policy) affects/influences inflation and unemployment rate, I started by:
1. Importing data from FRED using pandas_DataReader API for the three variables of interest.  
2. The data from FRED is monthly data. I converted the data to quarterly frequency, create unemployment growth rate variable. 
3. Imported federal funds rate data, resample to quarterly frequency.
4. Imported inflation rate data, resample to quarterly frequency.
5. Examined missing values and sum them for each variable.  
6. Verified duplicate entries.
7. Merged the data, ensuring indices are aligned for Vector Auto Regression model compatibility.
8. Visualized all three series (unemployment growth rate, federal funds rate, inflation rate)
9. Checked for stationarity in the series.
10. Applied differencing to the federal funds rate since it is not stationary
11. Ran ADF test on differenced federal funds variable to ensure it is stationary.
12. Visualized the series again to ensure trend removal in federal funds data through differencing.
13. Built a Vector Auto Regression (VAR) model using unemployment growth rate, differenced federal funds rate, and inflation rate, and interpreted the results.
14. Chose the order of the VAR model by investigating and selecting the optimal lag of the model.
15. Investigated the number of lags using Partial Auto Correlation Function, noting significant predictors between 3 and 5 lags. 
16. Splited the data into training (96%) and test (4%) sets and interpret the model.
17. Visualize predictions for all three variables (unemployment growth rate, differenced federal funds rate, inflation rate)
18. Evaluated the forecast using accuracy.
19. Performed Granger Causality tests to evaluate predictive power of the discovered relationships in the VAR model.
20. Concluded the analysis with key findings and policy recommendations.


### Result and recommendation 
The Vector Auto Regression model captured bidirectional relationships (e.g., differenced fedrate lending rates affects unemployment growth rate, and unemployment growth rate affects differenced fedrate lending rates), but it doesn’t explicitly test which direction is statistically significant. Granger Causality tests provide this directional insight showing that differenced fedrate lending rates Granger-causes unemployment growth rate (p < 0.001), meaning past changes in the federal funds rate are useful for predicting unemployment growth. This supports the idea that monetary policy impacts unemployment. Thus, **to address high levels of unemployment, the Federal Reserve should consider lowering interest rates or increasing the money supply to stimulate spending and encourage hiring.** 

**The lack of Granger Causality from differenced fedrate lending rates to inflation challenges the recommendation to raise rates to combat inflation, as the model suggests limited predictive power in this direction.** This finding aligns with the VAR results, where differenced federal lending rate unexpectedly increased inflation, prompting the need for a structural VAR (SVAR) to capture contemporaneous effects.



### Prerequisites

**Operating System:** macOS or Linux (Windows users can use WSL or Git Bash).
**Python:** Version 3.7 or higher.
**pip:** Python package manager.



### Setup Instructions
**Clone the Repository:**
git clone https://github.com/Agoons20/Evaluating-Monetary-Policy-Impact-on-Inflation-and-Unemployment-Using-VAR-Model.git

cd monetary-policy-var-model


### Set Up a Virtual Environment (recommended):
python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt


### Install Dependencies:
pip install -r requirements.txt


### Make the Bash Script Executable:
chmod +x run_all.sh


### Run the Pipeline:
bash run_all.sh

This will execute the entire pipeline, generating outputs in the results/ directory without saving intermediate data to disk.


### Expected Outputs

Plots: Time series, PACF, and forecast vs. actuals plots in results/plots/.

### Analysis Results:
Missing values and duplicates: results/analysis/missing_values.txt, results/analysis/duplicates.txt.
Stationarity tests: results/analysis/stationarity_results.txt.
Granger Causality: results/analysis/granger_causality_*.txt.
Forecast Metrics: results/analysis/forecast_metrics.txt.


### Model Outputs: VAR model summary, interpretation, and lag selection in results/models/.


### Conclusion: Key findings and policy recommendations in results/conclusion.txt.


### Dependencies
See requirements.txt for the full list. Key packages include:

pandas==2.0.3

numpy==1.24.3

statsmodels==0.14.0

matplotlib==3.7.0

pandas_datareader==0.10.0


### Notes for Recruiters

This project demonstrates skills in time series analysis, econometrics, and Python programming.

The pipeline is automated via run_all.sh, producing all results with a single command.

Intermediate data is processed in memory, with only final outputs saved to the results/ directory.

For Windows users, you may need to run python scripts/main.py directly or use WSL/Git Bash to execute run_all.sh.

For questions, please contact do.agoons@yahoo.com 
