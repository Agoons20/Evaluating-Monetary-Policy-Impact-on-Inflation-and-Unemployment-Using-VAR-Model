## Evaluating Monetary Policy Impact on Inflation and Unemployment Using Vector Autoregression Model

### Situation ✅

The Federal Reserve (Feds) sets the federal funds rate during its regular Board of Governors’ meetings (Upcoming Fed Meeting Scheduled for June 2025), a key component of the government’s monetary policy that determines the interbank lending interest rate. This rate plays a critical role in shaping economic conditions across the United States. **How does monetary policy influence two vital economic indicators: inflation and unemployment rates?** for the period spanning from 1970 to 2019 to evaluate these relationships (excluding post-2019 data to avoid distortions from the COVID-19 pandemic). 


### Task ✅
The objective was to construct and analyze a Vector AautoRegression model to assess the effects of monetary policy on inflation and unemployment rates and recommend a Federal Reserve policy to address the high levels of inflation and unemployment. 


### Action ⭕️
To accomplish these task, I used pandas_DataReader() to import inflation, interest rates and unemployment data at quarterly frequency from FRED from the period spanning 1970 to 2019 . 

**1.	Data Import and Preparation:**
  o	Retrieved quarterly unemployment, inflation, and federal funds rate data from FRED (1970–2019). I stopped just before 2020 to prevent COVID volatility from affecting our estimates
  o	Merged datasets into a single DataFrame (df) with aligned time indices.

**2.	Exploratory Data Analysis (EDA):**
o	Plotted time series to visualize trends of the unemployment, inflation and federal lending rate.
o	Generated summary statistics for initial insights.

**3.	Feature Engineering:**
o	Computed unemployment growth (unempgr) as percentage changes.

4.	Stationarity Check:
o	Applied the Augmented Dickey-Fuller (ADF) test to confirm stationarity.
o	Differenced the federal funds rate (dfedrate) to ensure stationarity.

6.	Model Building:
o	Fitted a VAR model using the transformed variables (dfedrate, unempgr, inflat).
o	Selected optimal lag order


**Stationarity Check:**
I tested the stationarity of the unemployment rate, inflation rate, and federal funds rate using the Augmented Dickey-Fuller (ADF) test.
If any variable was found to be non-stationary, I applied first differencing to transform it into a stationary series, ensuring the VAR model’s validity.


**Order Selection for VAR:**
First, I used the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) to determine the optimal lag order for the VAR model. I then used the logliklihood and correlation of residuals tradeoff to choose a max lag number of lags based on the output of the model. 
After testing models with various lag lengths, I selected the order that minimized these criteria, balancing model fit and complexity.

**VAR Model Estimation and Interpretation:**
I estimated the VAR model with the chosen lag order and examined the coefficients to uncover relationships between the variables.
For example, the results might show that an increase in the federal funds rate reduces inflation after a lag, while impacting unemployment more immediately.


**Residual Evaluation:**
I analyzed the residuals using diagnostic tests like the Durbin-Watson test for autocorrelation and the Breusch-Pagan test for heteroscedasticity.
The residuals satisfied the model’s assumptions, confirming the reliability of the results.

**Forecasting:**
- I divided the dataset into a training set (all data except the last 8 quarters) and a test set (the last 8 quarters).
Using the training data, I trained the VAR model and forecasted the variables for 8 quarters. If transformations like differencing were applied, I inverted them to express forecasts in their original scale.

- I plotted the forecasts alongside the actual test data for visual comparison.


**Forecast Accuracy:**
- I calculated MAPE and RMSE for each variable’s forecasts to quantify prediction accuracy.

- For instance, the MAPE might be 2.5% for inflation, 1.8% for unemployment, and 3.1% for the federal funds rate, indicating reliable forecasts.


**Granger Causality Test:**
- I conducted Granger Causality tests to explore whether one variable could predict another.
- The tests might reveal that the federal funds rate Granger-causes both inflation and unemployment, highlighting its predictive influence.


### Result and recommendation ✅
The VAR model captured bidirectional relationships (e.g., differenced fedrate lending rates affects unemployment growth rate, and unemployment growth rate affects differenced fedrate lending rates), but it doesn’t explicitly test which direction is statistically significant. Granger Causality tests provide this directional insight showing that differenced fedrate lending rates Granger-causes unemployment growth rate (p < 0.001), meaning past changes in the federal funds rate are useful for predicting unemployment growth. This supports the idea that monetary policy impacts unemployment. Thus, **to address high levels of unemployment, the Federal Reserve should consider lowering interest rates or increasing the money supply to stimulate spending and encourage hiring.** 

**The lack of Granger Causality from differenced fedrate lending rates to inflation challenges the recommendation to raise rates to combat inflation, as the model suggests limited predictive power in this direction.** This finding aligns with the VAR results, where differenced federal lending rate unexpectedly increased inflation, prompting the need for a structural VAR (SVAR) to capture contemporaneous effects.



===== 


Evaluating Monetary Policy Impact on Inflation and Unemployment Using VAR Model
Project Overview (STAR Format)
Situation
The Federal Reserve adjusts the federal funds rate to influence economic variables like inflation and unemployment as part of its monetary policy. Understanding these dynamics is vital for informed policymaking. This project uses historical data from 1970 to 2019 to evaluate these relationships, excluding post-2019 data to avoid distortions from the COVID-19 pandemic.
Task
The objective was to construct a Vector Autoregression (VAR) model to examine the interactions among:

Unemployment rate (UNRATE) and its percentage growth (unempgr).
Inflation rate (FLEXCPIM679SFRBATL).
Federal funds rate (FEDFUNDS), differenced into dfedrate for stationarity.

Key tasks included:

Importing and cleaning quarterly data from FRED.
Ensuring stationarity through differencing.
Building and fitting a VAR model with optimal lags.
Conducting Granger Causality tests.
Splitting data into training (96%) and test (4%) sets for forecasting.
Evaluating forecasts with metrics (ME, MAE, MPE, MAPE, RMSE).
Deriving policy recommendations.

Action
The project was executed through a systematic workflow:

Data Import and Preparation: Fetched quarterly unemployment, inflation, and federal funds rate data from FRED (1970–2019).
Preprocessing: Resampled data to quarterly frequency, computed unemployment growth (unempgr), and differenced the federal funds rate (dfedrate).
Exploratory Analysis: Visualized time series and checked stationarity with ADF tests.
VAR Modeling: Fitted a Vector Auto Regression model with 3 lags (chosen via lag selection methods) on unempgr, dfedrate, and inflat.
Forecasting and Evaluation: Forecasted 8 quarters, visualized predictions, and computed accuracy metrics.
Granger Causality: Tested predictive relationships (e.g., dfedrate → unempgr).
Policy Insights: Derived recommendations based on findings.

Result

Findings:
Granger Causality: dfedrate significantly predicts unempgr (p < 0.001), but not inflat (p > 0.24).
Forecast Accuracy: See results/analysis/forecast_metrics.txt (e.g., MAPE for dfedrate around 9.87%, indicating reasonable accuracy).


Visualizations: See results/plots/ for time series, PACF, and forecast vs. actuals plots.


Prerequisites

Operating System: macOS or Linux (Windows users can use WSL or Git Bash).
Python: Version 3.7 or higher.
pip: Python package manager.
Internet Connection: Required to fetch data from FRED.

Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/monetary-policy-var-model.git
cd monetary-policy-var-model


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Make the Bash Script Executable:
chmod +x run_all.sh


Run the Pipeline:
bash run_all.sh

This will execute the entire pipeline, generating outputs in the results/ directory without saving intermediate data to disk.


Expected Outputs

Plots: Time series, PACF, and forecast vs. actuals plots in results/plots/.
Analysis Results:
Missing values and duplicates: results/analysis/missing_values.txt, results/analysis/duplicates.txt.
Stationarity tests: results/analysis/stationarity_results.txt.
Granger Causality: results/analysis/granger_causality_*.txt.
Forecast Metrics: results/analysis/forecast_metrics.txt.


Model Outputs: VAR model summary, interpretation, and lag selection in results/models/.
Conclusion: Key findings and policy recommendations in results/conclusion.txt.

Dependencies
See requirements.txt for the full list. Key packages include:

pandas==2.0.3
numpy==1.24.3
statsmodels==0.14.0
matplotlib==3.7.0
pandas_datareader==0.10.0

Notes for Recruiters

This project demonstrates skills in time series analysis, econometrics, and Python programming.
The pipeline is automated via run_all.sh, producing all results with a single command.
Intermediate data is processed in memory, with only final outputs saved to the results/ directory.
For Windows users, you may need to run python scripts/main.py directly or use WSL/Git Bash to execute run_all.sh.

For questions, please contact do.agoons@yahoo.com 
