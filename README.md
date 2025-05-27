## Evaluating Monetary Policy Impact on Inflation and Unemployment Using Vector Autoregression Model
 
### Situation 
There’s been an increasing call for the Federal Reserve chair to cut interest rates (the president has been incredibly frustrated by the Fed chair’s unwillingness to cut rate, citing great job numbers and touting low inflation numbers). How does tweaking the federal funds rate ripples through the economy, specifically impacting unemployment growth and inflation? How are these variables related? Do past changes in interest rates reliably predict future economic trends, and what does this mean for policy decisions? 

### Task
Construct and analyze a Vector Auto Regression model to assess the effects of monetary policy on inflation and unemployment rates and recommend a Federal Reserve policy to address the high levels of inflation and unemployment. 

### Action

**The Quest for Data and Its Preparation**
The adventure began with gathering historical data from the Federal Reserve Economic Data (FRED) database, a treasure trove of economic indicators. The focus is on three key metrics from 1970 to 2019, deliberately stopping before the COVID-19 pandemic to avoid its chaotic distortions: 

•	Unemployment Rate (UNRATE): A measure of the percentage of the labor force that’s unemployed.

•	Inflation Rate (FLEXCPIM679SFRBATL): A gauge of price changes over time.

•	Federal Funds Rate (FEDFUNDS): The interest rate at which banks lend to each other overnight, a lever the Fed uses to steer the economy.

An application programming interface (API) was utilized to download the data from FRED. However, the raw unemployment rate (UNRATE) wasn’t the variable ultimately used. Instead, the unemployment growth rate (unempgr)—the percentage change in UNRATE over time—was utilized for this analysis. This shift was pivotal, and I’ll explain why as we dive deeper into the preprocessing and stationarity analysis – a prerequisite step to run Vector autoregression.

**Preprocessing the Data**
Raw data is like a wild river—it needs channeling to be useful. The first step was to ensure the datasets could work together seamlessly. The original data from FRED came in monthly increments, but it was converted to a quarterly frequency, using the last value of each quarter (with an index landing on dates like 1970-03-31, 1970-06-30, 1970-09-30 and so forth). Why quarterly frequency? Economic data often reveals seasonal patterns, and quarterly aggregation reduces noise in the data. 

The index of the unemployment growth rate calculated changed and wouldn’t align with the federal funds rate and inflation rate datasets. Merging the unemployment data with the federal funds rate and inflation rate datasets wasn’t straightforward. Each dataset had its own indexing rhythm, and for a Vector Autoregression (VAR) model, their indices had to align perfectly—like pieces of a puzzle snapping into place. To achieve this, the unemployment data was converted to a monthly period and then to a timestamp at the end of the month. This ensured compatibility across all datasets, setting the stage for a smooth merge into a single DataFrame.

Next, the merged dataframe was scanned for missing values and duplicates. Thankfully, the data was had neither. With the data prepped, lets start with:

**Exploratory Data Analysis (EDA):** In this phase, the time series variables are visualized, revealing trends that painted a clear picture of economic dynamics over decades. Here’s how these trends unfolded and what they mean for the relationships between the variables.
**Unemployment growth rate:** The plots showed significant increases in unemployment for more than two quarters (6 months) and this is generally referred to as a recession. We observe increased unemployment in early 1970s, 1980s, during the great depression (2008-2009) and more stable employment growth rate trend after the crash of the Lehman Brothers in 2008 (the Great recession). 

**Federal Funds Rate Trends:** In the late 1970s to 1980s, there was an aggressive monetary policy under Fed chair, Paul Volcker, to curb high rates of inflation (reduce the supply of money in the economy) and then rates gradually reduced and then were at 0% for much of the decade post-2008, with a gradual increase starting around 2015 as the economy recovered. 

**Inflation Trends:** We see high inflation rates in the mid 1970s towards 1980 and then it falls, showing a successful aggressive monetary policy from the Feds. From 1980 to 1990, we do see a stable inflation rate which hints at a stable economic environment (between 0% and 5%). During the great recession though, we see a sharp drop in inflation to negative values (deflation), followed by low inflation (0–5%) in the 2010s, reflecting weak demand and low oil prices. 

What does this mean for the relationship between these variables? In essence, high inflation often precedes federal funds rate increases, as the Fed tightens policy to restore price stability. This, in turn, can elevate unemployment by slowing the economy. Conversely, during downturns like 2008-2009, low rates aim to spur growth, stabilizing employment at the expense of prolonged low inflation.

**The technical journey was just as exciting—think data preprocessing**
All the variables were tested for stationarity using the Augmented Dickey-Fuller (ADF) test. A time series is stationary if its statistical properties — like mean, variance, and autocorrelation — stay constant over time. The Augmented Dickey-Fuller test checks if a time series is stationary by analyzing its past values (lags) to see if the mean, variance, and autocorrelation are stable over time. The federal funds rate variable wasn’t stationary, to correct this, differencing was applied to the series. Differencing means creating a new series using the operation Y’=Yn-Yn-1, where Yn is a federal funds rate (interest rate) at time N and Yn-1 for interest rate at time N-1. This removes the trend in the series. 

The vector autoregression model was then run, its output provides insights which helps understand how monetary policy (via the federal funds rate) affects unemployment and inflation. It is worth noting that the vector autoregression model treats all variables as endogenous, meaning each variable is a function of its own past values and the past values of the other variables (lags). Determining the number of past values (lags) to use in the VAR model is critical. 

The optimal number of lags for the Vector Autoregression (VAR) model was determined through a rigorous process. Initially, a model with 3 lags was fitted as a baseline. How was this done? 
-	The Akaike Information Criterion (AIC), which is the information lost by the model, was used to identify the lag order. 
-	The log likelihood – it tells us how well the model explains the data – was also used to determine the number of lags to use for VAR with higher values indicating better data fit. 
-	Additionally, the correlation matrix of residuals was analyzed to ensure low correlations of residuals. A high correlation of residuals means the residuals captures some behavior in the variables that is not picked up by the model. The lower the better. 
-	The Partial Auto-Correlation Function (PACF), whose output shows us which previous time series values affect the value of a the current period, was also used to determine the lag order.
  
While Partial Auto-Correlation Function (PACF) suggested 4–5 lags, iterative testing of lags 3 to 7 showed 3 lags achieved the lowest Akaike Information Criterion, highest log likelihood, and lowest residual correlations, ensuring an optimal fit for analyzing monetary policy impacts on inflation and unemployment.

### Result and recommendation 
The VAR model with three (3) lags was run. The VAR model captured bidirectional relationships (the differenced federal rate lending rates affects unemployment growth rate, and unemployment growth rate affects differenced federal rate lending rates), but it doesn’t explicitly test which direction is statistically significant.

The VAR model also found out that increasing the federal funds rates leads to an increase in inflation, which is counter intuitive. Maybe because the federal funds rate was differenced, that’s why it did not yield the obvious result of reducing inflation when federal funds rate are increased. 

Granger Causality tests the predictive power, not true causation of the relationship between any two time series variables. It checks if past variables of one time series (X) improve the predictions of another time series (Y) but does not necessarily mean X causes Y. It also gives a sense of direction to the relationship revealed by the Vector autoregression model. So, from VAR model, the differenced federal rate lending rates affects unemployment growth rate, and unemployment growth rate affects differenced federal rate lending rates. 

Granger Causality test provided this directional insight showing that differenced federal rate lending rates Granger-causes unemployment growth rate, meaning past changes in the federal funds rate are useful for predicting unemployment growth. This supports the idea that monetary policy impacts unemployment. Thus, to address high levels of unemployment, the Federal Reserve should consider lowering interest rates or increasing the money supply to stimulate spending and encourage hiring.

The lack of Granger Causality from differenced federal funds rate lending rates to inflation challenges the recommendation to raise rates to combat inflation, as the model suggests limited predictive power in this direction. This finding aligns with the VAR results, where, differenced federal lending rate unexpectedly increased inflation, prompting the need for a structural VAR (SVAR) to capture contemporaneous effects. 



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
