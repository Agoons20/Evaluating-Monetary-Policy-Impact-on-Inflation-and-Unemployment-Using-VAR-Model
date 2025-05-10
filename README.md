## Evaluating Monetary Policy Impact on Inflation and Unemployment Using VAR Model

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

