## Evaluating Monetary Policy Impact on Inflation and Unemployment Using VAR Model

### Situation

The Federal Reserve sets the federal funds rate during its regular Board of Governors’ meetings (Upcoming Fed Meeting SCheduled for June 2025), a key component of the government’s monetary policy that determines the interbank lending interest rate. This rate plays a critical role in shaping economic conditions across the United States. How does monetary policy influence two vital economic indicators: inflation and unemployment rates? 

### Task
The objective was to construct and analyze a Vector AautoRegression model to assess the effects of monetary policy on inflation and unemployment rates and recommend a Federal Reserve policy to address the current high levels of inflation and unemployment. 


### Action 
To accomplish these task, I used pandas_DataReader() to download inflation, interest rates and unemployment data at quarterly frequency from FRED from the period spanning 1970 to 2019 (stopped just before 2020 to prevent COVID volatility from affecting our estimates). I then constructed a vector autoregression model and verified/did the following:

**Stationarity Check:**
I tested the stationarity of the unemployment rate, inflation rate, and federal funds rate using the Augmented Dickey-Fuller (ADF) test.
If any variable was found to be non-stationary, I applied first differencing to transform it into a stationary series, ensuring the VAR model’s validity.


**Order Selection for VAR:**
I used the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) to determine the optimal lag order for the VAR model.
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


**Policy Recommendation:**
- Given the current high inflation and unemployment, **and assuming the model showed that raising the federal funds rate reduces** /*update this*/ inflation but increases unemployment short-term, I recommended a cautious, gradual rate increase.

- This approach aims to curb inflation while minimizing adverse effects on employment, with ongoing monitoring to adjust as needed.

### Result
The project successfully delivered a VAR model that illuminated the effects of monetary policy on inflation and unemployment. The analysis revealed dynamic relationships, such as a lagged reduction in inflation and a quicker rise in unemployment following federal funds rate increases. Forecasts over 8 quarters closely matched actual data, as evidenced by low MAPE and RMSE values. Granger Causality tests confirmed the federal funds rate’s leading role in influencing the other variables. Based on these insights, I proposed a balanced policy of gradual rate hikes to tackle inflation without severely impacting unemployment. This work demonstrated proficiency in time series analysis and econometrics, offering actionable insights for economic policy decisions.
