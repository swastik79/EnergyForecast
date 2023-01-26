#Importing the libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Above is a special style template for matplotlib, highly useful for visualizing time series data
from pylab import rcParams
rcParams['figure.figsize'] = 10,7

df = pd.read_csv('SG Electricity Data.csv')
print(df)

#Removing null values
df = df.dropna()
print(df)

#Renaming the columns
df.columns = ['Date' , 'Electricity_Generation']
df['Date'] = pd.to_datetime(df['Date'])

#Setting the date as index
df.set_index('Date', inplace = True)
print(df.head())

#Converting Electricity Generation Column into int
df = df.replace(',','', regex=True)
df['Electricity_Generation'] = pd.to_numeric(df['Electricity_Generation'])
print(type(df['Electricity_Generation'][0]))

#Visualization
plt.xlabel("Date")
plt.ylabel("Electricity Generated")
plt.title("Generation Graph")
plt.plot(df)
plt.show()

df.plot(style='k.') #Scatter plot
plt.show()

#Decomposing the model to show trend and seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df, model = 'multiplicative') #also try multiplicative to see which is better
result.plot()
plt.show()

#Performing ADF test
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determining rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    # Plot rolling statistics:
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    # perform dickey fuller test
    print("Results of dickey fuller test")
    adft = adfuller(timeseries['Electricity_Generation'], autolag='AIC')
    # output for dft will give us without defining what the values are.
    # hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],
                       index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values
    print(output)


test_stationarity(df)
#p-value > 0.05 and upward trend hence the series is not stationary

#Rolling average is calculated by taking the input for past 12 months and giving a mean generation value at every point
#ahead in the series
df_log = np.log(df)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.plot(df_log)
plt.plot(moving_avg, color="red")
plt.plot(std_dev, color ="black")
plt.show()

#Eliminating trend out of the series by taking the diff of the series and mean at every point
df_log_moving_avg_diff = df_log-moving_avg
df_log_moving_avg_diff.dropna(inplace=True)

#Performing ADF test again to test the stationary
test_stationarity(df_log_moving_avg_diff) #p-value < 0.05 hence the series is stationary

#Now from the graph we can see that the series is stationary

#To understand the trend of the data in series weighted average is calculated
weighted_average = df_log.ewm(halflife=12, min_periods=0,adjust=True).mean()

#df_log is subtracted with average-weight and ADF test is performed again
logScale_weightedMean = df_log-weighted_average
from pylab import rcParams
rcParams['figure.figsize'] = 10,6
test_stationarity(logScale_weightedMean)

#Since the p-value > 0.05 the series is made stationary by taking its difference
df_log_diff = df_log - df_log.shift()
plt.title("Shifted timeseries")
plt.xlabel("Date")
plt.ylabel("Generation")
plt.plot(df_log_diff)#Let us test the stationarity of our resultant series
df_log_diff.dropna(inplace=True)
test_stationarity(df_log_diff)

#Additive decomposition is next perfomed and ADF test is again done
#from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df_log, model='additive', freq = 12)
result.plot()
plt.show()
trend = result.trend
trend.dropna(inplace=True)
seasonality = result.seasonal
seasonality.dropna(inplace=True)
residual = result.resid
residual.dropna(inplace=True)
residual = pd.DataFrame({'Electricity_Generation':residual.values})
test_stationarity(residual)

#ACF and PACF plots are plotted to find the optimal parameter values
from statsmodels.tsa.stattools import acf,pacf
# we use d value here(data_log_shift)
acf = acf(df_log_diff, nlags=15)
pacf= pacf(df_log_diff, nlags=15,method='ols')#plot PACF
plt.subplot(121)
plt.plot(acf)
plt.axhline(y=0,linestyle='-',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.title('Auto corellation function')
plt.tight_layout()#plot ACF
plt.subplot(122)
plt.plot(pacf)
plt.axhline(y=0,linestyle='-',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.title('Partially auto correlation function')
plt.tight_layout()
plt.show()


#p = 1 and q = 1 as the acf and pacf cut to origin close to 1
#Hence we will use an ARIMA(1,1,1) model

#Building the ARIMA model
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df_log, order=(1,1,1))
result_AR = model.fit(disp = 0)
plt.plot(df_log_diff)
plt.plot(result_AR.fittedvalues, color='red')
plt.title("sum of squares of residuals")
print('RSS : %f' %sum((result_AR.fittedvalues-df_log_diff["Electricity_Generation"])**2)) 


#Doing the prediction

result_AR.plot_predict(1,700)
forecast, stderr, conf_int = result_AR.forecast(steps=200)

#Convertng the prediction in log form to the actual format
actual_forecast = np.exp(forecast)
print(actual_forecast)


