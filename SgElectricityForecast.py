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
