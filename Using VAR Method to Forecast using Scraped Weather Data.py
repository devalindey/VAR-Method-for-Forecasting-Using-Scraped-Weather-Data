# Time Series Anlysis of Scraped 

# Importing the libraries
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
import math
from sklearn.metrics import mean_squared_error

# Loading the Google Chrome Web Driver & importing a local HTML file
driver = webdriver.Chrome('chromedriver.exe')
driver.get('file:///C:/Users/daved/Desktop/ML/Projects/Scrape/Forecast.html')
content = driver.page_source
soup = BeautifulSoup(content)
table = soup.find_all('table', {'class':'twc-table'})

# Scraping data
l = []
for items in table:
 for i in range(len(items.find_all('tr'))-1):
  d = {}  
  d['Date_Time'] = items.find_all('span', {'class':'dsx-date'})[i].text
  d['Temperature'] = items.find_all('td', {'class':'temp'})[i].text
  d['Feels'] = items.find_all('td', {'class':'feels'})[i].text
  d['Precipitation'] = items.find_all('td', {'class':'precip'})[i].text
  d['Humidity'] = items.find_all('td', {'class':'humidity'})[i].text
  d['Wind'] = items.find_all('td', {'class':'wind'})[i].text
  l.append(d)
  
# Creating the dataframe & exporting it as a .CSV dataset
df = pd.DataFrame(l)
df2 = df['Wind'].str.split(' ', n = 1, expand = True)
df2[1] = df2[1].str.replace(r'\D', '')
df['Wind Speed'] = df2[1]
df['Temperature'] = df['Temperature'].str.replace(r'\D', '')
df['Feels'] = df['Feels'].str.replace(r'\D', '')
df['Precipitation'] = df['Precipitation'].str.replace(r'\D', '')
df['Humidity'] = df['Humidity'].str.replace(r'\D', '')
df = df.drop(['Wind'], axis = 1)
df3 = df['Date_Time']
df3.iloc[0:9, ] = '12-11-2019 ' + df3.iloc[0:9, ].astype(str)
df3.iloc[9:33, ] = '13-11-2019 ' + df3.iloc[9:33, ].astype(str)
df3.iloc[33:48, ] = '14-11-2019 ' + df3.iloc[33:48, ].astype(str)
df.to_csv('Weather Data.csv')

# Date_Time column is an object and we need to change it to datetime
df['Date_Time'] = pd.to_datetime(df.Date_Time , format = '%d-%m-%Y %H:%M')
data = df.drop(['Date_Time'], axis=1)
data.index = df.Date_Time

# Visualizing the Features
data.iloc[:, 0:5].astype(int).plot(figsize = (16, 8))
plt.legend()
plt.show()

# Augmented Dickey-Fuller test
ADF_Temp = data['Temperature'].values.astype(int)
result = adfuller(ADF_Temp)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
    
ADF_Wnsp = data['Wind Speed'].values.astype(int)
result = adfuller(ADF_Wnsp)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
    
ADF_Feel = data['Feels'].values.astype(int)
result = adfuller(ADF_Feel)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
    
ADF_Prcp = data['Precipitation'].values.astype(int)
result = adfuller(ADF_Prcp)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

ADF_Hmdt = data['Humidity'].values.astype(int)
result = adfuller(ADF_Hmdt)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

# Creating the Training and Validation set
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

# Fitting a VAR model
model = VAR(endog = train.astype(int))
model_fit = model.fit()
prediction = model_fit.forecast(model_fit.y, steps=len(valid))

# Converting Predictions to Dataframe
cols = data.columns
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,5):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]
       
# Checking the RMSE      
print('RMSE value for Temperature = ', math.sqrt(mean_squared_error(pred.iloc[:, 0:1], valid.iloc[:, 0:1].values)))
print('RMSE value for Feels = ', math.sqrt(mean_squared_error(pred.iloc[:, 1:2], valid.iloc[:, 1:2].values)))
print('RMSE value for Precipitation = ', math.sqrt(mean_squared_error(pred.iloc[:, 2:3], valid.iloc[:, 2:3].values)))
print('RMSE value for Humidity = ', math.sqrt(mean_squared_error(pred.iloc[:, 3:4], valid.iloc[:, 3:4].values)))
print('RMSE value for Wind Speed = ', math.sqrt(mean_squared_error(pred.iloc[:, 4:5], valid.iloc[:, 4:5].values)))

# Visual Comparison of train & valid sets
train.iloc[:,0:5].astype(int).plot(figsize = (16, 8))
plt.title('Training Set')
plt.legend(loc = 'best')
plt.show()

valid.iloc[:,0:5].astype(int).plot(figsize = (16, 8))
plt.title('Training Set')
plt.legend(loc = 'best')
plt.show()