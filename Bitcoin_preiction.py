from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
#import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
#py.init_notebook_mode(connected=True)
#%matplotlib inline
import quandl
quandl.ApiConfig.api_key = "WzSrfRD9PL7cWB_VSzY9"
data=quandl.get('BCHARTS/KRAKENUSD', returns='pandas')
data.info()
data.head()
data.tail()
#btc_trace = go.Scatter(x=data.index, y=data['Weighted Price'], name= 'Price')
#plt.plot([btc_trace])
data['Weighted Price'].replace(0, np.nan, inplace=True)
data['Weighted Price'].fillna(method='ffill', inplace=True)
from sklearn.preprocessing import MinMaxScaler
values = data['Weighted Price'].values.reshape(-1,1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
train_size = int(len(scaled) * 0.7)
test_size = len(scaled) - train_size
train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
print(len(train), len(test))
def create_dataset(dataset,look_back=1):
	dataX,dataY = [],[]
	for i in range(len(dataset)-look_back):
		a=dataset[i:(i+look_back),0]
		dataX.append(a)
		dataY.append(dataset[i+look_back,0])
	print(len(dataY))
	return np.array(dataX),np.array(dataY)

look_back =1
trainX,trainY=create_dataset(train,look_back)
testX,testY=create_dataset(test,look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model = Sequential()
model.add(LSTM(3, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(trainX, trainY, epochs=300, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)
yhat = model.predict(testX)
pyplot.plot(yhat, label='predict')
pyplot.plot(testY, label='true')
pyplot.legend()
pyplot.show()
yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(yhat_inverse, label='predict')
pyplot.plot(testY_inverse, label='actual', alpha=0.5)
pyplot.legend()
pyplot.show()