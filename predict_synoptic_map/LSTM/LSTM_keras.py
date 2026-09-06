import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
import time
import scipy.io as scio

from smp.config import load_config
from smp.utils import set_random_seed

config = load_config()
seed = config.random_seed
set_random_seed(seed)

def creat_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i: (i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)

file_dir = config.path("harmonic_coefficients")
data_str = scio.loadmat(file_dir)
data_mat = data_str['save_var'] # size: 619,100
data = np.zeros((617,1))
lm = 3
for i in range(0,617):
    data[i] = data_mat[i+2][lm]
dataframe = pd.DataFrame(data)
dataset = dataframe.values

# plt.figure(figsize=(12, 8))
# dataframe.plot()
# plt.ylabel('price')
# plt.yticks(np.arange(0, 300000000, 100000000))
# plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.reshape(-1, 1))

train_size = int(len(dataset)*0.7)
test_size = len(dataset)-train_size
train, test = dataset[0: train_size], dataset[train_size: len(dataset)]

look_back = 2
trainX, trainY = creat_dataset(train, look_back)
testX, testY = creat_dataset(test, look_back)

model = Sequential()

model.add(LSTM(input_dim=1, units=50, return_sequences=True))
#model.add(Dropout(0.2))

model.add(LSTM(input_dim=50, units=100, return_sequences=True))
#model.add(Dropout(0.2))

model.add(LSTM(input_dim=100, units=200, return_sequences=True))
#model.add(Dropout(0.2))

model.add(LSTM(300, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(150))
model.add(Dense(units=1))

model.add(Activation('relu'))
start = time.time()
model.compile(loss='mean_squared_error', optimizer='Adam')
model.summary()

history = model.fit(trainX, trainY, batch_size=64, epochs=100, 
                    validation_split=0.1, verbose=2)
print('compilatiom time:', time.time()-start)

trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
test_pred_num = test_size
test_last_step = trainX[-look_back-1:-1]
testfuturePredict = []
for i in range(test_pred_num):
    test_next_step = model.predict(test_last_step)
    testfuturePredict.append(test_next_step[-1])
    test_last_step = np.append(test_last_step[1:].reshape((1, look_back, 1)), test_next_step[-look_back-1:].reshape((1, look_back, 1)), axis=0)
testfuturePredict = np.array(testfuturePredict)
testfuturePredict = testfuturePredict.reshape((len(testfuturePredict), 1))
testPredict = testfuturePredict[look_back+1:]

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
print('Train Score %.2f RMSE' %(trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))
print('Test Score %.2f RMSE' %(testScore))

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:] = np.nan
trainPredictPlot = np.reshape(trainPredictPlot, (dataset.shape[0], 1))
trainPredictPlot[look_back: len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:] = np.nan
testPredictPlot = np.reshape(testPredictPlot, (dataset.shape[0], 1))
testPredictPlot[len(trainPredict)+(look_back*2)+1: len(dataset)-1, :] = testPredict

fig1 = plt.figure()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

fig2 = plt.figure()
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
# plt.ylabel('price')
# plt.xlabel('date')
plt.show()

fig3 = plt.figure()
plt.plot(np.arange(train_size+1, len(dataset)+1, 1), scaler.inverse_transform(dataset)[train_size:], label='dataset')
plt.plot(testPredictPlot, 'g', label='test')
# plt.ylabel('price')
# plt.xlabel('date')
plt.legend()
plt.show()

pred_num = 100
last_step = testX[-1]
last_step = last_step.reshape((1, look_back, 1))
futurePredict = []
for i in range(pred_num):
    next_step = model.predict(last_step)
    futurePredict.append(next_step)
    last_step = np.append(last_step[0][-look_back+1:], next_step)
    last_step = last_step.reshape((1, look_back, 1))
futurePredict = np.array(futurePredict)
futurePredict = futurePredict.reshape((len(futurePredict), 1))
futurePredict = scaler.inverse_transform(futurePredict)

data_future = np.append(data,np.zeros((pred_num,1)))
dataset_future = pd.DataFrame(data_future)
futurePredictPlot = np.empty_like(dataset_future)
futurePredictPlot[:] = np.nan
futurePredictPlot = np.reshape(futurePredictPlot, (dataset_future.shape[0], 1))
futurePredictPlot[-len(futurePredict):, :] = futurePredict

fig4 = plt.figure()
plt.plot(scaler.inverse_transform(dataset), label='dataset')
plt.plot(trainPredictPlot, label='train')
plt.plot(testPredictPlot, label='test')
plt.plot(futurePredictPlot, label='predict')
# plt.ylabel('price')
# plt.xlabel('date')
plt.legend()
plt.show()

db = 1
