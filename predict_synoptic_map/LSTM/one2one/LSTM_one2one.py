import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from PyEMD import EMD, EEMD, visualisation
import matplotlib.pyplot as plt 
import torch 
from torch.autograd import Variable
import scipy.io as scio
import csv
import random
import time

from smp.config import load_config
from smp.utils import set_random_seed

# 固定随机种子
config = load_config()
seed = config.random_seed
set_random_seed(seed)

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x, _ = self.lstm(x)
        s,b,h = x.shape
        x = x.reshape(s*b, h)
        x = self.fc(x)
        x = x.view(s,b,-1) 
        return x

# 导入数据
hc_dir = config.path("harmonic_coefficients")
sn_dir = config.path("sunspot_interpolated")
hc_str = scio.loadmat(hc_dir)
sn_str = scio.loadmat(sn_dir)
hc_mat = hc_str['save_var'] # size: 619,100
sn_mat = sn_str['save_var']
hc_lm1 = np.zeros((617,1))
hc_lm2 = np.zeros((617,1))
sn = np.zeros((617,1))
lm1 = 0
lm2 = 2
l1_cor = hc_mat[0][lm1]
m1_cor = hc_mat[1][lm1]
for i in range(0,617):
    hc_lm1[i] = hc_mat[i+2][lm1]
    hc_lm2[i] = hc_mat[i+2][lm2]
    sn[i] = sn_mat[i][2]
hc_lm1 = hc_lm1.flatten()
hc_lm2 = hc_lm2.flatten()
hc_lm1 = sn
# hc_lm1 = np.diff(hc_lm1, 1)
# hc_lm2 = np.diff(hc_lm2, 1)
sn = sn.flatten()

def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(data, window, mode='same')
    return smoothed

# hc_lm1 = moving_average(hc_lm1, 12)
# hc_lm2 = moving_average(hc_lm2, 12)

# EMD分解
# emd = EMD()
# IMFs1 = emd(hc_lm1,max_imf=3)
# IMFs2 = emd(hc_lm2,max_imf=3)
# hc_lm1 = IMFs1[0]
# hc_lm2 = IMFs2[0]

plt.figure()
plt.plot(hc_lm1,label='coefficient as X')
plt.plot(hc_lm2,label='coefficient as Y')
plt.legend()
# plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
hc_lm1 = pd.DataFrame(hc_lm1)
hc_lm2 = pd.DataFrame(hc_lm2)
hc_lm1_norm = scaler.fit_transform(hc_lm1)
hc_lm2_norm = scaler.fit_transform(hc_lm2)

look_back = 150
# 生成数据集
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back+1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i+look_back-1, :])
    return np.array(dataX), np.array(dataY)
dataX, _ = create_dataset(hc_lm1_norm,look_back)
_, dataY = create_dataset(hc_lm2_norm,look_back)

# 一对一预测
# dataX = hc_lm1_norm
# dataY = hc_lm2_norm
# look_back = 1

# 划分训练集、验证集、测试集
data_t = np.array(range(0,len(dataX)))
train_size = int(len(data_t) * 0.7)
val_size = int(len(data_t) * 0.1)
test_size = len(data_t) - train_size - val_size

train_t, val_t, test_t = data_t[:train_size], \
    data_t[:train_size+val_size], data_t
    # data_t[train_size:train_size+val_size], data_t[train_size+val_size:]
trainX, valX, testX = dataX[:train_size], \
    dataX[:train_size+val_size], dataX
    # dataX[train_size:train_size+val_size], dataX[train_size+val_size:]
trainY, valY, testY = dataY[:train_size], \
    dataY[:train_size+val_size], dataY
    # dataY[train_size:train_size+val_size], dataY[train_size+val_size:]
trainX = np.append(trainX, np.append(-trainX, np.append(0.5*trainX, -0.5*trainX)))
trainY = np.append(trainY, np.append(-trainY, np.append(0.5*trainY, -0.5*trainY)))
    
# 转换为 PyTorch 张量
trainX, trainY = trainX.reshape(-1,1,look_back), trainY.reshape(-1,1,1)
valX, valY = valX.reshape(-1,1,look_back), valY.reshape(-1,1,1)
testX, testY = testX.reshape(-1,1,look_back), testY.reshape(-1,1,1)

trainX = torch.from_numpy(trainX).type(torch.Tensor)
trainY = torch.from_numpy(trainY).type(torch.Tensor)
valX = torch.from_numpy(valX).type(torch.Tensor)
valY = torch.from_numpy(valY).type(torch.Tensor)
testX = torch.from_numpy(testX).type(torch.Tensor)
testY = torch.from_numpy(testY).type(torch.Tensor)

# 定义模型参数和超参数
input_dim = look_back
hidden_dim = 4
output_dim = 1
num_epochs = 1000
lr = 0.01
num_layers = 4

# 定义模型和损失函数
model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 定义早停策略
patience = 10
val_loss_min = np.Inf
val_loss_thres = 0.005
counter = 0

# 训练模型
start_time = time.time()
for epoch in range(num_epochs):
    outputs = model(trainX)
    optimizer.zero_grad()
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    val_outputs = model(valX)
    val_loss = criterion(val_outputs[-val_size:], valY[-val_size:])
    val_loss_np = val_loss.detach().numpy()
    
    # 训练集损失达到阈值以下，跳出循环
    if loss.item() < 1e-4:
        print('Epoch [{}/{}], Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch+1, num_epochs, loss.item(), val_loss_np))
        print("The loss value is reached")
        break
    elif (epoch+1) % 50 == 0:
        print('Epoch [{}/{}], Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch+1, num_epochs, loss.item(), val_loss_np))
    
    # 使用验证集进行早停策略
    if val_loss < val_loss_min:
        counter = 0
        val_loss_min = val_loss
        torch.save(model.state_dict(), 'best_model.pkl')
    else:
        counter += 1
        if (counter >= patience) & (val_loss_np < val_loss_thres) & (loss.item() < 0.005):
            print('Early stopping at Epoch [{}/{}], Train Loss: {:.5f}, Val Loss={:.5f}'.format(epoch+1, num_epochs, loss.item(), val_loss_np))
            break
print('Time Cost(s):', time.time()-start_time)

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pkl'))

# 测试模型
model.eval()
test_predict = model(testX)

# 反归一化
train_predict = scaler.inverse_transform([[inner] for inner in outputs.detach().numpy().flatten()])
trainY = scaler.inverse_transform([[inner] for inner in trainY.detach().numpy().flatten()])
val_predict = scaler.inverse_transform([[inner] for inner in val_outputs.detach().numpy().flatten()])
valY = scaler.inverse_transform([[inner] for inner in valY.detach().numpy().flatten()])
test_predict = scaler.inverse_transform([[inner] for inner in test_predict.detach().numpy().flatten()])
testY = scaler.inverse_transform([[inner] for inner in testY.detach().numpy().flatten()])

# 计算RMSE
print('Train RMSE: %.3f' % np.sqrt(mean_squared_error(trainY, train_predict)))
print('Val RMSE: %.3f' % np.sqrt(mean_squared_error(valY[-val_size:], val_predict[-val_size:])))
print('Test RMSE: %.3f' % np.sqrt(mean_squared_error(testY[-test_size:], test_predict[-test_size:])))

# 绘制训练集、验证集和测试集的真实值和预测值
plt.figure()
xmin, xmax = min(data_t),max(data_t)
plt.hlines(0,xmin,xmax,color='pink',linestyles='dashed')

plt.plot(np.array(range(0,len(hc_lm2))),hc_lm2,label='real',color='k')
plt.plot(train_t+look_back,train_predict[:train_size], label='train predict')
plt.plot(val_t[-val_size:]+look_back,val_predict[-val_size:], label='val predict')
plt.plot(test_t[-test_size:]+look_back,test_predict[-test_size:], label='test predict')
plt.legend()
# plt.suptitle('g_{}^{}'.format(int(l_cor), int(m_cor)))
plt.show()

db = 1
