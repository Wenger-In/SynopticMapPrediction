import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import torch 
from torch.autograd import Variable
import scipy.io as scio
import csv
import random
import time

from smp.config import load_config
from smp.utils import set_random_seed

save_or_not = 0

# 固定随机种子
config = load_config()
seed = config.random_seed
set_random_seed(seed)

def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(data, window, mode='same')
    return smoothed


# 数据导入和预处理
file_dir = config.path("sunspot")
data_str = scio.loadmat(file_dir)
# data_mat = data_str['save_var'] # size: 619,100
data_mat = data_str['sn']
data0 = np.zeros((len(data_mat),1))
lm = 1
# l_cor = data_mat[0][lm]
# m_cor = data_mat[1][lm]
for i in range(0,len(data_mat)):
    # data0[i] = data_mat[i+2][lm]
    # data0[i] = data_mat[i][2]
    data0[i] = data_mat[i]
data0 = data0.flatten()
data0_full = data0
data0_t_full = np.array(range(0,len(data0_full)))
# data0 = data0[:-120]

data_smooth = moving_average(data0, 12)
# data0 = data_smooth

data0_t = np.array(range(0,len(data0)))
data = pd.DataFrame(data0)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# 定义look_back
look_back = 30
look_forw = 132

# 生成训练集和测试集
def create_dataset(dataset, look_back, look_forw):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-look_forw):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i+look_back+look_forw, :])
    return np.array(dataX), np.array(dataY)
dataX, dataY = create_dataset(data,look_back,look_forw)

# 划分训练集、验证集、测试集
data_t = np.array(range(0,len(dataX)))
train_size = int(len(data_t) * 0.6)
val_size = int(len(data_t) * 0.2)
test_size = len(data_t) - train_size - val_size

# train_size = int(len(data) * 0.7)
# val_size = int(len(data) * 0.2)
# test_size = len(data) - train_size - val_size
# indices = np.random.permutation(len(data))
# train_data, val_data, test_data = data[indices[:train_size], :], \
#     data[indices[train_size:train_size+val_size], :], data[indices[train_size+val_size:], :]

train_t, val_t, test_t = data_t[:train_size], \
    data_t[train_size:train_size+val_size], data_t[train_size+val_size:]
trainX, valX, testX = dataX[:train_size], \
    dataX[train_size:train_size+val_size], dataX[train_size+val_size:]
trainY, valY, testY = dataY[:train_size], \
    dataY[train_size:train_size+val_size], dataY[train_size+val_size:]

trainX, trainY = trainX.reshape(-1,1,look_back), trainY.reshape(-1,1,1)
valX, valY = valX.reshape(-1,1,look_back), valY.reshape(-1,1,1)
testX, testY = testX.reshape(-1,1,look_back), testY.reshape(-1,1,1)

# 转换为PyTorch张量
trainX = torch.from_numpy(trainX).type(torch.Tensor)
trainY = torch.from_numpy(trainY).type(torch.Tensor)
valX = torch.from_numpy(valX).type(torch.Tensor)
valY = torch.from_numpy(valY).type(torch.Tensor)
testX = torch.from_numpy(testX).type(torch.Tensor)
testY = torch.from_numpy(testY).type(torch.Tensor)

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
    
# 定义模型参数
input_dim = look_back
hidden_dim = look_back
num_layers = 3
output_dim = 1
lr = 0.01
num_epochs = 2000

# 定义模型和损失函数
model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 定义早停策略
patience = 10
val_loss_min = np.Inf
val_loss_thres = 0.007
counter = 0

# 训练模型
start_time = time.time()
train_loss_lst = []
val_loss_lst = []
for epoch in range(num_epochs):
    outputs = model(trainX)
    optimizer.zero_grad()
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    val_outputs = model(valX)
    val_loss = criterion(val_outputs, valY)
    val_loss_np = val_loss.detach().numpy()
    
    train_loss_lst.append(loss.item())
    val_loss_lst.append(val_loss_np)
    
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
        slt_epoch = epoch
    else:
        counter += 1
        if (counter >= patience) & (val_loss_np < val_loss_thres) & (loss.item() < 0.003):
            print('Early stopping at Epoch [{}/{}], Train Loss: {:.5f}, Val Loss={:.5f}'.format(epoch+1, num_epochs, loss.item(), val_loss_np))
            break
print('Time Cost(s):', time.time()-start_time)
        
# 加载最佳模型
model.load_state_dict(torch.load('best_model.pkl'))

# 计算模型误差
outputs = model(trainX)
model_loss = criterion(outputs, trainY).item()

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
print('Val RMSE: %.3f' % np.sqrt(mean_squared_error(valY, val_predict)))
print('Test RMSE: %.3f' % np.sqrt(mean_squared_error(testY, test_predict)))
mean_rmse = (np.sqrt(mean_squared_error(trainY, train_predict))*train_size + \
            np.sqrt(mean_squared_error(valY, val_predict))*val_size + \
            np.sqrt(mean_squared_error(testY, test_predict))*test_size) / (train_size+val_size+test_size)

# 生成预测未来的输入数据集
def create_future_inputs(dataset,look_back,look_forw):
    dataX_ext = []
    for i in range(len(dataset)-look_back-look_forw+1):
        a = dataset[i:(i+look_back)]
        dataX_ext.append(a)
    return np.array(dataX_ext)

# 预测未来的时间序列
future_step = 120
future_t = np.array(range(0,future_step))
future_predict = np.zeros((future_step,1))
data_ext = data
for i in range(future_step):
    dataX_ext = create_future_inputs(data_ext, look_back,look_forw)
    dataX_ext = dataX_ext.reshape(-1,1,look_back)
    dataX_ext = torch.from_numpy(dataX_ext).type(torch.Tensor)
    outputs_ext = model(dataX_ext)
    future_predict[i] = outputs_ext[-1].detach().numpy()
    data_ext = np.append(data_ext,future_predict[i])
future_predict = scaler.inverse_transform(future_predict)
pred_error = mean_rmse / max(future_predict)
future_predict_ulim = future_predict * (1 + pred_error)
future_predict_llim = future_predict * (1 - pred_error)

# 绘制训练集、验证集和测试集的真实值和预测值
plt.figure(figsize=(10, 6))
# xmin, xmax = min(data0_t), max(future_t + len(data0_t))
# plt.hlines(0,xmin,xmax,color='pink',linestyles='dashed')
# ymin, ymax = min(data0), max(data0)
# plt.vlines(617,ymin,ymax,color='pink',linestyles='dashed')

plt.plot(data0_t_full,data0_full,label='real',color='k')
plt.plot(train_t+look_back+look_forw,train_predict[:train_size], label='train predict')
plt.plot(val_t+look_back+look_forw,val_predict, label='val predict')
plt.plot(test_t+look_back+look_forw,test_predict, label='test predict')
plt.plot(future_t+len(data0_t),future_predict, label='future predict')
# plt.plot(future_t+len(data0_t),future_predict_ulim, label='future predict upper limit',linestyle='dashed')
# plt.plot(future_t+len(data0_t),future_predict_llim, label='future predict lower limit',linestyle='dashed')
plt.legend()
# plt.suptitle('g_{}^{}'.format(int(l_cor), int(m_cor)))

plt.figure()
plt.plot(train_loss_lst,label='train loss')
plt.plot(val_loss_lst,label='val loss')
plt.legend()
plt.yscale('log')

if save_or_not == 1:
    save_dir = config.ensure_output("sunspot_output")
    save_file = 'train_predict.csv'
    np.savetxt(save_dir / save_file, train_predict[:train_size], delimiter=',')
    save_file = 'val_predict.csv'
    np.savetxt(save_dir / save_file, val_predict, delimiter=',')
    save_file = 'test_predict.csv'
    np.savetxt(save_dir / save_file, test_predict, delimiter=',')
    save_file = 'future_predict.csv'
    np.savetxt(save_dir / save_file, future_predict, delimiter=',')
    save_file = 'future_predict_ulim.csv'
    np.savetxt(save_dir / save_file, future_predict_ulim, delimiter=',')
    save_file = 'future_predict_llim.csv'
    np.savetxt(save_dir / save_file, future_predict_llim, delimiter=',')

plt.show()
