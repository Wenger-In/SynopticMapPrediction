import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
from torch import nn
from torch.autograd import Variable
import scipy.io as scio
from scipy.interpolate import interp1d
import csv
from torch.optim.lr_scheduler import StepLR
import random

from smp.config import load_config
from smp.constants import LSTM_ORIGINAL_RANDOM_SEED
from smp.utils import set_random_seed

# 固定随机种子
config = load_config()
seed = LSTM_ORIGINAL_RANDOM_SEED
set_random_seed(seed)

# 导入数据
file_dir = config.path("harmonic_coefficients")
data_str = scio.loadmat(file_dir)
data_mat = data_str['save_var'] # size: 619,100
data = np.zeros((617,1))
data_t = range(0,617)
lm = 3 # 选择预测哪个编号的球谐系数
for i in range(0,617):
    data[i] = data_mat[i+2][lm]
df = pd.DataFrame(data)
data_csv = df

# plt.plot(data_csv)
# plt.show()
# 数据预处理
data_csv = data_csv.dropna() 
dataset = data_csv.values
dataset = dataset.astype('float32')   # 变量类型转换
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value-min_value
dataset_norm = list(map(lambda x: x/scalar, dataset)) # 数据标准化

def create_dataset(dataset,look_back):
    dataX, dataY=[], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY) 

look_back = 2 # 用作输入变量来预测下一个时间段的以前的时间步数
hidden_size = 4 # 隐藏层数
data_X, data_Y = create_dataset(dataset_norm,look_back)

train_size = int(len(data_X) * 0.7) # 训练集
test_size = len(data_X)-train_size # 测试集
 
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
 
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]
 
train_X = train_X.reshape(-1,1,look_back)
train_Y = train_Y.reshape(-1,1,1)
test_X = test_X.reshape(-1,1,look_back)
 
train_x = torch.from_numpy(train_X) # 转化成pytorch中的tensor，便后续操作
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)

class lstm_reg(nn.Module):
    def __init__(self,input_size,hidden_size, output_size=1,num_layers=2):
        super(lstm_reg,self).__init__()
        self.rnn = nn.LSTM(input_size,hidden_size,num_layers)
        self.reg = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x, _ = self.rnn(x)
        s,b,h = x.shape
        x = x.view(s*b, h) 
        x = self.reg(x)
        x = x.view(s,b,-1) 
        return x

net = lstm_reg(look_back,hidden_size)
 
criterion = nn.MSELoss()  # 损失函数均方差
optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)

for e in range(1000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
 
    out = net(var_x)
    loss = criterion(out, var_y)
 
    optimizer.zero_grad() # 梯度置零
    loss.backward()  # 回传损失
    optimizer.step() # 回传损失过程中计算梯度
    if loss.item() < 1e-4:
        print('Epoch: {}, Loss: {:.6f}'.format(e+1, loss.item()))
        print("The loss value is reached")
        break
    elif (e+1)%100 == 0:
        print('Epoch: {}, Loss:{:.5f}'.format(e+1, loss.item()))
        
# torch.save(net.state_dict(), 'net_params_'+str(lm)+'.pkl') # 保存训练文件

data_X = data_X.reshape(-1, 1, look_back)
data_X = torch.from_numpy(data_X) 
var_data = Variable(data_X)
pred_test = net(var_data)  # 产生预测结果
pred_test = pred_test.view(-1).data.numpy()
pred_test = pred_test * scalar
pred_test_size = len(pred_test)

pred_num = 150 # 向后预测的步数
pred_Y = np.zeros((pred_num+1,1))
pred_size = len(pred_Y)
dataset_sub = dataset[len(dataset)-look_back-1:]
dataset_sub = dataset_sub.astype('float32') 
max_value_sub = np.max(dataset_sub)
min_value_sub = np.min(dataset_sub)
scalar_sub = max_value_sub-min_value_sub
dataset_norm_sub = list(map(lambda x: x/scalar, dataset_sub)) # 标准化
data_X_sub, data_Y_sub = create_dataset(dataset_norm_sub,look_back)
data_X_sub = data_X_sub.reshape(-1, 1, look_back)
data_X_sub = torch.from_numpy(data_X_sub) 
var_data_sub = Variable(data_X_sub)
pred_sub = net(var_data_sub)
pred_Y[0] = pred_sub.detach().numpy() * scalar
    
for i_pred in range(pred_num): # 循环，将预测得到的数据输入用于下一步的预测
    dataset_sub = np.append(dataset_sub[1:],pred_Y[i_pred])
    dataset_sub = dataset_sub.astype('float32') 
    max_value_sub = np.max(dataset_sub)
    min_value_sub = np.min(dataset_sub)
    scalar_sub = max_value_sub-min_value_sub
    dataset_norm_sub = list(map(lambda x: x/scalar, dataset_sub))
    data_X_sub, data_Y_sub = create_dataset(dataset_norm_sub,look_back)
    data_X_sub = data_X_sub.reshape(-1, 1, look_back) 
    data_X_sub = torch.from_numpy(data_X_sub) 
    var_data_sub = Variable(data_X_sub) 
    pred_sub = net(var_data_sub)
    pred_Y[i_pred+1] = pred_sub.detach().numpy() * scalar

plt.plot(np.linspace(0, pred_size, pred_size+1)+len(dataset),np.insert(pred_Y,0,dataset[-1]), 'r', label='prediction')
plt.plot(np.linspace(1, pred_test_size, pred_test_size)+look_back,pred_test, 'g', label='test')
plt.plot(np.linspace(1, train_size, train_size)+look_back,pred_test[:train_size], 'k', label='train')
plt.plot(np.linspace(1, len(dataset), len(dataset)),dataset, 'b', label='real')
plt.legend(loc='best')
plt.show()

db = 1
