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
from smp.utils import set_random_seed

# 固定随机种子
config = load_config()
seed = config.random_seed
set_random_seed(seed)

# test
#LSTM（Long Short-Term Memory）是长短期记忆网络
file_dir = config.path("harmonic_coefficients")
data_str = scio.loadmat(file_dir)
data_mat = data_str['save_var'] # size: 619,100
data = np.zeros((617,1))
data_t = range(0,617)
lm = 3
for i in range(0,617):
    data[i] = data_mat[i+2][lm]
df = pd.DataFrame(data)
df_np = data.flatten()
data_diff = df.diff(1)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

data_smooth = moving_average(df_np,5)
df_smooth = pd.DataFrame(data_smooth)

# 创建插值函数
interp_rate = 1
f_interp = interp1d(data_t, df_np, kind='cubic',fill_value="extrapolate")
# 创建新的一维数组
data_t_interp = np.linspace(0, len(data_t), num=len(data_t)*interp_rate, endpoint=True)
# 使用插值函数进行插值
data_interp = f_interp(data_t_interp)
df_interp = pd.DataFrame(data_interp)
# 保存插值后的文件
# with open('interp.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(data_interp)

data_csv = df
#pandas.read_csv可以读取CSV（逗号分割）文件、文本类型的文件text、log类型到DataFrame
#原有两列，时间和乘客数量，usecols=1：只取了乘客数量一列

# plt.plot(data_t,df,label='data')
# plt.plot(data_t_interp,df_interp,label='interp')
# plt.legend(loc='best')
# plt.show()

# plt.plot(df_smooth,label='data_smooth')
# plt.legend(loc='best')
# plt.show()

# plt.plot(data_diff,label='data_diff')
# plt.legend(loc='best')
# plt.show()
#数据预处理
data_csv = data_csv.dropna() #去掉na数据
dataset = data_csv.values      #字典(Dictionary) values()：返回字典中的所有值
dataset = dataset.astype('float32')   #astype(type):实现变量类型转换
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value-min_value
dataset_norm = list(map(lambda x: x/scalar, dataset)) #将数据标准化到0~1之间
#lambda:定义一个匿名函数，区别于def
#map(f(x),Itera):map()接收函数f和一个list,把函数f依次作用在list的每个元素上,得到一个新的object并返回



'''
接着我们进行数据集的创建，我们想通过前面几个月的流量来预测当月的流量，
比如我们希望通过前两个月的流量来预测当月的流量，我们可以将前两个月的流量
当做输入，当月的流量当做输出。同时我们需要将我们的数据集分为训练集和测试
集，通过测试集的效果来测试模型的性能，这里我们简单的将前面几年的数据作为
训练集，后面两年的数据作为测试集。
'''
def create_dataset(dataset,look_back):#look_back 以前的时间步数用作输入变量来预测下一个时间段
    dataX, dataY=[], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i+look_back)]  #i和i+1赋值
        dataX.append(a)
        dataY.append(dataset[i+look_back])  #i+2赋值
    return np.array(dataX), np.array(dataY)  #np.array构建数组

look_back = 150*interp_rate
hidden_size = 4
data_X, data_Y = create_dataset(dataset_norm,look_back)
#data_X: 2*142     data_Y: 1*142

#划分训练集和测试集，70%作为训练集
train_size = int(len(data_X) * 0.7)
test_size = len(data_X)-train_size
 
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
 
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]
 
train_X = train_X.reshape(-1,1,look_back) #reshape 中，-1使元素变为一行，然后输出为1列，每列2个子元素
train_Y = train_Y.reshape(-1,1,1) #输出为1列，每列1个子元素
test_X = test_X.reshape(-1,1,look_back)
 
train_x = torch.from_numpy(train_X) #torch.from_numpy(): numpy中的ndarray转化成pytorch中的tensor(张量)
train_y = torch.from_numpy(train_Y) #以便于后面的 Variable(train_x) 操作
test_x = torch.from_numpy(test_X)

#定义模型 输入维度input_size是2，因为使用2个月的流量作为输入，隐藏层维度hidden_size可任意指定，这里为4
class lstm_reg(nn.Module):
    def __init__(self,input_size,hidden_size, output_size=1,num_layers=2,dropout=0.02):
        super(lstm_reg,self).__init__()
        #super() 函数是用于调用父类(超类)的一个方法，直接用类名调用父类
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout) #LSTM 网络
        # self.dropout1 = nn.Dropout(dropout) #Dropout 层，进行dropout正则化，减少神经元之间的依赖关系，从而缓解过拟合问题
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.reg = nn.Linear(hidden_size,output_size) #Linear 函数继承于nn.Module
    def forward(self,x):   #定义model类的forward函数
        x, _ = self.lstm1(x)
        # x = self.dropout1(x)
        x, _ = self.lstm2(x)
        # x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        s,b,h = x.shape   #矩阵从外到里的维数
                   #view()函数的功能和reshape类似，用来转换size大小
        x = x.view(s*b, h) #输出变为（s*b）*h的二维
        x = self.reg(x)
        x = x.view(s,b,-1) #卷积的输出从外到里的维数为s,b,一列
        return x

net = lstm_reg(look_back,hidden_size) #input_size=2，hidden_size=4
 
criterion = nn.MSELoss()  #损失函数均方差
optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)
#构造一个优化器对象 Optimizer,用来保存当前的状态，并能够根据计算得到的梯度来更新参数
#Adam 算法:params (iterable)：可用于迭代优化的参数或者定义参数组的 dicts   lr:学习率
# gamma = 0.1
# step_size = 10
# scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
# 学习率衰减


for e in range(1000):
    var_x = Variable(train_x) #转为Variable（变量）
    var_y = Variable(train_y)
 
    out = net(var_x)
    loss = criterion(out, var_y)
        
    optimizer.zero_grad() #把梯度置零，也就是把loss关于weight的导数变成0.
    loss.backward()  #计算得到loss后就要回传损失，这是在训练的时候才会有的操作，测试时候只有forward过程
    optimizer.step() #回传损失过程中会计算梯度，然后optimizer.step()根据这些梯度更新参数
    
    if loss.item() < 1e-3:
        print('Epoch: {}, Loss: {:.6f}'.format(e+1, loss.item()))
        print("The loss value is reached")
        break
    elif (e+1)%100 == 0:
        print('Epoch: {}, Loss:{:.5f}'.format(e+1, loss.item()))
        
# torch.save(net.state_dict(), 'net_params_'+str(lm)+'.pkl') #保存训练文件net_params.pkl
#state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系

# net.load_state_dict(torch.load('net_params.pkl')) 


data_X = data_X.reshape(-1, 1, look_back) #reshape中，-1使元素变为一行，然后输出为1列，每列2个子元素
data_X = torch.from_numpy(data_X) #torch.from_numpy(): numpy中的ndarray转化成pytorch中的tensor（张量）
var_data = Variable(data_X) #转为Variable（变量）
pred_test = net(var_data)  #产生预测结果
pred_test = pred_test.view(-1).data.numpy() #view(-1)输出为一行
pred_test = pred_test * scalar
pred_test_size = len(pred_test)

pred_num = 150*interp_rate
pred_Y = np.zeros((pred_num+1,1))
pred_size = len(pred_Y)
dataset_sub = dataset[-look_back-1:]
dataset_sub = dataset_sub.astype('float32')   #astype(type):实现变量类型转换
dataset_norm_sub = list(map(lambda x: x/scalar, dataset_sub)) #将数据标准化到0~1之间
data_X_sub, data_Y_sub = create_dataset(dataset_norm_sub,look_back)
data_X_sub = data_X_sub.reshape(-1, 1, look_back) #reshape中，-1使元素变为一行，然后输出为1列，每列2个子元素
data_X_sub = torch.from_numpy(data_X_sub) #torch.from_numpy(): numpy中的ndarray转化成pytorch中的tensor（张量）
# max_value_sub = np.max(dataset_sub)
# min_value_sub = np.min(dataset_sub)
# scalar_sub = max_value_sub-min_value_sub
var_data_sub = Variable(data_X_sub) #转为Variable（变量）
pred_sub = net(var_data_sub)  #产生预测结果
pred_Y[0] = pred_sub.detach().numpy() * scalar
    
for i_pred in range(pred_num):
    dataset_sub = np.append(dataset_sub[1:],pred_Y[i_pred])
    dataset_sub = dataset_sub.astype('float32')   #astype(type):实现变量类型转换
    dataset_norm_sub = list(map(lambda x: x/scalar, dataset_sub)) #将数据标准化到0~1之间
    data_X_sub, data_Y_sub = create_dataset(dataset_norm_sub,look_back)
    data_X_sub = data_X_sub.reshape(-1, 1, look_back) #reshape中，-1使元素变为一行，然后输出为1列，每列2个子元素
    data_X_sub = torch.from_numpy(data_X_sub) #torch.from_numpy(): numpy中的ndarray转化成pytorch中的tensor（张量）
    var_data_sub = Variable(data_X_sub) #转为Variable（变量）
    pred_sub = net(var_data_sub)  #产生预测结果
    pred_Y[i_pred+1] = pred_sub.detach().numpy() * scalar

# pred_Y = pred_Y.view(-1).data.numpy() #view(-1)输出为一行

plt.plot(np.linspace(0, pred_size, pred_size+1)+len(dataset),np.insert(pred_Y,0,dataset[-1]), 'r', label='prediction')
plt.plot(np.linspace(1, pred_test_size, pred_test_size)+look_back,pred_test, 'g', label='test')
plt.plot(np.linspace(1, train_size, train_size)+look_back,pred_test[:train_size], 'k', label='train')
plt.plot(np.linspace(1, len(dataset), len(dataset)),dataset, 'b', label='real')
plt.legend(loc='best') #loc显示图像  'best'表示自适应方式
plt.show()

db = 1
# pred_restored = np.insert(np.cumsum(pred_Y),0,df_np[-1])
# pred_test_restored = np.insert(np.cumsum(pred_test),0,df_np[train_size])

# plt.plot(np.linspace(0, pred_size, pred_size+1)+len(dataset),pred_restored, 'r', label='prediction')
# plt.plot(np.linspace(0, pred_test_size, pred_test_size+1)+look_back,pred_test_restored, 'g', label='test')
# plt.plot(np.linspace(0, train_size, train_size+1)+look_back,pred_test_restored[:train_size+1], 'k', label='train')
# plt.plot(np.linspace(0, len(dataset), len(dataset)+1),df_np, 'b', label='real')
# plt.legend(loc='best') #loc显示图像  'best'表示自适应方式
# plt.show()




























