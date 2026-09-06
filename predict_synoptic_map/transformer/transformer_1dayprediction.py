import torch                            # 导入神经网络框架Pytorch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle
from tqdm import tqdm
import scipy.io as scio

from smp.config import load_config
from smp.utils import set_random_seed

#print(torch.__version__)

# #print(train_df.head(5))
# train_df = train_df[[28, 29, 30]].copy()
# valid_df = valid_df[[28, 29, 30]].copy()

# new_train_df = train_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))  # 对每一列数据归一化
# new_valid_df = valid_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))  # 对每一列数据归一化
# val_max = np.max(valid_df[30])
# val_min = np.min(valid_df[30])
# #print(train_df.shape)
# #print(valid_df.shape)

# # 时间点长度
# time_stamp = 576
# x_train = []
# y_train = []
# new_train_df = np.array(new_train_df)
# new_valid_df = np.array(new_valid_df)

# for i in tqdm(range(time_stamp, len(new_train_df)-288)):
#     x_train.append(new_train_df[i - time_stamp:i,:2])
#     #print("x_train_single shape: {}".format(new_train_df[i - time_stamp:i].shape))
#     y_train.append(new_train_df.iloc[i:i+288, 2])
#     #print("y_train_single shape: {}".format(new_train_df.iloc[i:i+288, 2].shape))

# x_train, y_train = np.array(x_train), np.array(y_train)
# ratio = 3
# length = (x_train).shape[0]
# train_length = (ratio-1.0)/ratio*length
# x_train_train = x_train[:train_length]
# x_train_val = x_train[train_length:]

# print("x_train_array shape: {}".format(x_train.shape))
# x_valid = []
# y_valid = []

# for i in tqdm(range(time_stamp, len(new_valid_df)-288, 288)):
#     x_valid.append(new_valid_df[i - time_stamp:i,:2])
#     y_valid.append(new_valid_df[i:i+288, 2])

# x_valid, y_valid = np.array(x_valid), np.array(y_valid)
# print("x_valid_array shape: {}".format(x_valid.shape))
# # 定义迭代器
# x_train_features = torch.from_numpy(x_train[:].astype(np.float32))
# y_train_features = torch.from_numpy(y_train[:].astype(np.float32))
# #print(x_train_features.shape)
# #print(y_train_features.shape)

# x_valid_features = torch.from_numpy(x_valid[:].astype(np.float32))
# y_valid_features = torch.from_numpy(y_valid[:].astype(np.float32))
# #print(x_train_features.shape)
# #print(y_train_features.shape)
# #变成二维数组
# x_train_features = x_train_features.reshape(x_train_features.shape[0], -1)
# x_valid_features = x_valid_features.reshape(x_valid_features.shape[0], -1)

# train_set = TensorDataset(x_train_features, y_train_features[:])
# valid_set = TensorDataset(x_valid_features, y_valid_features[:])

# train_data = DataLoader(dataset=train_set, batch_size=64, shuffle=True, drop_last= True)
# test_data = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, drop_last= True)

# 读取单一时间序列的CSV文件
config = load_config()
set_random_seed(config.random_seed)
file_dir = config.path("sunspot")
# time_series_values = time_series_df['values']
time_series_df = scio.loadmat(file_dir)
time_series_values = time_series_df['sn']

# 时间点长度
time_stamp = 576

x_data = []
y_data = []

for i in tqdm(range(time_stamp, len(time_series_values) - 288)):
    x_data.append(time_series_values[i - time_stamp:i])
    y_data.append(time_series_values[i:i + 288])

x_data, y_data = np.array(x_data), np.array(y_data)

# 转换为PyTorch的Tensor
x_tensor = torch.from_numpy(x_data.astype(np.float32))
y_tensor = torch.from_numpy(y_data.astype(np.float32))

# 构建TensorDataset和DataLoader
dataset = TensorDataset(x_tensor, y_tensor)
train_data = DataLoader(dataset=dataset, batch_size=64, shuffle=True, drop_last=True)

# 测试数据
test_x = time_series_values[-time_stamp:]
test_x_tensor = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0)  # 添加一维作为批次维度
test_data = DataLoader(dataset=TensorDataset(test_x_tensor), batch_size=1, shuffle=False)

# 构建网络结构
class Net(torch.nn.Module):# 继承 torch 的 Module
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.layer1 = torch.nn.Linear(n_feature, 64)  #
        self.layer2 = torch.nn.Linear(64, 128)   #
        self.layer3 = torch.nn.Linear(128, n_output)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        x = self.layer1(x)
        x = torch.relu(x)      #
        x = self.layer2(x)
        x = torch.relu(x)      #
        x = x.unsqueeze(1)
        x = self.encoder_layer(x)
        x = x.squeeze(1)
        x = self.layer3(x)
        return x

net = Net(1152, 288)  # 生成网络

#反向传播算法 SGD Adam等
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
#均方损失函数
criterion =	torch.nn.MSELoss()

losses = []  # 记录每次迭代后训练的loss
eval_losses = []  # 测试的
minimal_eval_loss = 1e8
overfitting = 0
threshold = 10

for i in tqdm(range(500)):
    train_loss = 0
    # train_acc = 0
    net.train() #网络设置为训练模式 暂时可加可不加
    for tdata, tlabel in train_data:
        #前向传播
        #print("input shape: {}".format(tdata.size()))
        y_ = net(tdata)
        #print("output shape: {}".format(y_.size()))
        #记录单批次一次batch的loss
        loss = criterion(y_, tlabel)
        print("[Training loss, iteration {}: {}]".format(i,loss))
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #累计单批次误差
        train_loss = train_loss + loss.item()

    losses.append(train_loss / len(train_data))
    # This dataset contains only the future input window, so it cannot provide
    # a validation loss. Training loss is retained for backward compatibility.

print('打印loss')
print('训练loss：', losses)
print('验证loss：', eval_losses)

y_ = []
for (edata,) in test_data:
    output = net(edata).detach().numpy()
    y_.extend(output)


result_dict = {
    #'true_label': y_valid.tolist(),
    'pred_label': y_
}
result_df = pd.DataFrame(result_dict)

result_df.to_csv('model_flux_flux_3year_2day_predict1day_5min_2005.csv', encoding='utf8', index=False)
torch.save(net.state_dict(), "checkpoint_1dayprediction.pth.tar")


