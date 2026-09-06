import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD, EEMD, visualisation
import pandas as pd
import scipy.io as scio
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from smp.config import load_config

# 导入数据
config = load_config()
file_dir = config.path("harmonic_coefficients")
data_str = scio.loadmat(file_dir)
data_mat = data_str['save_var'] # size: 619,100
data_lm = np.zeros((617,1))
data_t = range(0,617)
lm = 1
for i in range(0,617):
    data_lm[i] = data_mat[i+2][lm]
data = data_lm.flatten()
# 数据归一化
# max_value = np.max(data)
# min_value = np.min(data)
# scalar = max_value - min_value
# data = list(map(lambda x: x/scalar, data))
# data = np.array(data)
# EMD分解
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]
train_t, test_t = data_t[:train_size], data_t[train_size:]
emd = EMD(max_imf=10, MAX_ITERATIONS=1000)
IMFs = emd(train_data)
# IMFs相加之和与原始数据比较
IMFs_sum = np.sum(IMFs, axis=0)
plt.subplot(9,1,1)
plt.plot(data, label="data")
plt.plot(IMFs_sum, label="IMFs_sum")
plt.legend()
for i, IMF in enumerate(IMFs):
    plt.subplot(9,1,i+2)
    plt.plot(IMF, label="IMF {}".format(i+1))
    plt.legend()
plt.title("IMFs")
plt.show()
# EMD测试预测
IMFs_test_pred = emd(train_data) # 对整个训练集进行EMD分解
IMFs_test_pred[-1] = IMFs[-1] # 将趋势项替换为训练集的趋势项
test_pred_IMF = IMFs_test_pred[-1][-len(test_data):] # 获取需要预测的数据的最后一个IMF
test_pred = []
for i in range(len(test_data)):
    IMFs_test = emd(np.append(train_data, test_pred)[-len(train_data):])
    test_pred_sub = IMFs_test[-1][-1] + test_pred_IMF[-1]
    test_pred.append(test_pred_sub)
# 评估EMD模型的预测性能
mse = mean_squared_error(test_data, test_pred)
mae = mean_absolute_error(test_data, test_pred)
r2 = r2_score(test_data, test_pred)
smape = np.mean(np.abs(test_pred - test_data) / (np.abs(test_pred) + np.abs(test_data))) * 200
print("MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}, SMAPE: {:.4f}".format(mse, mae, r2, smape))
# 对未来150步进行预测
IMFs_future_pred = emd(np.append(train_data, test_data)[-len(train_data):]) # 对整个数据集进行EMD分解
IMFs_future_pred[-1] = IMFs[-1] # 将趋势项替换为训练集的趋势项
future_pred_IMF = IMFs_future_pred[-1][-len(test_data):] # 获取需要预测的数据的最后一个IMF
future_pred = []
future_t = np.linspace(618,618+150,150)
for i in range(150):
    IMFs_future = emd(np.append(data, future_pred)[-len(train_data):])
    future_pred_sub = IMFs_future[-1][-1] + future_pred_IMF[-1]
    future_pred.append(future_pred_sub)
future_pred = np.array(future_pred)
plt.plot(data_t,data, label="Real Data")
plt.plot(train_t,IMFs_sum, label="IMFs Sum")
plt.plot(test_t,test_pred, label="Test Data")
plt.plot(future_t,future_pred, label="Predicted Data")
plt.title("EMD Predictions for Future 150 Steps")
plt.legend()
plt.show()
