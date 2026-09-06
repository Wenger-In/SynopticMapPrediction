import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
import scipy.io as scio
import matplotlib.pyplot as plt

from smp.config import load_config

# 导入数据
config = load_config()
file_dir = config.path("harmonic_coefficients")
data_str = scio.loadmat(file_dir)
data_mat = data_str['save_var'] # size: 619,100
data_lm = np.zeros((617,1))
data0_t = np.array(range(0,617))
lm = 2
l_cor = data_mat[0][lm]
m_cor = data_mat[1][lm]
for i in range(0,617):
    data_lm[i] = data_mat[i+2][lm]
    # data_lm[i] = data_mat[i][2]
data = data_lm.flatten()

# 傅里叶变换和反变换
fft = np.fft.fft(data)
freqs = np.fft.fftfreq(len(data))
idxs = np.where(freqs > 0)

# 绘制功率谱密度图
plt.plot(freqs[idxs], np.abs(fft[idxs])**2, label='Power spectrum')
# 添加水平线
xmin, xmax = 0, max(freqs)
plt.hlines(0,xmin,xmax,color='red',linestyles='solid')
# 添加竖直线
ymin, ymax = 0, max(np.abs(fft)**2)
for i in range(10,100,10):
    plt.vlines(freqs[i],ymin,ymax,color='gray',linestyles='dashed')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax*1.1)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('g_{}^{} Power Spectrum Density'.format(int(l_cor), int(m_cor))) 
plt.show()

# 根据功率谱密度，选择截断频率，将高频成分置零
icut = 60
fft[idxs[0][icut:]] = 0
smooth_values = np.fft.ifft(fft).real

# 向后预测150个时间步
predict_step = 150
predict_values = np.concatenate((smooth_values, np.zeros(predict_step)))
for i in range(len(data), len(data)+predict_step):
    fft = np.fft.fft(predict_values[i-len(data):i])
    idxs = np.where(freqs > 0)
    fft[idxs[0][icut:]] = 0
    smooth = np.fft.ifft(fft).real
    predict_values[i] = smooth[-1]

# 添加水平线
xmin, xmax = 0, len(data)+predict_step
plt.hlines(0,xmin,xmax,color='red',linestyles='solid')
# 绘制原始序列和预测序列的图表
plt.plot(data, label='Original Time Series')
plt.plot(predict_values, label='Predicted Time Series')
plt.legend()
plt.title('g_{}^{}'.format(int(l_cor), int(m_cor))) 
plt.show()
