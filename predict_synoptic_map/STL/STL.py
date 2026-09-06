import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

from smp.config import load_config
from smp.utils import set_random_seed

# 导入数据
config = load_config()
set_random_seed(config.random_seed)
file_dir = config.path("harmonic_coefficients")
data_str = scio.loadmat(file_dir)
data_mat = data_str['save_var'] # size: 619,100
data_lm = np.zeros((617,1))
data_t = range(0,617)
lm = 10
for i in range(0,617):
    data_lm[i] = data_mat[i+2][lm]
data = data_lm.flatten()

# 创建一个示例序列
np.random.seed(config.random_seed)
time_series = np.sin(np.linspace(0, 10*np.pi, 200)) + np.random.randn(200)
time_series = data

# 进行时间序列分解
stl_result = sm.tsa.seasonal_decompose(time_series, model='additive', period=50)
trend = stl_result.trend
seasonal = stl_result.seasonal
residual = stl_result.resid

# 可视化分解结果
plt.figure(figsize=(10, 8))

plt.subplot(411)
plt.plot(time_series, label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='best')

plt.tight_layout()
plt.show()
