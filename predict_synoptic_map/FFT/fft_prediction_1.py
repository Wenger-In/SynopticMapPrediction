import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

from smp.config import load_config

# 导入数据
config = load_config()
file_dir = config.path("harmonic_coefficients")
data_str = scio.loadmat(file_dir)
data_mat = data_str['save_var'] # size: 619,100
data_lm = np.zeros((617,1))
data0_t = np.array(range(0,617))
lm = 1
l_cor = data_mat[0][lm]
m_cor = data_mat[1][lm]
for i in range(0,617):
    data_lm[i] = data_mat[i+2][lm]
    # data_lm[i] = data_mat[i][2]
data = data_lm.flatten()

# 定义一个周期为 10，长度为 25 的正弦信号
t = np.linspace(0, 617, 617)
x = np.sin(2 * np.pi * t / 180)
t = data0_t
x = data

N = len(x)
freq = np.fft.fftfreq(N, d=t[1]-t[0])
fft = np.fft.fft(x)
X = fft

amplitudes = 2 * np.abs(fft) / N
amplitudes[10:-10] = 0
frequencies = freq[:N//2]
phases = np.angle(fft)[:N//2]

# 绘制功率谱密度图
# plt.plot(freq, np.abs(fft)**2, label='Power spectrum')
# # 添加水平线
# xmin, xmax = 0, max(freq)
# plt.hlines(0,xmin,xmax,color='red',linestyles='solid')
# # 添加竖直线
# ymin, ymax = 0, max(np.abs(fft)**2)
# for i in range(10,100,10):
#     plt.vlines(freq[i],ymin,ymax,color='gray',linestyles='dashed')
# plt.xlim(xmin, xmax)
# plt.ylim(ymin, ymax*1.1)
# plt.xlabel('Frequency')
# plt.ylabel('Power')
# plt.title('g_{}^{} Power Spectrum Density'.format(int(l_cor), int(m_cor))) 
# plt.show()


# 计算傅里叶系数
# N = len(x)
# k = np.arange(0, N)
# T = N / 10
# freq = k / T
# X = np.fft.fft(x) / N
# plt.plot(X)
X[500:-500] = 0
# X = X[:N//2]
# b = 2 * np.abs(X[1:])

# # 使用傅里叶系数来表示信号
# x_reconstructed = np.zeros_like(x)
# for i in range(1, len(b) + 1):
#     x_reconstructed += b[i-1] * np.sin(2 * np.pi * i / T * t)

# 提取每个正弦信号的振幅、频率和相位
# fft = X
# amplitudes = 2 * np.abs(fft) / N
# frequencies = freq[:N//2]
# phases = np.angle(fft)[:N//2]
# amplitudes = np.abs(fft)
# frequencies = np.fft.fftfreq(N, d=t[1]-t[0])
# phases = np.angle(fft)

# 输出每个正弦信号的振幅、频率和相位
# for i in range(10):
    # print("Frequency: {}, Amplitude: {}, Phase: {}".format(frequencies[i], amplitudes[i], phases[i]))

future_step = 150
t_ext = np.array(range(0,617+future_step))
x_sum = np.zeros_like(x)
x_ext = np.zeros(617+future_step)
for amplitude, frequency, phase in zip(amplitudes, frequencies, phases):
    if frequency == 0:
        amplitude = amplitude / 2
    sin_wave = amplitude * np.cos(2*np.pi*frequency*t + phase)
    sin_wave_ext = amplitude * np.cos(2*np.pi*frequency*t_ext + phase)
    x_sum += sin_wave
    x_ext += sin_wave_ext
    if amplitude != 0:
        plt.plot(sin_wave_ext)
# 添加竖直线
ymin, ymax = -1, 1
plt.vlines(617,ymin,ymax,color='black',linestyles='solid')
    
x_reconstructed = np.fft.ifft(X)

plt.figure()
# 添加水平线
xmin, xmax = 0, len(data)+future_step
plt.hlines(0,xmin,xmax,color='red',linestyles='solid')
# 绘制原始信号和使用傅里叶系数表示的信号
plt.plot(t_ext,x_ext,label='predict')
plt.plot(t, x, label='Original signal')
# plt.plot(t, x_reconstructed, label='Reconstructed signal')
plt.plot(t,x_sum,label='sum')
plt.plot(t, x-x_sum, label='error')
plt.legend()
# plt.title('g_{}^{}'.format(int(l_cor), int(m_cor))) 
plt.show()


db=1
