import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from scipy.io import loadmat

from smp.config import load_config
from smp.constants import CARRINGTON_ROTATIONS_PER_YEAR


def CWT(data,l,m,fs=1):
    t = np.arange(0, len(data)) / fs
    # wavename = "cgau8"   # cgau8 小波
    # wavename = "morl"  # morlet 小波
    # wavename = "cmor"  # cmor 小波
    wavename='cmorl1.5-1.0'
    
    cr2year = CARRINGTON_ROTATIONS_PER_YEAR
    # totalscale = 256
    # fc = pywt.central_frequency(wavename)  # 中心频率
    # cparam = 2 * fc * totalscale
    
    # scales = cparam / np.arange(totalscale, 1, -0.2)
    
    # scales = range(1,256,4)
    
    num_freq = 75
    wave_freq = np.logspace(-2.6, np.log10(fs/2), num_freq)
    scales = fs / wave_freq
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / fs)  # 连续小波变换

    
    # np.savetxt(save_dir+'cwt_'+str(l)+'^'+str(m)+'.csv', cwtmatr, delimiter = ',')
    # np.savetxt(save_dir+'freq_'+str(l)+'^'+str(m)+'.csv', frequencies, delimiter = ',')
    
    # plt.figure(figsize=(12, 6))
    # ax1 = plt.subplot(2,1,1)
    # plt.plot(t, data)
    # plt.xlabel("CR", fontsize = 14)
    # plt.ylabel("Amplitude", fontsize=14)
    # ax2 = plt.subplot(2,1,2)
    
    # f = np.array(frequencies)
    # print(f)
    # period_cr = 1 / f
    # period_year = period_cr / cr2year
    # print(period_year)
    # plt.pcolor(t,period_year, abs(cwtmatr))

    # yt = [1,2,5,10,20,25]
    # # ax2.set_yscale('log')
    # ax2.set_yticks(yt)
    # ax2.set_yticklabels(yt)

    # # print("min(frequencies):", min(frequencies))
    # # print("max(frequencies):", max(frequencies))
    # ax2.set_ylim([min(period_year), max(period_year)])

    # plt.xlabel("Time(s)", fontsize = 14)
    # plt.ylabel("Period(year)", fontsize=14)
    # plt.title(file_name, fontsize=14 )
    # plt.tight_layout()
    # # plt.savefig("./cwt_figures/" + file_name + "_CWT" + ".png")
    # plt.show()


def gener_simul_data():
    fs = 1024
    t = np.arange(0, 1.0, 1.0 / fs)
    f1 = 100
    f2 = 200
    f3 = 300
    data = np.piecewise(t, [t<1, t<0.8, t<0.3],
                        [lambda t: np.sin(2 * np.pi * f1 * t),
                         lambda t: np.sin(2 * np.pi * f2 * t),
                         lambda t: np.sin(2 * np.pi * f3 * t)])
    return data

if __name__ == "__main__":
    # print(pywt.families())
    # print(pywt.wavelist('morl'))
    
    ##  PART 0: import data
    config = load_config()
    data_dir = config.path("harmonic_coefficients")
    data0 = loadmat(data_dir, mat_dtype=True)
    data = data0['save_var']
    
    # extract data
    l_lst = data[0,:]
    m_lst = data[1,:]
    hc_mat = data[2:,:]

    # select order
    for l in range(1):
        for m in range(-l,l+1):
            m_num = 2*l + 1
            col_beg = l**2 + 1
            col_end = (l + 1)**2
            m_sub = m_lst[col_beg-1:col_end]
            hc_sub = hc_mat[:,col_beg-1:col_end]
            i_m = np.where(m_sub==m)
            i_m = i_m[0]

            # calculate wavelet
            CWT(np.squeeze(hc_sub[:,i_m]),l,m, fs=1)
