import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD, EEMD, visualisation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch 
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import scipy.io as scio
import csv
import random
import time

from smp.config import load_config
from smp.constants import EMD_LSTM_RANDOM_SEED
from smp.utils import set_random_seed

save_or_not = 0
config = load_config()
# lm = 1 # g30(9), g31(10), g3-3(15), g41(17), g5-2(29), g53(30), g5-3(31)
look_back_mat = [60, 126, 50, 50,\
                 25, 97, 150, 120, \
                 25, 43, 50, 50, \
                 25, 27, 50, 50, \
                 25, 40, 110, 120, \
                 150, 150, 120, 50, \
                 25, 78, 110, 50, \
                 25, 23, 120, 50, \
                 25, 76, 50, 50, \
                 25, 26, 150, 102, \
                 25, 150, 50, 50, \
                 25, 70, 120, 50, \
                 150, 150, 120, 50, \
                 120, 140, 120, 50, \
                 25, 53, 110, 50, \
                 25, 56, 110, 50, \
                 30, 90, 120, 50, \
                 25, 110, 88, 50, \
                 25, 110, 88, 90, \
                 25, 44, 150, 50, \
                 25, 110, 150, 50, \
                 160, 129, 50, 50, \
                 120, 150, 60, 50, \
                 25, 123, 50, 50, \
                 25, 140, 40, 50, \
                 25, 91, 90, 61, \
                 25, 150, 50, 60, \
                 50, 27, 30, 60, \
                 40, 29, 50, 50, \
                 150, 150, 50, 50, \
                 25, 150, 150, 60, \
                 25, 150, 60, 40, \
                 25, 123, 54, 40, \
                 36, 160, 150, 90, \
                 25, 30, 119, 90, \
                 30, 141, 120, 87] # lm=0 ~ 35

for lm in range(1,2):
    look_back_lst = look_back_mat[4*lm:4*lm+4]

    def moving_average(data, window_size):
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(data, window, mode='same')
        return smoothed

    # 导入数据
    file_dir = config.path("harmonic_coefficients")
    data_str = scio.loadmat(file_dir)
    # data_mat = data_str['sn']
    data_mat = data_str['save_var']
    data_lm = np.zeros((len(data_mat)-2,1))
    data0_t = np.array(range(0,len(data_mat)-2))
    cr_base = 1642
    l_cor = data_mat[0][lm]
    m_cor = data_mat[1][lm]
    for i in range(0,len(data_mat)-2):
        data_lm[i] = data_mat[i+2][lm]
        # data_lm[i] = data_mat[i][2]
        # data_lm[i] = data_mat[i]
    data = data_lm.flatten()
    dev = 141 # CR 2258 begins 2022:05:28, CR 2117 begins 2011:11:16
    data_raw = data
    data0_t = data0_t[:-dev]
    data = data[:-dev]
    # data = moving_average(data, 12)

    # 整个程序的运行计时
    start_time_tot = time.time()

    # EMD分解
    emd = EMD()
    IMFs = emd(data,max_imf=3)
    IMFs_sum = np.sum(IMFs, axis=0)

    # EMD分解绘图
    # plt.subplot(IMFs.shape[0]+1,1,1)
    # plt.plot(data,label='data')
    # plt.legend()
    # for i, IMF in enumerate(IMFs):
    #     if i < IMFs.shape[0]-1:
    #         plt.subplot(IMFs.shape[0]+1,1,i+2)
    #         plt.plot(IMF, label='IMF {}'.format(i+1))
    #         plt.legend()
    #     elif i == IMFs.shape[0]-1:
    #         plt.subplot(IMFs.shape[0]+1,1,i+2)
    #         plt.plot(IMF, label='Residual')
    #         plt.legend()
    # plt.suptitle('g_{}^{}'.format(int(l_cor), int(m_cor)))
    # plt.show()

    # 定义函数；生成训练集和测试集
    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i+look_back)]
            dataX.append(a)
            dataY.append(dataset[i+look_back])
        return np.array(dataX), np.array(dataY)

    # 定义函数：生成预测未来的输入数据集
    def create_future_inputs(dataset,look_back):
        dataX_ext = []
        for i in range(len(dataset)-look_back+1):
            a = dataset[i:(i+look_back)]
            dataX_ext.append(a)
        return np.array(dataX_ext)

    # 定义模型：LSTM
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

    # 定义预测步数
    future_step = 150

    # 建立空数组用于存储
    IMFs_model = np.zeros((IMFs.shape[0],len(data)+future_step))

    # 对每个IMF进行LSTM预测
    for i, IMF in enumerate(IMFs):
        if i<4:
            # 固定随机种子
            seed = EMD_LSTM_RANDOM_SEED
            set_random_seed(seed)
            
            IMF = pd.DataFrame(IMF)
            scaler = MinMaxScaler(feature_range=(0, 1))
            IMF = scaler.fit_transform(IMF)
            
            # 定义look_back
            look_back = look_back_lst[i]
            
            # 划分训练集、验证集、测试集
            IMFX, IMFY = create_dataset(IMF,look_back)
            IMF_t = np.array(range(0,len(IMFX)))
            train_size = int(len(IMF_t) * 0.6)
            val_size = int(len(IMF_t) * 0.2)
            test_size = len(IMF_t) - train_size - val_size
            
            train_t, val_t, test_t = IMF_t[:train_size], \
                IMF_t[:train_size+val_size], IMF_t
                # IMF_t[train_size:train_size+val_size], IMF_t[train_size+val_size:]
            trainX, valX, testX = IMFX[:train_size], \
                IMFX[:train_size+val_size], IMFX
                # IMFX[train_size:train_size+val_size], IMFX[train_size+val_size:]
            trainY, valY, testY = IMFY[:train_size], \
                IMFY[:train_size+val_size], IMFY
                # IMFY[train_size:train_size+val_size], IMFY[train_size+val_size:]

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

            # 定义模型参数
            input_dim = look_back
            hidden_dim = look_back
            num_layers = 3
            output_dim = 1
            lr = 0.01
            num_epochs = 2000
            loss_thres = 1e-5
            
            # 定义模型和损失函数
            model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # 定义早停策略
            patience = 100
            val_loss_min = np.Inf
            val_loss_thres = 0.003
            counter = 0
            # if i == 0:
            #     val_loss_thres = 0.005
            # if (lm in [1,9,25,49,81]) & (i == IMFs.shape[0]-1):
            #     val_loss_thres = 1e-7
            #     num_epochs = 4000
            #     loss_thres = 1e-7

            # 训练模型
            start_time = time.time()
            if i < IMFs.shape[0]-1:
                print('========== Training for IMF {} =========='.format(i+1))
            elif i == IMFs.shape[0]-1:
                print('========== Training for Residual ==========')
            for epoch in range(num_epochs):
                outputs = model(trainX)
                optimizer.zero_grad()
                loss = criterion(outputs, trainY)
                loss_np = loss.detach().numpy()
                loss.backward()
                optimizer.step()
                
                val_outputs = model(valX)
                val_loss = criterion(val_outputs[-val_size:], valY[-val_size:])
                
                # 训练集损失达到阈值以下，跳出循环
                if loss.item() < loss_thres:
                    torch.save(model.state_dict(), 'best_model.pkl')
                    print('Epoch [{}/{}], Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch+1, num_epochs, loss.item(), val_loss))
                    print("The loss value is reached")
                    break
                elif (epoch+1) % 100 == 0:
                    print('Epoch [{}/{}], Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch+1, num_epochs, loss.item(), val_loss))
                
                # 使用验证集进行早停策略
                if val_loss < val_loss_min:
                    counter = 0
                    val_loss_min = val_loss
                    torch.save(model.state_dict(), 'best_model.pkl')
                else:
                    counter += 1
                    if (counter >= patience) & (val_loss < val_loss_thres) & (loss.item() < 0.005):
                        print('Early stopping at Epoch [{}/{}], Train Loss: {:.5f}, Val Loss={:.5f}'.format(epoch+1, num_epochs, loss.item(), val_loss))
                        break
            print('Time Cost(s):', time.time()-start_time)
            
            # 加载最佳模型
            # model.load_state_dict(torch.load('best_model.pkl'))
            
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
            train_RMSE = np.sqrt(mean_squared_error(trainY, train_predict))
            val_RMSE = np.sqrt(mean_squared_error(valY[-val_size:], val_predict[-val_size:]))
            test_RMSE = np.sqrt(mean_squared_error(testY[-test_size:], test_predict[-test_size:]))
            aver_RMSE = (train_RMSE*train_size+val_RMSE*val_size+test_RMSE*test_size) / (train_size+val_size+test_size)
            print('Train RMSE: %.3f' % train_RMSE)
            print('Val RMSE: %.3f' % val_RMSE)
            print('Test RMSE: %.3f' % test_RMSE)
            print('Average RMSE: %.3f' % aver_RMSE)

            # 预测未来的时间序列
            future_t = np.array(range(0,future_step))
            future_predict = np.zeros((future_step,1))
            IMF_ext = IMF
            for j in range(future_step):
                IMFX_ext = create_future_inputs(IMF_ext, look_back)
                IMFX_ext = IMFX_ext.reshape(-1,1,look_back)
                IMFX_ext = torch.from_numpy(IMFX_ext).type(torch.Tensor)
                outputs_ext = model(IMFX_ext)
                future_predict[j] = outputs_ext[-1].detach().numpy()
                IMF_ext = np.append(IMF_ext,future_predict[j])
            future_predict = scaler.inverse_transform(future_predict)
            
            # 存储数据
            IMFs_model[i] = np.append(IMF[:look_back], np.append(train_predict.flatten(), \
                np.append(val_predict[-val_size:].flatten(), np.append(test_predict[-test_size:].flatten(), future_predict.flatten()))))
            
            # 模型结果绘图
            plt.subplot(IMFs.shape[0]+1,1,i+2)
            
            if i < IMFs.shape[0]-1:
                plt.plot(data0_t+cr_base,IMFs[i], label='IMF {}'.format(i+1),color='k')
            elif i == IMFs.shape[0]-1:
                plt.plot(data0_t+cr_base,IMFs[i], label='Residual',color='k')
            plt.plot(train_t+look_back+cr_base,train_predict, label='train predict',color='b')
            plt.plot(val_t[-val_size:]+look_back+cr_base,val_predict[-val_size:], label='val predict',color='m')
            plt.plot(test_t[-test_size:]+look_back+cr_base,test_predict[-test_size:], label='test predict',color='g')
            plt.plot(future_t+len(data0_t)+cr_base,future_predict, label='future predict',color='r')
            
            if save_or_not == 1:
                # np.savetxt(save_dir+str(i)+'_imf.csv',IMFs[i],delimiter=',')
                # np.savetxt(save_dir+str(i)+'_train.csv',train_predict,delimiter=',')
                # np.savetxt(save_dir+str(i)+'_val.csv',val_predict[-val_size:],delimiter=',')
                # np.savetxt(save_dir+str(i)+'_test.csv',test_predict[-test_size:],delimiter=',')
                # np.savetxt(save_dir+str(i)+'_future.csv',future_predict,delimiter=',')
                plt.close()

    data_model = np.sum(IMFs_model, axis=0)

    # 计时结束
    print('Total Time Cost(s):', time.time()-start_time_tot)

    # 预测结果绘图
    plt.subplot(IMFs.shape[0]+1,1,1)
    # plt.plot(data_predict,label='predict',color='r')
    plt.plot(np.array(range(0,len(data_model)))[max(look_back_lst):]+cr_base,data_model[max(look_back_lst):],label='model',color='r')
    plt.plot(np.array(range(0,len(data)))+cr_base,data,label='data',color='k')
    # plt.plot(train_t+look_back,data_train,label='train',color='b')
    # plt.plot(val_t[-val_size:]+look_back,data_val[-val_size:],label='val',color='m')
    # plt.plot(test_t[-test_size:]+look_back,data_test[-test_size:],label='test',color='g')
    # plt.legend()
    # plt.suptitle('g_{}^{}'.format(int(l_cor), int(m_cor)))
    plt.close()

    plt.figure()
    xmin, xmax = min(data0_t), max(data0_t)+future_step
    plt.hlines(0,xmin+cr_base,xmax+cr_base,color='pink',linestyles='dashed')
    # plt.plot(data_predict,label='predict',color='r')
    plt.plot(np.array(range(0,len(data_model)))[max(look_back_lst):]+cr_base,data_model[max(look_back_lst):],label='model',color='r')
    plt.plot(np.array(range(0,len(data_raw)))+cr_base,data_raw,label='data',color='k')
    # plt.plot(train_t+look_back,data_train,label='train',color='b')
    # plt.plot(val_t[-val_size:]+look_back,data_val[-val_size:],label='val',color='m')
    # plt.plot(test_t[-test_size:]+look_back,data_test[-test_size:],label='test',color='g')
    plt.legend()
    # plt.title('g_{}^{}'.format(int(l_cor), int(m_cor)))

    # 保存预测序列
    if save_or_not == 1:
        save_dir = config.ensure_output("model_output")
        save_file = 'No_' + str(lm) + '.csv'
        np.savetxt(save_dir / save_file, data_model[max(look_back_lst):], delimiter=',')
        
        save_file = 'g_' + str(int(l_cor)) + '_' + str(int(m_cor)) + '.csv'
        np.savetxt(save_dir / save_file, data_model[-future_step:], delimiter=',')
        
        save_png = 'g_' + str(int(l_cor)) + '_' + str(int(m_cor)) + '.png'
        plt.savefig(save_dir / save_png)
            
    # plt.show()
    plt.close()
