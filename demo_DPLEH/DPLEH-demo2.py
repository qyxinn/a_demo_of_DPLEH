
#new data  实际数据2 gbsg
#age	孕激素受体	雌激素受体	淋巴结转移数量	绝经状态	肿瘤大小	肿瘤分级	激素治疗	time	cens

import scipy.stats
from pandas.core.frame import DataFrame
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import itertools
import pandas as pd
from mylossfun import eaftloss  #
from mycindex import *
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import RepeatedKFold


ibs2 = []
ibs2_std = []
cindex2 = []
cindex2_std = []

path = 'D:/Pycharm/PyCharm Community Edition 2022.2.1/pyworks/SuppCode/data/gbsg_data.csv'
data = pd.read_csv(path)

EPOCH = 500  # 一个epoch表示： 所有的数据送入网络中， 完成了一次前向计算 + 反向传播的过程
rept = 1
betas = (0.91, 0.999)
Riemann_sum_gap = .5
integral_id = 0
LR_decay = 0.0

LR = 0.001
wt_decay = 0.05
layer_a = 2
neuron_n = 256
dropout_pr = 0
rs = 1
# 5-fold CV 5折交叉验证  #step1:将数据集分为5堆；# step2:选取一堆作为测试集，另外四堆作为训练集；# step3:共重复step2 五次，每次选取的训练集不同。
kf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=rs)

val_loss_min = np.zeros((7, 10))
val_loss_percentile = np.zeros((7, 10))

Test_c_index = np.zeros((7, 10))
Time_mean_1 = np.zeros((7, 10))
Time_std_1 = np.zeros((7, 10))

AP_mean = np.zeros((7, 10))
AP_std = np.zeros((7, 10))
AP_rate = np.zeros((7, 10))
AFT_rate = np.zeros((7, 10))
IBS = np.zeros((7, 10))

cv_ind = -1
K = EPOCH
beta1 =[]

for train_index, test_index in kf.split(data):
    cv_ind = cv_ind + 1
    train_index, test_index = pd.Index(train_index), pd.Index(test_index)  # index提取列
    df_train, df_test = data.iloc[train_index, :], data.iloc[test_index]
    df_val = df_train.sample(frac=0.2, random_state=rs)  # frac   设置抽样比例 random_state   设置随机数种子rs=1
    df_train = df_train.drop(df_val.index)  # 删除

    #####
    cols_standardize = [ '1', '2','3','4']
    cols_leave = ['5', '6', '7','8']

    standardize = [([col], StandardScaler()) for col in cols_standardize]  # StandardScaler类是一个用来讲数据进行归一化和标准化的类
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(standardize + leave)

    x_train = x_mapper.fit_transform(df_train).astype('float32')  # 适合数据，然后转换它
    x_val = x_mapper.transform(df_val).astype('float32')  # 通过居中和缩放执行标准化
    x_test = x_mapper.transform(df_test).astype('float32')

    delta_train = df_train['delta'].values.astype('float32')
    time_train = df_train['time'].values.astype('float32')

    delta_test = df_test['delta'].values.astype('float32')
    time_test = df_test['time'].values.astype('float32')

    delta_val = df_val['delta'].values.astype('float32')
    time_val = df_val['time'].values.astype('float32')

    x_train = torch.from_numpy(x_train)  # 从numpy数组创建一个张量，数组和张量共享相同内存
    y_train = x_train[:, 0:4]  # 连续
    z_train = x_train[:, 4:]  # 离散
    delta_train = torch.from_numpy(delta_train)
    time_train = torch.from_numpy(time_train)

    x_val = torch.from_numpy(x_val)
    y_val = x_val[:, 0:4]  # 连续
    z_val = x_val[:, 4:]  # 离散
    delta_val = torch.from_numpy(delta_val)
    time_val = torch.from_numpy(time_val)

    x_test = torch.from_numpy(x_test)
    y_test = x_test[:, 0:4]  # 连续
    z_test = x_test[:, 4:]  # 离散
    delta_test = torch.from_numpy(delta_test)
    time_test = torch.from_numpy(time_test)

    n_tx, feature_px = x_train.size()
    n_ty, feature_py = y_train.size()
    n_tz, feature_pz = z_train.size()
    C_index_TEST1 = []

    kk = -1
    RES = np.zeros((4, 6))

    for t_seed in range(rept):
        kk = kk + 1
        # neural network structure
        torch.manual_seed(t_seed)  # 设置 CPU 生成随机数的 种子 ，方便下次复现实验结果。为 CPU 设置 种子 用于生成随机数，以使得结果是确定的。

        class EAFTNet(nn.Module):

            def __init__(self, feature_p, neuron_n, layer_a, dropout_pr):
                super(EAFTNet, self).__init__()  # super().__init__() 就是调用父类的init方法
                self.p = feature_p
                self.a = layer_a
                self.n = neuron_n
                self.dp = dropout_pr

                self.f1 = nn.Linear(self.p, self.n)  # nn.Linear(输入层数，输出层数)：用于设置网络中的全连接层，需要注意的是全连接层的输入与输出都是二维张量
                self.f2 = nn.Linear(self.n, self.n)

                self.g1 = nn.Linear(self.n, 1)
                self.dropout = nn.Dropout(
                    p=self.dp)  # 防止或减轻过拟合而使用的函数，它一般用在全连接层Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了

            def forward(self, x):
                x = torch.selu(self.f1(x))  # 激活函数  将输入信号的总和转换为输出信号
                x = self.dropout(x)

                for i in range(self.a):
                    x = torch.selu(self.f2(x))
                    x = self.dropout(x)

                out = self.g1(x)

                return out

        class LinearNet(nn.Module):
            def __init__(self, feature_p, neuron_n, layer_a, dropout_pr):
                super(LinearNet, self).__init__()
                self.p = feature_p
                self.a = layer_a
                self.n = neuron_n
                self.dp = dropout_pr

                self.g3 = nn.Linear(self.p, 1, bias=False)  # 输入和输出的维度都是1

            def forward(self, x):
                out = self.g3(x)
                return out

        netx = EAFTNet(feature_px, neuron_n, layer_a, dropout_pr)
        nety = EAFTNet(feature_py, neuron_n, layer_a, dropout_pr)
        netz = LinearNet(feature_pz, neuron_n, layer_a, dropout_pr)

        ### training neural network
        Loss_train = torch.zeros(1, K)
        Loss_val = torch.zeros(1, K)
        c_val = []
        c_test = np.zeros(K)

        loss_temp = 1000.

        for t in range(K):
            learning_rate = LR / (1 + t * LR_decay)
            optimizer = torch.optim.AdamW(itertools.chain(netx.parameters(), nety.parameters(), netz.parameters()),
                              lr=learning_rate, betas=betas, weight_decay=wt_decay)

            outx = netx(x_train)  # h1
            outy = nety(y_train)  # h2
            outz = netz(z_train)  # beta*z
            train = [outx, outy, outz]
            out = torch.cat(train, dim=1)

            loss = eaftloss(out, time_train, delta_train)  ####
            Loss_train[0, t] = loss

            if t % 20 == 0:
                print('epoch: {}, loss: {:.4}'.format(t, loss))

            out_valx = netx(x_val)  # h1
            out_valy = nety(y_val)  # h2
            out_valz = netz(z_val)  # beta*z
            val = [out_valx, out_valy, out_valz]
            out_val = torch.cat(val, dim=1)

            loss_val = eaftloss(out_val, time_val, delta_val)
            Loss_val[0, t] = loss_val

            # find minimum value of loss for validation set
            if loss_val.item() <= loss_temp:
                tt = t

                out_train = out
                out_testx = netx(x_test)
                out_testy = nety(y_test)
                out_testz = netz(z_test)
                test = [out_testx, out_testy, out_testz]
                out_test = torch.cat(test, dim=1)

                loss_temp = loss_val.item()

            if (torch.isnan(out[0, 0]) or torch.isnan(out_val[0, 0])):
                print('Break due to NaN at step:', t)
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for param in netz.parameters():
            print(param)

        print("beta bias:")
        for name in netz.state_dict():  # 获得beta和bias
            print(name)  # g3.weight g3.bias
            print(netz.state_dict()[name])

        betawe = netz.state_dict()['g3.weight'].tolist()
        betaw = betawe[0]
        betaw = betawe[0]
        beta1.append(betaw)

        c_index_test, IBS[cv_ind, kk] = C_index(out_train, time_train, delta_train, out_test, time_test, delta_test,
                                                Riemann_sum_gap,
                                                integral_id)
        Test_c_index[cv_ind, kk] = c_index_test
        #
        ##########################

Test_c_index[5, :] = np.mean(Test_c_index[range(5), :], 0)
Test_c_index[6, :] = np.std(Test_c_index[range(5), :], 0)

IBS[5, :] = np.mean(IBS[range(5), :], 0)
IBS[6, :] = np.std(IBS[range(5), :], 0)

ibs2.append(IBS[5, 0])
ibs2_std.append(IBS[6, 0])
cindex2.append(Test_c_index[5, 0])
cindex2_std.append(Test_c_index[6, 0])

ibsmean = IBS[5, 0]
ibs2.append(ibsmean)

cmean = Test_c_index[5, 0]
cindex2.append(cmean)


print('integrated Brier Score')
print(IBS)
print('C-index:')
print(Test_c_index)
print("BIAS:")
print(beta1)
print('ibs:')
print(np.mean(ibs2))
print(np.mean(ibs2_std))
print('cindex:')
print(np.mean(cindex2))
print(np.mean(cindex2_std))
print("bias:")
print(np.mean(beta1, axis=0))





