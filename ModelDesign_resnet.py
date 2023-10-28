# -*- coding: utf-8 -*-

import numpy as np
from mindspore import nn, ops
import math
# d_model = 128
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore
from mindspore import Tensor
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out

# MIMO-OFDM Parameters
SC_num = 120  # subcarrier number
Tx_num = 32  # Tx antenna number
Rx_num = 8  # Rx antenna number
sigma2_UE = 1e-6

class ConvBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, pad_mode="same", has_bias=False)
        self.flatten1 = nn.Flatten()
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = ops.ExpandDims()(x, 2)
        out = self.conv(x)
        out = self.flatten1(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResidualBlock(nn.Cell):
    def __init__(self, num_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(num_channels, num_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = None
        self.flatten1 = None
        if stride != 1:
            self.downsample = nn.Conv1d(num_channels, num_channels, kernel_size=1, stride=stride, has_bias=False)
            self.flatten1 = nn.Flatten()
        self.add = P.TensorAdd()
        # self.add = P.Add()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = ops.ExpandDims()(x, 2)
            identity = self.downsample(x)
            identity = self.flatten1(identity)
        out = self.add(out, identity)
        return out

class TempNet(nn.Cell):
    def __init__(self):
        super(TempNet, self).__init__()
        self.in_channels = 256
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=256, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.mish1 = nn.Mish()
        self.residual_blocks = nn.SequentialCell([
            ResidualBlock(256, stride=1),
            ResidualBlock(256, stride=2)
        ])
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        # self.fc = nn.Dense(in_channels=256, out_channels=Rx_num * Tx_num * SC_num * 2, weight_init='xavier_uniform')
        self.fc = nn.Dense(in_channels=256, out_channels=Tx_num * SC_num * 2, weight_init='xavier_uniform')
        self.droplayer = nn.Dropout(keep_prob=0.6)

    def construct(self, x):
        x = ops.ExpandDims()(x, 2)
        out = self.conv1(x)
        out = self.flatten1(out)
        out = self.bn1(out)
        out = self.mish1(out)
        out = self.residual_blocks(out)
        out = self.fc(out)
        out = self.droplayer(out)
        # out = out.view(-1, 8, 120, 32, 2)
        out = out.reshape(-1, 120, 32, 2)
        return out


class ChannelResponseNet(nn.Cell):
    def __init__(self):
        super(ChannelResponseNet, self).__init__()
        # 在这里定义您的第一个神经网络结构
        # 可以包括多个卷积层、全连接层等
        self.layer1 = nn.Dense(in_channels=3, out_channels=50, weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.layer2 = nn.Dense(in_channels=50, out_channels=100, weight_init='xavier_uniform')
        self.bn2 = nn.BatchNorm1d(100, eps=0.001, momentum=0.99)
        self.layer3 = nn.Dense(in_channels=100, out_channels=Rx_num * Tx_num * SC_num * 2, weight_init='xavier_uniform')

    def construct(self, x):
        # 在这里实现前向传播逻辑
        # 输入x为xyz坐标，输出无线信道响应
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        y = self.layer3(x)
        y = y.reshape((-1, Rx_num, Tx_num, SC_num, 2))
        return y  # 这里的y即为无线信道响应

class PrecodingVectorNet(nn.Cell):
    def __init__(self):
        super(PrecodingVectorNet, self).__init__()
        # 在这里定义您的第二个神经网络结构
        # 可以包括多个卷积层、全连接层等
        self.layer1 = nn.Dense(in_channels=(SC_num * Tx_num * Rx_num * 2+3), out_channels=20000, weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm1d(20000, eps=0.001, momentum=0.99)
        self.layer2 = nn.Dense(in_channels=20000, out_channels=10000, weight_init='xavier_uniform')
        self.bn2 = nn.BatchNorm1d(10000, eps=0.001, momentum=0.99)
        self.layer3 = nn.Dense(in_channels=10000, out_channels=Tx_num * SC_num * 2, weight_init='xavier_uniform')

    def construct(self, x, CSI):
        # 在这里实现前向传播逻辑
        # 输入x为xyz坐标和channel_response为无线信道响应，输出赋形向量
        CSI = ops.Reshape()(CSI, (-1, SC_num * Tx_num * Rx_num * 2))  # 将 CSI 重新构造为二维张量，形状为 [None, 120 * 32 * 8 * 2]
        input_data = ops.Concat(axis=1)([CSI, x])  # 在第二个维度上拼接地理坐标和 CSI 数据

        x = self.layer1(input_data)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        z = self.layer3(x)
        return z  # 这里的z即为赋形向量

class NeuralNetwork(nn.Cell):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.CSInet = ChannelResponseNet()
        self.PVnet = PrecodingVectorNet()

    def construct(self, x):
        out = self.CSInet(x)
        out = self.PVnet(x, out)
        out = out.reshape((-1, SC_num, Tx_num, 2))
        return out



def RadioMap_Model(data, net):
    P_est = net(data).asnumpy()
    P_est = P_est[:, :, :, 0] + 1j * P_est[:, :, :, 1]
    P_est = np.expand_dims(P_est, axis=-1)
    return P_est

    # CSI_est = net(data)
    # HH_est = ops.reshape(CSI_est, (-1, Rx_num, Tx_num, SC_num, 2))
    # HH_complex_est = ops.Complex()(HH_est[:, :, :, :, 0], HH_est[:, :, :, :, 1])
    # HH_complex_est = ops.transpose(HH_complex_est, (0, 3, 1, 2))
    # MatDiag, MatRx, MatTx = np.linalg.svd(HH_complex_est.asnumpy(), full_matrices=True)
    #
    # PrecodingVector = MatTx[:, :, :, 0]
    # PrecodingVector = np.reshape(PrecodingVector, (-1, SC_num, Tx_num, 1))
    # return PrecodingVector

def EqChannelGainJoint(Channel_BS1, Channel_BS2, PrecodingVector_BS1, PrecodingVector_BS2):
    # The authentic CSI
    # HH1
    HH1 = np.reshape(Channel_BS1, (-1, Rx_num, Tx_num, SC_num, 2)) ## Rx, Tx, Subcarrier, RealImag
    HH1_complex = HH1[:,:,:,:,0] + 1j * HH1[:,:,:,:,1]  ## Rx, Tx, Subcarrier
    HH1_complex = np.transpose(HH1_complex, [0,3,1,2])

    # HH2
    HH2 = np.reshape(Channel_BS2, (-1, Rx_num, Tx_num, SC_num, 2))  # Rx, Tx, Subcarrier, RealImag
    HH2_complex = HH2[:,:,:,:,0] + 1j * HH2[:,:,:,:,1]  # Rx, Tx, Subcarrier
    HH2_complex = np.transpose(HH2_complex, [0,3,1,2])

    # Power Normalization of the precoding vector
    # PrecodingVector1
    Power = np.matmul(np.transpose(np.conj(PrecodingVector_BS1), (0, 1, 3, 2)), PrecodingVector_BS1)
    Power = np.sum(Power.reshape(-1, SC_num), axis=-1).reshape(-1, 1)
    Power = np.matmul(Power, np.ones((1, SC_num)))
    Power = Power.reshape(-1, SC_num, 1, 1)
    PrecodingVector_BS1 = np.sqrt(SC_num) * PrecodingVector_BS1 / np.sqrt(Power)

    # PrecodingVector2
    Power = np.matmul(np.transpose(np.conj(PrecodingVector_BS2), (0, 1, 3, 2)), PrecodingVector_BS2)
    Power = np.sum(Power.reshape(-1, SC_num), axis=-1).reshape(-1, 1)
    Power = np.matmul(Power, np.ones((1, SC_num)))
    Power = Power.reshape(-1, SC_num, 1, 1)
    PrecodingVector_BS2 = np.sqrt(SC_num) * PrecodingVector_BS2 / np.sqrt(Power)

    # Effective channel gain
    R = np.matmul(HH1_complex, PrecodingVector_BS1) + np.matmul(HH2_complex, PrecodingVector_BS2)
    R_conj = np.transpose(np.conj(R), (0, 1, 3, 2))
    h_sub_gain = np.matmul(R_conj, R)
    h_sub_gain = np.reshape(np.absolute(h_sub_gain), (-1, SC_num))  # channel gain of SC_num subcarriers
    return h_sub_gain

# Data Rate
def DataRate(h_sub_gain, sigma2_UE):  ### Score
    SNR = h_sub_gain / sigma2_UE
    Rate = np.log2(1 + SNR)  ## rate
    Rate_OFDM = np.mean(Rate, axis=-1)  ###  averaging over subcarriers
    Rate_OFDM_mean = np.mean(Rate_OFDM)  ### averaging over CSI samples
    return Rate_OFDM_mean
