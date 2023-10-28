import numpy as np
from typing import Any, Union
import torch
from torch import nn, Tensor


# 定义常量
Rx_num = 4  # 接受天线数量
Tx_num = 32  # 发射天线数量
SC_num = 128  # 子载波数量
sigma2_UE = 0.1  # 评估时的噪声的平方
VERSION = 2


# 一个通用的, 带skip connect的MLP模块
class MLP(nn.ModuleList):
    def __init__(self, channels, skips=None, use_bn=True, act: Any = nn.GELU, dropout=0.):
        super().__init__()
        self.num_layers = len(channels) - 1
        if skips is None:
            skips = {}
        self.skips = skips
        self.channels = channels
        for i in range(1, self.num_layers + 1):
            in_channels = channels[i - 1] + (channels[skips[i]] if i in skips else 0)
            layers = [nn.Linear(in_channels, channels[i])]
            if i < self.num_layers:
                if use_bn:
                    layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(act())
            if i + 1 == self.num_layers and dropout > 0:
                layers.append(nn.Dropout(dropout, inplace=True))
            self.append(nn.Sequential(*layers))

    def forward(self, x):
        xs = [x]
        for i in range(self.num_layers):
            if i + 1 in self.skips:
                x = torch.cat([xs[self.skips[i + 1]], x], dim=-1)
            x = self[i](x)
            xs.append(x)
        return x


class ANN_TypeI(nn.Module):  ## Neural Network (input:location, output: the estimated CSI)
    def __init__(self, input_dim=3, version=1):
        super(ANN_TypeI, self).__init__()
        self.version = version

        if version == 1:
            out_dim = Rx_num * Tx_num * SC_num * 2
        else:
            out_dim = 1 * Tx_num * SC_num * 2
        self.net = MLP(
            [input_dim, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, out_dim],
            # skips={3: 1, 5: 3, 7: 5, 9: 7},
            act=nn.GELU,
            # dropout=0.1
        )

    def forward(self, p):
        output = self.net(p)
        if self.version == 1:
            output = output.reshape((-1, Rx_num, Tx_num, SC_num, 2))
        else:
            output = output.reshape((-1, SC_num, Tx_num, 1, 2))
        return output


def DownPrecoding(channel_est: Union[np.ndarray, Tensor]):
    if isinstance(channel_est, Tensor):
        ### estimated channel
        HH_est = torch.reshape(channel_est, (-1, Rx_num, Tx_num, SC_num, 2))  ## Rx, Tx, Subcarrier, RealImag
        HH_complex_est = torch.complex(HH_est[:, :, :, :, 0], HH_est[:, :, :, :, 1])  ## Rx, Tx, Subcarrier
        HH_complex_est = torch.permute(HH_complex_est, [0, 3, 1, 2])

        ### precoding based on the estimated channel
        MatRx, MatDiag, MatTx = torch.linalg.svd(HH_complex_est, full_matrices=True)  ## SVD
        PrecodingVector = torch.conj(MatTx[:, :, 0, :])  ## The best eigenvector (MRT transmission)
        PrecodingVector = torch.reshape(PrecodingVector, (-1, SC_num, Tx_num, 1))
    else:
        ### estimated channel
        HH_est = np.reshape(channel_est, (-1, Rx_num, Tx_num, SC_num, 2))  ## Rx, Tx, Subcarrier, RealImag
        HH_complex_est = HH_est[:, :, :, :, 0] + 1j * HH_est[:, :, :, :, 1]  ## Rx, Tx, Subcarrier
        HH_complex_est = np.transpose(HH_complex_est, [0, 3, 1, 2])

        ### precoding based on the estimated channel
        MatRx, MatDiag, MatTx = np.linalg.svd(HH_complex_est, full_matrices=True)  ## SVD
        PrecodingVector = np.conj(MatTx[:, :, 0, :])  ## The best eigenvector (MRT transmission)
        PrecodingVector = np.reshape(PrecodingVector, (-1, SC_num, Tx_num, 1))
    return PrecodingVector


class Trans_encoder(nn.Module):
    def __init__(self):
        super(Trans_encoder, self).__init__()
        d_model = 128
        hidden_dim = d_model * 4
        nhead = 4
        num_layers = 1
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, hidden_dim, dropout=0.1),
            num_layers
        )

        self.linear_up = nn.Linear(Tx_num*2, d_model)
        self.linear_down = nn.Linear(d_model, 64)

        self.net_down = MLP(
            [d_model, 32],
            # skips={3: 1, 5: 3, 7: 5, 9: 7},
            act=nn.GELU,
            # dropout=0.1
        )

        self.net_up = MLP(
            [32, d_model],
            # skips={3: 1, 5: 3, 7: 5, 9: 7},
            act=nn.GELU,
            # dropout=0.1
        )

    def forward(self, g):
        # x = x.repeat((128, 1, 1)).permute(1, 0, 2)
        g = g.reshape((g.size(0), SC_num, Tx_num*2))
        g = self.linear_up(g)

        g = self.transformer_encoder(g)
        mid = self.net_down(g)
        return mid

class Trans_decoder(nn.Module):
    def __init__(self):
        super(Trans_decoder, self).__init__()
        d_model = 128
        hidden_dim = d_model * 4
        nhead = 4
        num_layers = 1
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, hidden_dim, dropout=0.1),
            num_layers
        )

        self.transformer_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, hidden_dim, dropout=0.1),
            num_layers
        )
        self.linear_up = nn.Linear(Tx_num*2, d_model)
        self.linear_down = nn.Linear(d_model, 64)

        self.net_down = MLP(
            [d_model, 32],
            # skips={3: 1, 5: 3, 7: 5, 9: 7},
            act=nn.GELU,
            # dropout=0.1
        )

        self.net_up = MLP(
            [32, d_model],
            # skips={3: 1, 5: 3, 7: 5, 9: 7},
            act=nn.GELU,
            # dropout=0.1
        )

    def forward(self, mid):
        output = self.net_up(mid)
        output = self.transformer_encoder(output)
        output = self.linear_down(output)
        output = output.reshape((output.shape[0], SC_num, Tx_num, 1, 2))

        return output


class RadioMap_Model_TypeI_pre_train(nn.Module):  ### Generate RadioMapI (Input:location, Output:beamforming vector)
    def __init__(self, input_dim=3, split_size=1024, version=VERSION):
        super(RadioMap_Model_TypeI_pre_train, self).__init__()
        self.version = version
        self.ann = ANN_TypeI(input_dim, version)  ## Neural Network (input:location, output: the estimated CSI)
        self.split_size = split_size  # 分批计算, 避免OOM
        self.transformer_encoder = Trans_encoder()
        self.transformer_decoder = Trans_decoder()

    def forward(self, positions: Tensor, g, stage_1=True, stage_2=True, train=True):
        results = []
        results_mid = []
        for p in positions.split(self.split_size, dim=0):
            if stage_1 and train:
                mid = self.transformer_encoder(g)
                x = self.transformer_decoder(mid)
            elif stage_1 and not train:
                p = p.unsqueeze(1).repeat(1, 128, 1)
                x = self.transformer_decoder(p)
            else:
                x = p
            # x = self.ann(p) if stage_1 else p
            if stage_2:
                if self.version == 1:
                    x = DownPrecoding(x)
                else:
                    x = torch.complex(x[..., 0], x[..., 1])

            results.append(x)
            if stage_1 and train:
                results_mid.append(mid)

        results = torch.cat(results, dim=0)
        if stage_1 and train:
            results_mid = torch.cat(results_mid, dim=0)
            return results, results_mid
        else:
            return results


def EqChannelGain(channel, PrecodingVector):
    ### The authentic CSI
    HH = np.reshape(channel, (-1, Rx_num, Tx_num, SC_num, 2))  ## Rx, Tx, Subcarrier, RealImag
    HH_complex = HH[:, :, :, :, 0] + 1j * HH[:, :, :, :, 1]  ## Rx, Tx, Subcarrier
    HH_complex = np.transpose(HH_complex, [0, 3, 1, 2])

    ### Power Normalization of the precoding vector
    Power = np.matmul(np.transpose(np.conj(PrecodingVector), (0, 1, 3, 2)), PrecodingVector)
    PrecodingVector = PrecodingVector / np.sqrt(Power)

    ### Effective channel gain
    R = np.matmul(HH_complex, PrecodingVector)
    R_conj = np.transpose(np.conj(R), (0, 1, 3, 2))
    h_sub_gain = np.matmul(R_conj, R)
    h_sub_gain = np.reshape(np.absolute(h_sub_gain), (-1, SC_num))  ### channel gain of SC_num subcarriers
    return h_sub_gain


def DataRate(h_sub_gain, sigma2_UE):  ### Score
    SNR = h_sub_gain / sigma2_UE
    Rate = np.log2(1 + SNR)  ## rate
    Rate_OFDM = np.mean(Rate, axis=-1)  ###  averaging over subcarriers
    Rate_OFDM_mean = np.mean(Rate_OFDM)  ### averaging over CSI samples
    return Rate_OFDM_mean


class DataRateLoss(nn.Module):
    def __init__(self, sigma2_UE=0.1):
        super().__init__()
        self.sigma2_UE = sigma2_UE

    def forward(self, h: Tensor, H: Tensor):
        if not torch.is_complex(H):
            H = torch.complex(H[..., 0], H[..., 1])
        H = torch.permute(H, (0, 3, 1, 2))  # shape: [N, SC, Rx, Tx]

        if not torch.is_complex(h):
            h = torch.complex(h[..., 0], h[..., 1])
        h = h / (h.mH @ h).sqrt()  # shape: [N, SC, Tx, 1]

        R = H @ h  # shape: [N, SC, Rx, 1]
        h_sub_gain = R.mH @ R  # shape: [N, SC, 1, 1]
        h_sub_gain = h_sub_gain.abs().view(-1, SC_num)

        rate = torch.log2(1 + h_sub_gain / self.sigma2_UE).mean()
        return 5.83 - rate
