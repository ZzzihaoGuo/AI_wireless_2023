# -*- coding: utf-8 -*-

import numpy as np
from mindspore import nn, ops
from mindspore import Tensor
from mindspore import dtype as mstype
# MIMO-OFDM Parameters
SC_num = 120  # subcarrier number
Tx_num = 32  # Tx antenna number
Rx_num = 8  # Rx antenna number
sigma2_UE = 1e-6
d_model = 256
import math
import mindspore
import mindspore.nn as nn
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out


class Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1,
                 group=1, has_bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, pad_mode, padding, dilation, group, has_bias,
                         weight_init='normal', bias_init='zeros')
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        # self.weight = Parameter(initializer(HeUniform(math.sqrt(5)), self.weight.shape), name='weight')
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))


class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias,
                         activation=activation)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mindspore.float32,
                 padding_idx=None):
        if embedding_table == 'normal':
            embedding_table = Normal(1.0)
        super().__init__(vocab_size, embedding_size, use_one_hot, embedding_table, dtype, padding_idx)

    @classmethod
    def from_pretrained_embedding(cls, embeddings: Tensor, freeze=True, padding_idx=None):
        rows, cols = embeddings.shape
        embedding = cls(rows, cols, embedding_table=embeddings, padding_idx=padding_idx)
        embedding.embedding_table.requires_grad = not freeze
        return embedding


class ScaledDotProductAttention(nn.Cell):
    def __init__(self, d_k):
        super().__init__()
        self.scale = Tensor(d_k, mindspore.float32)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, Q, K, V):
        K = K.transpose((0, 1, 3, 2))
        scores = ops.matmul(Q, K) / ops.sqrt(
            self.scale)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        # scores = scores.masked_fill(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = self.softmax(scores)
        context = ops.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Cell):
    def __init__(self, d_model, d_k, n_heads):
        super().__init__()
        self.d_k = d_k
        self.n_heads = n_heads
        self.W_Q = Dense(d_model, d_k * n_heads)
        self.W_K = Dense(d_model, d_k * n_heads)
        self.W_V = Dense(d_model, d_k * n_heads)
        self.linear = Dense(n_heads * d_k, d_model)
        self.layer_norm = nn.LayerNorm((d_model,), epsilon=1e-5)
        self.attention = ScaledDotProductAttention(d_k)

    def construct(self, Q, K, V):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.shape[0]
        q_s = self.W_Q(Q).view((batch_size, -1, self.n_heads, self.d_k))
        k_s = self.W_K(K).view((batch_size, -1, self.n_heads, self.d_k))
        v_s = self.W_V(V).view((batch_size, -1, self.n_heads, self.d_k))
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = q_s.transpose((0, 2, 1, 3))  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = k_s.transpose((0, 2, 1, 3))  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = v_s.transpose((0, 2, 1, 3))  # v_s: [batch_size x n_heads x len_k x d_v]

        # attn_mask = attn_mask.expand_dims(1)
        # attn_mask = ops.tile(attn_mask, (1, self.n_heads, 1, 1))  # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = self.attention(q_s, k_s, v_s)
        context = context.transpose((0, 2, 1, 3)).view(
            (batch_size, -1, self.n_heads * self.d_k))  # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn  # output: [batch_size x len_q x d_model]

class PoswiseFeedForward(nn.Cell):
    def __init__(self, d_ff, d_model):
        super().__init__()
        self.conv1 = Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm((d_model,), epsilon=1e-5)
        self.relu = nn.ReLU()

    def construct(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = inputs.transpose((0, 2, 1))
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = output.transpose((0, 2, 1))
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Cell):
    def __init__(self, d_model, d_k, n_heads, d_ff):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, n_heads)
        self.pos_ffn = PoswiseFeedForward(d_ff, d_model)

    def construct(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class Encoder(nn.Cell):
    def __init__(self, d_model, d_k, n_heads, d_ff, n_layers):
        super().__init__()
        # self.src_emb = Embedding(src_vocab_size, d_model)
        # self.pos_emb = Embedding.from_pretrained_embedding(get_sinusoid_encoding_table(src_len + 1, d_model),
        #                                                    freeze=True)
        self.layers = nn.CellList([EncoderLayer(d_model, d_k, n_heads, d_ff) for _ in range(n_layers)])
        # temp positional indexes
        # self.pos = Tensor([[1, 2, 3, 4, 0]])

    def construct(self, enc_inputs):
        # enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(self.pos)
        enc_self_attns = []
        for layer in self.layers:
            enc_inputs, enc_self_attn = layer(enc_inputs)
            enc_self_attns.append(enc_self_attn)
        return enc_inputs


class NeuralNetwork(nn.Cell):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_linear_1 = nn.Dense(3, d_model)
        # self.input_linear_2 = nn.Dense(3, d_model)
        # self.input_linear_3 = nn.Dense(3, d_model)
        n_heads = 4
        n_layers = 20
        d_ff = d_model*4
        # self.transformer_each = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer_1 = Encoder(d_model, d_model//n_heads, n_heads, d_ff, n_layers)
        # self.transformer_2 = Encoder(d_model, d_model//n_heads, n_heads, d_ff, n_layers)
        # self.transformer_3 = Encoder(d_model, d_model//n_heads, n_heads, d_ff, n_layers)
        
        self.out_1 = nn.Dense(d_model, 64)
        # self.out_2 = nn.Dense(d_model, 64)
        # self.out_3 = nn.Dense(d_model, 64)
    def construct(self, x):
        x = ops.ExpandDims()(x, -1)
        x = x.repeat(SC_num, -1).transpose((0, 2, 1))
        
        # x1 = x.reshape(-1, 60, d_model)
        # x2 = x[:,40:80,:]
        # x3 = x[:,80:120,:]

        x1 = self.input_linear_1(x)
        # x2 = self.input_linear_2(x2)
        # x3 = self.input_linear_3(x3)
        
        x1 = self.transformer_1(x1)
        # x2 = self.transformer_2(x2)
        # x3 = self.transformer_3(x3)
        # op = ops.Concat(1)
        # z = op((x1, x2, x3))
        z = self.out_1(x1)
        # x2 = self.out_2(x2)
        # x3 = self.out_3(x3)
        
        # z = ops.cat((x1, x2, x3), axis=1)
        # op = ops.Concat(1)
        # z = op((x1, x2, x3))

        # x2=   self.out2(x)
        # x=x.reshape(-1,120,32,2)
        # z = z.reshape(-1, 120, 64)
        z = z.reshape(-1, 120, 32, 2)
        # x2=x2.reshape(-1,120,32,2)
        return z  # ,x2


class CombineNetwork(nn.Cell):
    def __init__(self):
        super(CombineNetwork, self).__init__()
        self.net_1 = NeuralNetwork()
        self.net_2 = NeuralNetwork()

    def construct(self, x):
        x1 = self.net_1(x)
        x2 = self.net_2(x)

        return x1, x2


def RadioMap_Model(data, net):
    CSI_est = net(data).asnumpy()
    CSI_est = CSI_est[:, :, :, 0] + 1j * CSI_est[:, :, :, 1]

    CSI_est = np.expand_dims(CSI_est, axis=-1)
    return CSI_est


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
