# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as scio

import mindspore as ms
import mindspore.dataset as ds

from data.ModelDesign import *

ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE)

# DL Parameters
ckpt_path1 = './modelBS1.ckpt'
ckpt_path2 = './modelBS2.ckpt'

# eval
# Parameters
batch_size = 16
sigma2_UE = 1e-6

# Load data
data = scio.loadmat('./data_test_0.mat')
location_data = data['0'].astype(np.float32)
channel_data_1 = data['1'].astype(np.float32)
channel_data_2 = data['2'].astype(np.float32)

data = {'1': location_data, '2': channel_data_1, '3': channel_data_2}
dataset = ds.NumpySlicesDataset(data=data, column_names=["loc", "CSI_1", "CSI_2"])
datasize = dataset.get_dataset_size()
print('datasize eval: ', datasize)
dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
step = dataset.get_dataset_size()
print('step eval: ', step)


# Load model
net1 = NeuralNetwork()
# parameter = ms.load_checkpoint(ckpt_path1)
# ms.load_param_into_net(net1, parameter)

net2 = NeuralNetwork()
# parameter = ms.load_checkpoint(ckpt_path2)
# ms.load_param_into_net(net2, parameter)

data_rates = []
net1.set_train(False)
for iter, data in enumerate(dataset.create_dict_iterator()):
    result1 = RadioMap_Model(data['loc'], net1)  # StubTensor:(64, 120, 32, 1)
    result2 = RadioMap_Model(data['loc'], net2)  # StubTensor:(64, 120, 32, 1)

    # Calculate the score
    d1 = data['CSI_1'].asnumpy()
    d2 = data['CSI_2'].asnumpy()



    SubCH_gain_codeword = EqChannelGainJoint(d1, d2, result1, result2)

    data_rate = DataRate(SubCH_gain_codeword, sigma2_UE)
    data_rates.append(data_rate)
    if iter % 100 == 0:
        print(f'{iter} step')
        print('The score is %f bps/Hz' % data_rate)

score = sum(data_rates)/len(data_rates)

print('The mean score is %f bps/Hz' % score)
