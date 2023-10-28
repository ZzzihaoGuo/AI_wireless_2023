# -*- coding: utf-8 -*-

import time, os
import collections
import scipy.io as scio
import mindspore.numpy as mnp
import numpy as np
import mindspore as ms
from mindspore import nn, ops
import mindspore.ops.operations as operations
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
import mindspore.dataset as ds
from mindspore import save_checkpoint
import matplotlib.pyplot as plt
from ModelDesign import *
from mindspore import amp
import zipfile

##############示例中相关库的import##################
import os
import argparse
import json
import moxing as mox
from mindspore.train.callback import Callback
import zipfile
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
import time
from openi import openi_multidataset_to_env as DatasetToEnv
from openi import env_to_openi
from openi import EnvToOpenIEpochEnd

def file_display(filepath):
    for each in os.listdir(filepath): #得出文件的绝对路径
        absolute_path = os.path.join(filepath,each)
        is_file = os.path.isfile(absolute_path) #判断文件或目录得出布尔值
        if is_file:
            print(absolute_path)
        else:file_display(absolute_path)
 
parser = argparse.ArgumentParser(description='MindSpore Lenet Example')


parser.add_argument('--multi_data_url',
                    help='使用数据集训练时，需要定义的参数',
                    default= '[{}]')                        

parser.add_argument('--train_url',
                    help='回传结果到启智，需要定义的参数',
                    default= '')                          

parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'CPU'],
    help='device where the code will be implemented (default: Ascend),if to use the CPU on the Qizhi platform:device_target=CPU')

ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)

sigma2_UE = 1e-6



class Rate_loss(nn.Cell):
    def __init__(self, ):
        super(Rate_loss, self).__init__()
        self.sigma2_UE = 1e-6
        self.reduce_mean = ops.ReduceMean()

    def construct(self, f1, f2, H1, H2):
        # f1 = ops.Complex()(f1[..., 0], f1[..., 1])
        # f2 = ops.Complex()(f2[..., 0], f1[..., 1])
        # H1 = ops.Complex()(H1[..., 0], f1[..., 1])
        # H2 = ops.Complex()(H2[..., 0], f1[..., 1])
        power = ops.ReduceSum()(ops.Square()(f1), axis=(-2, -1))
        power = ops.ReduceSum()(power, axis=-1).reshape(-1, 1)
        power = power.repeat(SC_num, -1)
        power = power.reshape(-1, SC_num, 1, 1)
        f1 = 10.95445 * f1 / ops.sqrt(power)

        power = ops.ReduceSum()(ops.Square()(f2), axis=(-2, -1))
        power = ops.ReduceSum()(power, axis=-1)
        power = power.repeat(SC_num, -1)
        power = power.reshape(-1, SC_num, 1, 1)
        f2 = 10.95445 * f2 / ops.sqrt(power)
        H1_real = H1[..., 0].transpose([0, 3, 1, 2])
        H1_imag = H1[..., 1].transpose([0, 3, 1, 2])
        H2_real = H2[..., 0].transpose([0, 3, 1, 2])
        H2_imag = H2[..., 1].transpose([0, 3, 1, 2])
        f1_real = ops.ExpandDims()(f1[..., 0], -1)
        f1_imag = ops.ExpandDims()(f1[..., 1], -1)
        f2_real = ops.ExpandDims()(f2[..., 0], -1)
        f2_imag = ops.ExpandDims()(f2[..., 1], -1)
        R_real = ops.matmul(H1_real, f1_real) - ops.matmul(H1_imag, f1_imag) + \
                 ops.matmul(H2_real, f2_real) - ops.matmul(H2_imag, f2_imag)
        R_imag = ops.matmul(H1_real, f1_imag) + ops.matmul(H1_imag, f1_real) + \
                 ops.matmul(H2_real, f2_imag) + ops.matmul(H2_imag, f2_real)

        R = ops.Square()(R_real) + ops.Square()(R_imag)
        h_sub_gain = ops.ReduceSum()(R, axis=(-2, -1))
        SNR = h_sub_gain / sigma2_UE
        Rate = ops.log2(1 + SNR)
        Rate_OFDM_mean = ops.ReduceMean()(Rate)
        return 10 - Rate_OFDM_mean

loss_fn = Rate_loss()
# loss_fn = RateLoss()

if __name__ == '__main__':

    ###请在代码中加入args, unknown = parser.parse_known_args()，可忽略掉--ckpt_url参数报错等参数问题
    #  路径设置

    # 指定文件夹路径

    args, unknown = parser.parse_known_args()
    data_dir = '/cache/data'  # 数据集的路径
    train_dir = '/cache/output' # 输出模型的路径
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)      
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    print(args.multi_data_url)


    multi_data_json = json.loads(args.multi_data_url)
    try:
        mox.file.copy_parallel(multi_data_json[0]["dataset_url"], data_dir+'/data.zip') 
        print("Successfully Download {} to {}".format(multi_data_json[0]["dataset_url"],data_dir+'/data.zip'))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(
            multi_data_json[0]["dataset_url"], data_dir) + str(e))

    folder_path = '/cache'
    file_name = 'data.zip'  # 替换成你要检查的文件名

    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        print(f'文件 {file_name} 存在于文件夹 {folder_path} 中')
    else:
        print(f'文件 {file_name} 不存在于文件夹 {folder_path} 中')


    folder_path = '/cache'  # 替换成你要查看的文件夹路径

    # 使用os.walk遍历文件夹及其子文件夹中的文件结构
    for foldername, subfolders, filenames in os.walk(folder_path):
        print(f'当前文件夹: {foldername}')

        # 打印当前文件夹内的文件
        for filename in filenames:
            print(f'文件: {filename}')

        # 打印当前文件夹内的子文件夹
        for subfolder in subfolders:
            print(f'子文件夹: {subfolder}')



    zip_file=zipfile.ZipFile('/cache/data/data.zip')
    zip_list=zip_file.namelist()
    print(zip_list)
    for f in zip_list:
        zip_file.extract(f,'/cache/data')
    zip_file.close()

    print("拷贝成功")
    

    folder_path = '/cache'  # 替换成你要查看的文件夹路径

    # 使用os.walk遍历文件夹及其子文件夹中的文件结构
    for foldername, subfolders, filenames in os.walk(folder_path):
        print(f'当前文件夹: {foldername}')

        # 打印当前文件夹内的文件
        for filename in filenames:
            print(f'文件: {filename}')

        # 打印当前文件夹内的子文件夹
        for subfolder in subfolders:
            print(f'子文件夹: {subfolder}')


    total_epoch = 120
    LR = 1e-4
    batch_size = 400
    # BS=1
    BS = 2
    checkpoints_path = '/cache/output'
    print('checkpoints_path: ', checkpoints_path)

    load_org_data = True
    # Read Data
    if load_org_data is True:
        print('data_train_1.mat')
        data_1 = scio.loadmat(data_dir+'/data/data_train_1.mat')
        location_data_BS1 = data_1['loc_1']  # ndarray:(2400, 3)
        channel_data_BS1 = data_1['CSI_1']  # ndarray:(2400, 8, 32, 120, 2)

        # permutation_1 = np.random.permutation(location_data_BS1.shape[0])
        # location_data_BS1 = location_data_BS1[permutation_1]
        # channel_data_BS1 = channel_data_BS1[permutation_1]

        data_1 = {'1': location_data_BS1, '2': channel_data_BS1}

        print('data_train_2.mat')
        data_2 = scio.loadmat(data_dir+'/data/data_train_2.mat')
        location_data_BS2 = data_2['loc_2']
        channel_data_BS2 = data_2['CSI_2']

        # permutation_2 = np.random.permutation(location_data_BS2.shape[0])
        # location_data_BS2 = location_data_BS2[permutation_2]
        # channel_data_BS2 = channel_data_BS2[permutation_2]
        data_2 = {'1': location_data_BS2, '2': channel_data_BS2}

        print(len(location_data_BS1), len(location_data_BS2))

        data_combine_generate = [[], [], []]
        for i in range(len(location_data_BS1)):
            for j in range(len(location_data_BS2)):
        # for i in range(300):
        #     for j in range(300):
                if (location_data_BS2[j] == location_data_BS1[i]).all():
                    data_combine_generate[0].append(i)
                    data_combine_generate[1].append(i)
                    data_combine_generate[2].append(j)

        data_combine_dict = {}
        for i in range(len(data_combine_generate)):
            data_combine_dict[str(i)] = np.array(data_combine_generate[i])

    # scio.savemat('data_test_2.mat', data_combine_dict)
    # print('test_data saved')
    else:
        data = scio.loadmat('./data_test_0.mat')
        location_data_BS = data['0']
        channel_data_BS1 = data['1']
        channel_data_BS2 = data['2']
        data_combine_dict = {'1': location_data_BS, '2': channel_data_BS1, '3': channel_data_BS2}
    dataset = ds.NumpySlicesDataset(data=data_combine_dict, column_names=["loc", "CSI_1", "CSI_2"], shuffle=True)
    datasize = dataset.get_dataset_size()
    print('datasize: ', datasize)

    data_all_dict = {}
    data_all_dict["loc"] = location_data_BS1
    data_all_dict["CSI_1"] = channel_data_BS1
    data_all_dict["CSI_2"] = channel_data_BS2
    dataset_all_data = ds.NumpySlicesDataset(data=data_all_dict, column_names=["loc", "CSI_1", "CSI_2"], shuffle=False)
    dataset_all_data = dataset_all_data.batch(batch_size=dataset_all_data.get_dataset_size(), drop_remainder=True)


    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    step = dataset.get_dataset_size()
    print('step: ', step)

    # param_dict_1 = load_checkpoint(data_dir+'/data/modelBS1.ckpt')
    # param_dict_2 = load_checkpoint(data_dir+'/data/modelBS2.ckpt')

    net_1 = CombineNetwork()
    net_1 = amp.auto_mixed_precision(net_1, "O2")
    # load_param_into_net(net_1.net_1, param_dict_1)
    # load_param_into_net(net_1.net_2, param_dict_2)

    opt_1 = nn.Adam(params=net_1.trainable_params(), learning_rate=LR)
 
    def forward_fn(data, data_all):
        loc_data = data_all['loc']
        data_BS1 = data_all['CSI_1']
        data_BS2 = data_all['CSI_2']
        loc = loc_data[data['loc']]
        output1, output2 = net_1(loc)
        H_1 = data_BS1[data['CSI_1']]
        H_2 = data_BS2[data['CSI_2']]
        loss = loss_fn(output1, output2, H_1, H_2)
        # loss = loss_fn(output1, output2)
        return loss

    # 梯度方法
    grad_fn = ops.value_and_grad(forward_fn, None, net_1.trainable_params())

    def train_step(data_1, data_all):
        # 计算判别器损失和梯度
        loss_1, grads = grad_fn(data_1, data_all)
        opt_1(grads)
        return loss_1
    initial_lr = LR
    max_iter = total_epoch
    warmup_steps = 100

    os.makedirs(checkpoints_path, exist_ok=True)
    # net_1.set_train()
    loss_epoch = []
    losssave = 0
    for epoch in range(total_epoch):
        loss_all = []
        for data_each in dataset_all_data.create_dict_iterator():
            data_all = data_each
        for iter, data in enumerate(dataset.create_dict_iterator()):
            loss = train_step(data, data_all)
            loss_all.append(loss.asnumpy())
            if epoch > 100 and epoch<107: 
                LR = 1e-5
                ops.assign(opt_1.learning_rate, ms.Tensor(LR, ms.float32))
            if epoch > 107: 
                LR = 7e-7
                ops.assign(opt_1.learning_rate, ms.Tensor(LR, ms.float32))

        print("epoch {}".format(epoch + 1), "    mean loss: ", sum(loss_all)/len(loss_all))
        loss_epoch.append(sum(loss_all)/len(loss_all))
        losssave = sum(loss_all)/len(loss_all)
        # 根据epoch保存模型权重文件
        if epoch < 100:
            if (epoch + 1) % 5 == 0:
                save_checkpoint(net_1._cells['_backbone'].net_1, train_dir + f"/modelBS1_{int(epoch)}_{losssave}.ckpt")
                save_checkpoint(net_1._cells['_backbone'].net_2, train_dir + f"/modelBS2_{int(epoch)}_{losssave}.ckpt")
                print('model saved')
        if epoch > 100:
            if (epoch + 1) % 1 == 0:
                save_checkpoint(net_1._cells['_backbone'].net_1, train_dir + f"/modelBS1_{int(epoch)}_{losssave}.ckpt")
                save_checkpoint(net_1._cells['_backbone'].net_2, train_dir + f"/modelBS2_{int(epoch)}_{losssave}.ckpt")
                print('model saved')
    print('debug')

    plt.plot(loss_epoch)
    plt.show()
