import os

import math
import torch
import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

from utils import *
# from ModelDesign import *

# from ModelDesign_Split import *
# from ModelDesign_resnet import *
# from find_best_data_set import *
from ModelDesign_preliminary_round import *

save_path = os.getcwd() + '/model.pth'
validation_rate = 0.1  # 测试集比例
# 训练参数
epochs = 2000
warmup_epochs = 200
lr = 1e-4
min_lr = 1e-4
weight_decay = 0.01
batch_size = 512
# 训练设备
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# set all random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


set_seed(42)

# Parameters Cannot Change
SC_num = 128  # sub carrier number
Tx_num = 32  # Tx antenna number
Rx_num = 4  # Rx antenna number
sigma2_UE = 0.1

# Read Data
f = scio.loadmat('./train.mat')
data = f['train']
location_data = data[0, 0]['loc']
channel_data = data[0, 0]['CSI']

g_data = np.load('g_data.npy')
g_data = np.concatenate((g_data.real, g_data.imag), -1)
# location_data = location_data[:800, ...]  # pick part data
# channel_data = channel_data[:800, ...]
print('data:', location_data.shape, channel_data.shape)


# 划分训练集和测试集
groups = [(location_data[i:i+40:, ...], channel_data[i:i+40, ...], g_data[i:i+40, ...])
          for i in range(0, len(location_data), 40)]
# random.shuffle(groups)
shuffled_data_X, shuffled_data_Y, shuffled_data_G = zip(*groups)
val_len = int(validation_rate * len(shuffled_data_X))
shuffled_data_X_train = shuffled_data_X[:len(shuffled_data_X) - val_len]
shuffled_data_X_val = shuffled_data_X[len(shuffled_data_X) - val_len:]
shuffled_data_Y_train = shuffled_data_Y[:len(shuffled_data_X) - val_len]
shuffled_data_Y_val = shuffled_data_Y[len(shuffled_data_X) - val_len:]
shuffled_data_G_train = shuffled_data_G[:len(shuffled_data_X) - val_len]
shuffled_data_G_val = shuffled_data_G[len(shuffled_data_X) - val_len:]

data_X_train = np.array([item for group in shuffled_data_X_train for item in group])
data_X_val = np.array([item for group in shuffled_data_X_val for item in group])
data_Y_train = np.array([item for group in shuffled_data_Y_train for item in group])
data_Y_val = np.array([item for group in shuffled_data_Y_val for item in group])
data_G_train = np.array([item for group in shuffled_data_G_train for item in group])
data_G_val = np.array([item for group in shuffled_data_G_val for item in group])

# remake train data to mean
# data_Y_train = np.array([each_group.mean(0)[np.newaxis, ...].repeat(40, 0) for each_group in shuffled_data_Y_train])
# data_Y_train = data_Y_train.reshape((-1, 4, 32, 128, 2))


# 生成dataloader
features_tensor = torch.tensor(location_data, device=device)

features_tensor_train = torch.tensor(data_X_train, device=device)
label_tensor_train = torch.tensor(data_Y_train, device=device)
g_tensor_train = torch.tensor(data_G_train, device=device)

features_tensor_val = torch.tensor(data_X_val, device=device)
label_tensor_val = torch.tensor(data_Y_val, device=device)
g_tensor_val = torch.tensor(data_G_val, device=device)
# min-max
# min_vals, _ = torch.min(features_tensor_train, dim=0)
# max_vals, _ = torch.max(features_tensor_train, dim=0)
# features_tensor_train = (features_tensor_train - min_vals) / (max_vals - min_vals + 1e-5)
# features_tensor_val = (features_tensor_val - min_vals) / (max_vals - min_vals + 1e-5)

label_train_g = DownPrecoding(label_tensor_train.numpy())
label_val_g = DownPrecoding(label_tensor_val.numpy())

train_dataset = Data.TensorDataset(features_tensor_train, label_tensor_train, g_tensor_train)
val_dataset = Data.TensorDataset(features_tensor_val, label_tensor_val, g_tensor_val)
train_dataloader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f'训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}')


# Compute score up limit
def compute_up_limit(data_Y):
    gt_h = DownPrecoding(data_Y)
    print('h:', gt_h.shape)
    data_rate = DataRate(EqChannelGain(data_Y, gt_h), sigma2_UE)
    print('score limit (best):', data_rate)
    indices = np.random.randint(0, 40, gt_h.shape[0]) + np.arange(gt_h.shape[0]) // 40 * 40
    # assert np.all(location_data == location_data[indices])
    data_rate = DataRate(EqChannelGain(data_Y, gt_h[indices]), sigma2_UE)
    print('score limit (random):', data_rate)
    return data_rate


def compute_up_limit_mean(data_Y):
    data_Y_mean = np.array([each_group.mean(0)[np.newaxis, ...].repeat(40, 0) for each_group in shuffled_data_Y_val]).reshape((-1, 4, 32, 128, 2))
    gt_h = DownPrecoding(data_Y_mean)
    print('h:', gt_h.shape)
    data_rate = DataRate(EqChannelGain(data_Y, gt_h), sigma2_UE)
    print('score limit (mean):', data_rate)
    return data_rate


data_rate_random = compute_up_limit(data_Y_val)
data_rate_mean = compute_up_limit_mean(data_Y_val)


# Model Set
network = RadioMap_Model_TypeI_pre_train(input_dim=3, version=VERSION).to(device)
if VERSION == 1:
    loss_fn = nn.MSELoss()
else:
    loss_fn = DataRateLoss(sigma2_UE=sigma2_UE)
    loss_fn_eval = nn.MSELoss()
    # loss_fn = nn.MSELoss()
    print('use mode two')
print('Network:', network)
# self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg.lr)
optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        # 预热阶段，学习率逐渐增加
        return (epoch + 1) / warmup_epochs
    else:
        # 余弦退火学习率衰减
        rr = epoch / len(train_dataloader)/epochs
        rr = (1 - rr) ** 1.5
        rr = (rr*(lr-min_lr) + min_lr)/lr

        return rr


lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# 训练
def train_epoch():
    network.train()
    total_loss = 0
    total_size = 0
    for batch, (X, y, g) in enumerate(train_dataloader, 1):
        # Compute prediction error
        pred_g, mid = network(X, g, stage_1=True, stage_2=False, train=True)
        loss = loss_fn(pred_g.to(torch.float64), y.to(torch.float64))
        X = X.unsqueeze(-1).repeat(1, 1, SC_num).permute((0, 2, 1))
        # loss_eval = loss_fn_eval(mid.to(torch.float64), X.to(torch.float64))/120
        # loss += loss_eval
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X)
        total_size += len(X)
    lr_scheduler.step()
    # print(f"Train Avg loss: {total_loss / total_size:>7f}, lr={lr_scheduler.get_last_lr()[0]:.4e}")
    return total_loss / total_size, lr_scheduler.get_last_lr()[0]


# 验证
@torch.no_grad()
def validation():
    num_batches = len(val_dataloader)
    network.eval()
    test_loss = 0
    test_loss_eval = 0
    predictions = []
    gt_data = []
    for X, y, g in val_dataloader:

        pred_g, _ = network(X, g, stage_1=True, stage_2=False, train=True)

        test_loss += loss_fn(pred_g.to(torch.float64), y.to(torch.float64)).item()
        pred_g = network(pred_g, g, stage_1=False, stage_2=True)
        predictions.append(pred_g)
        gt_data.append(y)
    test_loss /= num_batches

    PrecodingVector = torch.cat(predictions, dim=0).cpu().numpy()
    PrecodingVector = np.reshape(PrecodingVector, (-1, SC_num, Tx_num, 1))
    SubCH_gain_codeword = EqChannelGain(torch.cat(gt_data).cpu().numpy(), PrecodingVector)
    data_rate = DataRate(SubCH_gain_codeword, sigma2_UE)
    # print(f"Val   Avg loss: {test_loss:>8f}, score: {data_rate:.3f} bps/Hz \n")
    return test_loss, data_rate


# 训练过程
train_losses = []
val_losses = []
data_rates = []
learning_rate = []
best_val_loss = 999
for i in range(epochs):
    train_loss, lr = train_epoch()
    val_loss, val_score = validation()
    train_losses.append(train_loss)
    learning_rate.append(lr)
    val_losses.append(val_loss)
    data_rates.append(val_score)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(network.state_dict(), save_path)
        print(f"At Epoch: '{i}' Saved PyTorch Model State to '{save_path}'")
    # show train progress
    print('Epoch: ', i, 'train_loss: ', np.float32(train_loss), 'val_loss: ',
          np.float32(val_loss), 'data_rates: ', np.float32(val_score))

# Model Paint
plt.figure(figsize=(20, 6))
plt.subplot(131)
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='val loss')
plt.legend(loc='best')
plt.title('Losses')
plt.xlim(0, epochs)

plt.subplot(132)
plt.plot(data_rates)
plt.axhline(data_rate_mean, color='b', linestyle='--')
plt.axhline(data_rate_random, color='r', linestyle='--')
plt.xlim(0, epochs)
plt.title('Score')

plt.subplot(133)
plt.plot(learning_rate)
plt.xlim(0, epochs)
plt.yscale('log')
plt.title('learning_rate')
plt.show()

#  Prediction
network.load_state_dict(torch.load(save_path, map_location='cpu'))
network.eval()
with torch.no_grad():
    pred = network(features_tensor)
PrecodingVector = pred.cpu().numpy()
PrecodingVector = np.reshape(PrecodingVector, (-1, SC_num, Tx_num, 1))

# Calculate the score
SubCH_gain_codeword = EqChannelGain(channel_data, PrecodingVector)
data_rate = DataRate(SubCH_gain_codeword, sigma2_UE)
print('Attention: not valid. The score of RadioMap_TypeI is %f bps/Hz' % data_rate)
