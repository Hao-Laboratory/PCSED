import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import math
import time
import os

dtype = torch.float
device_data = torch.device('cpu')
device_train = torch.device('cuda:0')
device_test = torch.device('cpu')

Material = 'Meta'
# Material = 'TF'
TrainingDataRatio = 0.8
DataSize = 9 ** 4
TrainingDataSize = int(DataSize * TrainingDataRatio)
TestingDataSize = DataSize - TrainingDataSize
BatchSize = 2000
BatchEnable = True
EpochNum = 1001
TestInterval = 20
lr = 1e-3
lr_decay_step = 100
lr_decay_gamma = 0.8
if Material == 'Meta':
    params_min = torch.tensor([200, 100, 50, 300])
    params_max = torch.tensor([400, 200, 200, 400])
else:
    params_min = torch.tensor([100])
    params_max = torch.tensor([300])

folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
path = 'nets/rnet/' + folder_name + '/'

if Material == 'Meta':
    fnet_path = 'nets/fnet/Meta/fnet.pkl'
else:
    fnet_path = 'nets/fnet/TF_100-300nm/fnet.pkl'
    # fnet_path = 'nets/fnet/TF_0-150nm/fnet.pkl'

if Material == 'Meta':
    data = h5py.File('data/Metasurfaces/data_bricks.mat', 'r')
    StartWL = 400
    EndWL = 701
    Resolution = 2
    WL = np.arange(StartWL, EndWL, Resolution)
    InputNum = WL.size
    Params_data = torch.tensor(data['params'][:, 0:DataSize], device=device_data, dtype=dtype).T * 1e9
    Trans_data = torch.tensor(data['T'][:, 0:DataSize], device=device_data, dtype=dtype).T
    idx = torch.randperm(DataSize)
    Params_data = Params_data[idx, :]
    Trans_data = Trans_data[idx, :]
    Params_train = Params_data[0:TrainingDataSize, :]
    Trans_train = Trans_data[0:TrainingDataSize, :]
    Params_test = Params_data[TrainingDataSize:TrainingDataSize+TestingDataSize, :]
    Trans_test = Trans_data[TrainingDataSize:TrainingDataSize+TestingDataSize, :]
    assert InputNum == Trans_train.shape[1]
else:
    data = scio.loadmat('data/ThinFilms/data_TF_100-300nm.mat')
    # data = scio.loadmat('data/ThinFilms/data_TF_0-150nm.mat')
    StartWL = 400
    EndWL = 701
    Resolution = 2
    WL = np.arange(StartWL, EndWL, Resolution)
    assert WL.size == np.array(data['WL']).size - 50
    InputNum = len(WL)
    Trans_train = torch.tensor(data['Trans_train'][10:161, 0:TrainingDataSize], device=device_data, dtype=dtype).T
    Trans_test = torch.tensor(data['Trans_test'][10:161, 0:TestingDataSize], device=device_test, dtype=dtype).T

del data

fnet = torch.load(fnet_path)
OutputNum = fnet.state_dict()['0.weight'].data.size(1)
fnet.to(device_train)
fnet.eval()

rnet = nn.Sequential(
    nn.Linear(InputNum, 2000),
    nn.BatchNorm1d(2000),
    nn.LeakyReLU(),
    nn.Linear(2000, 2000),
    nn.BatchNorm1d(2000),
    nn.LeakyReLU(),
    nn.Linear(2000, 800),
    nn.BatchNorm1d(800),
    nn.LeakyReLU(),
    nn.Linear(800, 800),
    nn.BatchNorm1d(800),
    nn.LeakyReLU(),
    nn.Linear(800, 100),
    nn.BatchNorm1d(100),
    nn.LeakyReLU(),
    nn.Linear(100, OutputNum),
    nn.Sigmoid(),
)
rnet.to(device_train)

LossFcn = nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(rnet.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

loss = torch.tensor([0], device=device_train)
loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))

os.makedirs(path, exist_ok=True)
log_file = open(path + 'TrainingLog.txt', 'w+')
time_start = time.time()
time_epoch0 = time_start
for epoch in range(EpochNum):
    Trans_train_shuffled = Trans_train[torch.randperm(TrainingDataSize), :]
    for i in range(0, TrainingDataSize // BatchSize):
        InputBatch = Trans_train_shuffled[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
        Output_pred = fnet((params_max - params_min).to(device_train) * rnet(InputBatch) + params_min.to(device_train))
        loss = LossFcn(InputBatch, Output_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    if epoch % TestInterval == 0:
        fnet.to(device_test)
        rnet.to(device_test)
        rnet.eval()
        Out_test_pred = fnet((params_max - params_min).to(device_test) * rnet(Trans_test) + params_min.to(device_test))
        fnet.to(device_train)
        rnet.to(device_train)
        rnet.train()
        loss_train[epoch // TestInterval] = loss.data
        loss_t = LossFcn(Trans_test, Out_test_pred)
        loss_test[epoch // TestInterval] = loss_t.data
        if epoch == 0:
            time_epoch0 = time.time()
            time_remain = (time_epoch0 - time_start) * EpochNum
        else:
            time_remain = (time.time() - time_epoch0) / epoch * (EpochNum - epoch)
        print('Epoch: ', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler.get_lr()[0], '| remaining time: %.0fs (to %s)'
              % (time_remain, time.strftime('%H:%M:%S', time.localtime(time.time() + time_remain))))
        print('Epoch: ', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler.get_lr()[0], file=log_file)
time_end = time.time()
time_total = time_end - time_start
m, s = divmod(time_total, 60)
h, m = divmod(m, 60)
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s))
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s), file=log_file)

rnet.eval()
torch.save(rnet, path + 'rnet.pkl')
fnet.to(device_test)
rnet.to(device_test)

Params_temp = (params_max - params_min).to(device_test) * rnet(Trans_train[0, :].to(device_test).unsqueeze(0)).squeeze(0) + params_min.to(device_test)
Output_temp = fnet(Params_temp.unsqueeze(0)).squeeze(0)
FigureTrainLoss = LossFcn(Trans_train[0, :].to(device_test), Output_temp)

print('Structure parameters of curve in figure \'train.png\':')
print('Structure parameters of curve in figure \'train.png\':', file=log_file)
print(Params_train[0, :])
print(Params_train[0, :], file=log_file)
print('Designed parameters of curve in figure \'train.png\':')
print('Designed parameters of curve in figure \'train.png\':', file=log_file)
print(Params_temp)
print(Params_temp, file=log_file)
plt.figure()
plt.plot(WL.T, Trans_train[0, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='lower right')
plt.savefig(path + 'train')
plt.show()

Params_temp = (params_max - params_min).to(device_test) * rnet(Trans_test[0, :].to(device_test).unsqueeze(0)).squeeze(0) + params_min.to(device_test)
Output_temp = fnet(Params_temp.unsqueeze(0)).squeeze(0)
FigureTestLoss = LossFcn(Trans_test[0, :].to(device_test), Output_temp)

print('Structure parameters of curve in figure \'test.png\':')
print('Structure parameters of curve in figure \'test.png\':', file=log_file)
print(Params_test[0, :])
print(Params_test[0, :], file=log_file)
print('Designed parameters of curve in figure \'test.png\':')
print('Designed parameters of curve in figure \'test.png\':', file=log_file)
print(Params_temp)
print(Params_temp, file=log_file)
plt.figure()
plt.plot(WL.T, Trans_test[0, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='lower right')
plt.savefig(path + 'test')
plt.show()

print('Training finished!',
      '| loss in figure \'train.png\': %.5f' % FigureTrainLoss.data.item(),
      '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item())
print('Training finished!',
      '| loss in figure \'train.png\': %.5f' % FigureTrainLoss.data.item(),
      '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item(), file=log_file)
log_file.close()

plt.figure()
plt.plot(range(0, EpochNum, TestInterval), loss_train.detach().cpu().numpy())
plt.plot(range(0, EpochNum, TestInterval), loss_test.detach().cpu().numpy())
plt.semilogy()
plt.legend(['Loss_train', 'Loss_test'], loc='upper right')
plt.savefig(path + 'loss')
plt.show()
