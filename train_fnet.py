import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import time
import math
import os

dtype = torch.float
device_data = torch.device('cpu')
device_train = torch.device('cuda:0')
device_test = torch.device('cpu')

Material = 'Meta'

if Material == 'TF':
    TrainingDataSize = 500000
    TestingDataSize = 50000
    IsParallel = False
    EpochNum = 2001
    TestInterval = 20
    BatchSize = 1000
    lr = 1e-3
    if IsParallel:
        BatchSize = BatchSize * torch.cuda.device_count()
        lr = lr * torch.cuda.device_count()
    lr_decay_step = 200
    lr_decay_gamma = 0.8

    folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = 'nets/fnet/' + folder_name + '/'

    data = scio.loadmat('data/ThinFilms/data_TF_100-300nm.mat')
    InputNum = int(data['LayersNum'])
    StartWL = 400
    EndWL = 701
    Resolution = 2
    WL = np.arange(StartWL, EndWL, Resolution)
    OutputNum = WL.size
    assert WL.size == np.array(data['WL']).size - 50
    Input_train = torch.tensor(data['Thick_train'][:, 0:TrainingDataSize], device=device_data, dtype=dtype).T
    Output_train = torch.tensor(data['Trans_train'][10:161, 0:TrainingDataSize], device=device_data, dtype=dtype).T
    Input_test = torch.tensor(data['Thick_test'][:, 0:TestingDataSize], device=device_test, dtype=dtype).T
    Output_test = torch.tensor(data['Trans_test'][10:161, 0:TestingDataSize], device=device_test, dtype=dtype).T

    del data

else:
    TrainingDataRatio = 0.8
    DataSize = 9 ** 4
    TrainingDataSize = int(DataSize * TrainingDataRatio)
    TestingDataSize = DataSize - TrainingDataSize
    IsParallel = False
    EpochNum = 2001
    TestInterval = 20
    BatchSize = 2000
    lr = 1e-3
    if IsParallel:
        BatchSize = BatchSize * torch.cuda.device_count()
        lr = lr * torch.cuda.device_count()
    lr_decay_step = 200
    lr_decay_gamma = 0.8

    folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = 'nets/fnet/' + folder_name + '/'

    data = h5py.File('data/Metasurfaces/data_bricks.mat', 'r')
    StartWL = 400
    EndWL = 701
    Resolution = 2
    WL = np.arange(StartWL, EndWL, Resolution)
    OutputNum = WL.size
    Input_data = torch.tensor(data['params'][:, 0:DataSize], device=device_data, dtype=dtype).T * 1e9
    Output_data = torch.tensor(data['T'][:, 0:DataSize], device=device_data, dtype=dtype).T
    idx = torch.randperm(DataSize)
    Input_data = Input_data[idx, :]
    Output_data = Output_data[idx, :]
    Input_train = Input_data[0:TrainingDataSize, :]
    Output_train = Output_data[0:TrainingDataSize, :]
    Input_test = Input_data[TrainingDataSize:TrainingDataSize+TestingDataSize, :]
    Output_test = Output_data[TrainingDataSize:TrainingDataSize+TestingDataSize, :]
    InputNum = Input_train.shape[1]
    assert WL.size == Output_train.shape[1]

    del data, Input_data, Output_data

fnet = nn.Sequential(
    nn.Linear(InputNum, 200),
    nn.BatchNorm1d(200),
    nn.LeakyReLU(inplace=True),
    nn.Linear(200, 800),
    nn.BatchNorm1d(800),
    nn.LeakyReLU(inplace=True),
    nn.Linear(800, 800),
    nn.Dropout(0.1),
    nn.BatchNorm1d(800),
    nn.LeakyReLU(inplace=True),
    nn.Linear(800, 800),
    nn.Dropout(0.1),
    nn.BatchNorm1d(800),
    nn.LeakyReLU(inplace=True),
    nn.Linear(800, 800),
    nn.Dropout(0.1),
    nn.BatchNorm1d(800),
    nn.LeakyReLU(inplace=True),
    nn.Linear(800, OutputNum),
    nn.Dropout(0.1),
    nn.Sigmoid()
)
if IsParallel:
    fnet = nn.DataParallel(fnet)
fnet.to(device_train)
fnet.train()

LossFcn = nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(fnet.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

loss = torch.tensor([0], device=device_train)
loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))

os.makedirs(path, exist_ok=True)
log_file = open(path + 'TrainingLog.txt', 'w+')
time_start = time.time()
time_epoch0 = time_start
for epoch in range(EpochNum):
    idx = torch.randperm(TrainingDataSize, device=device_data)
    Input_train = Input_train[idx, :]
    Output_train = Output_train[idx, :]
    for i in range(0, TrainingDataSize // BatchSize):
        InputBatch = Input_train[i * BatchSize: i * BatchSize + BatchSize, :]
        OutputBatch = Output_train[i * BatchSize: i * BatchSize + BatchSize, :]
        Output_pred = fnet(InputBatch.to(device_train))
        loss = LossFcn(OutputBatch.to(device_train), Output_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    if epoch % TestInterval == 0:
        fnet.to(device_test)
        fnet.eval()
        Out_test_pred = fnet(Input_test)
        fnet.to(device_train)
        fnet.train()
        loss_train[epoch // TestInterval] = loss.data
        loss_t = LossFcn(Output_test, Out_test_pred)
        loss_test[epoch // TestInterval] = loss_t.data
        if epoch == 0:
            time_epoch0 = time.time()
            time_remain = (time_epoch0 - time_start) * EpochNum
        else:
            time_remain = (time.time() - time_epoch0) / epoch * (EpochNum - epoch)
        print('Epoch:', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler.get_lr()[0], '| remaining time: %.0fs (to %s)'
              % (time_remain, time.strftime('%H:%M:%S', time.localtime(time.time() + time_remain))))
        print('Epoch:', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler.get_lr()[0], file=log_file)
time_end = time.time()
time_total = time_end - time_start
m, s = divmod(time_total, 60)
h, m = divmod(m, 60)
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s))
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s), file=log_file)

fnet.eval()
torch.save(fnet, path + 'fnet.pkl')

fnet.to(device_test)
Output_temp = fnet(Input_train[0, :].to(device_test).unsqueeze(0)).squeeze(0)
FigureTrainLoss = LossFcn(Output_train[0, :].to(device_test), Output_temp)
plt.figure()
plt.plot(WL.T, Output_train[0, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='lower right')
plt.savefig(path + 'train')
plt.show()

Output_temp = fnet(Input_test[0, :].to(device_test).unsqueeze(0)).squeeze(0)
FigureTestLoss = LossFcn(Output_test[0, :].to(device_test), Output_temp)
plt.figure()
plt.plot(WL.T, Output_test[0, :].cpu().numpy())
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
