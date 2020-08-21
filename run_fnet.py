import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import time

dtype = torch.float
device = torch.device("cpu")

TrainingDataRatio = 0.8
DataSize = 9 ** 4
TrainingDataSize = int(DataSize * TrainingDataRatio)
TestingDataSize = DataSize - TrainingDataSize

StartWL = 400
EndWL = 701
Resolution = 2
WL = np.arange(StartWL, EndWL, Resolution)
data = h5py.File('data/Metasurfaces/data_bricks.mat', 'r')
StartWL = 400
EndWL = 701
Resolution = 2
OutputNum = WL.size
Input_data = torch.tensor(data['params'][:, 0:DataSize], device=device, dtype=dtype).T * 1e9
Output_data = torch.tensor(data['T'][:, 0:DataSize], device=device, dtype=dtype).T
idx = torch.randperm(DataSize)
Input_data = Input_data[idx, :]
Output_data = Output_data[idx, :]
Input_train = Input_data[0:TrainingDataSize, :]
Output_train = Output_data[0:TrainingDataSize, :]
Input_test = Input_data[TrainingDataSize:TrainingDataSize + TestingDataSize, :]
Output_test = Output_data[TrainingDataSize:TrainingDataSize + TestingDataSize, :]
InputNum = Input_train.shape[1]
assert WL.size == Output_train.shape[1]

del data, Input_data, Output_data
print(Input_train.shape)
print(Output_train.shape)
print(Input_test.shape)
print(Output_test.shape)
print(Input_train[0, :])

net = torch.load('nets/fnet/Meta/fnet.pkl')
net = net.to(device)
net.eval()

LossFcn = nn.MSELoss(reduction='mean')

time_start = time.time()
Output_temp = net(Input_train)
time_train = time.time() - time_start
FinalTrainLoss = LossFcn(Output_train, Output_temp)
plt.plot(WL.T, Output_train[0, :].cpu().numpy())
plt.plot(WL.T, Output_temp[0, :].detach().cpu().numpy())
plt.legend(['GT', 'pred'], loc='lower right')
plt.show()
time_start = time.time()
Output_temp = net(Input_test)
time_test = time.time() - time_start
FinalTestLoss = LossFcn(Output_test, Output_temp)
plt.plot(WL.T, Output_test[15, :].cpu().numpy())
plt.plot(WL.T, Output_temp[15, :].detach().cpu().numpy())
plt.legend(['GT', 'pred'], loc='lower right')
plt.show()
print('Running finished!', '| train loss: %.5f' % FinalTrainLoss.data.item(),
      '| test loss: %.5f' % FinalTestLoss.data.item())
print('Running time on training set: %.1fs' % time_train)
print('Average running time of training sample: %.8fs' % (time_train / TrainingDataSize))
print('Running time on testing set: %.1fs' % time_test)
print('Average running time of testing sample: %.8fs' % (time_test / TestingDataSize))
