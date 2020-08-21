import HybridNet
import torch
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import math
import time
import os

dtype = torch.float
device_data = torch.device("cpu")
device_train = torch.device("cuda:0")
device_test = torch.device("cpu")

TrainingDataSize = 1000000
TestingDataSize = 100000
BatchSize = 2000
EpochNum = 501
TestInterval = 10
lr = 1e-3
lr_decay_step = 100
lr_decay_gamma = 0.8
beta_range = 1e-3
TFNum = 4
thick_min = 100
thick_max = 300

StartWL = 400
EndWL = 701
Resolution = 2
WL = np.arange(StartWL, EndWL, Resolution)
SpectralSliceNum = WL.size

path_data = 'data/'
Specs_train = torch.zeros([TrainingDataSize, SpectralSliceNum], device=device_data, dtype=dtype)
Specs_test = torch.zeros([TestingDataSize, SpectralSliceNum], device=device_test, dtype=dtype)
data = h5py.File(path_data + 'ICVL/SpectralCurves/ICVLSpecs_PchipInterp.mat', 'r')
Specs_all = np.array(data['Specs'])
np.random.shuffle(Specs_all)
Specs_train[0:TrainingDataSize//2, :] = torch.tensor(Specs_all[0:TrainingDataSize//2, :])
Specs_test[0:TestingDataSize//2, :] = torch.tensor(
    Specs_all[TrainingDataSize//2:TrainingDataSize//2 + TestingDataSize//2, :])
data = h5py.File(path_data + 'CAVE/SpectralCurves/ColumbiaSpecs_PchipInterp.mat', 'r')
Specs_all = np.array(data['Specs'])
np.random.shuffle(Specs_all)
Specs_train[TrainingDataSize//2:TrainingDataSize, :] = torch.tensor(Specs_all[0:TrainingDataSize//2, :])
Specs_test[TestingDataSize//2:TestingDataSize, :] = torch.tensor(
    Specs_all[TrainingDataSize//2:TrainingDataSize//2 + TestingDataSize//2, :])

data.close()
del Specs_all, data
assert SpectralSliceNum == Specs_train.size(1)

folder_name = time.strftime("%Y%m%d_%H%M%S",time.localtime())
path = 'nets/hybnet/' + folder_name + '/'
fnet_path = 'nets/fnet/TF_100-300nm/fnet.pkl'

hybnet_size = [SpectralSliceNum, TFNum, SpectralSliceNum]
hybnet = HybridNet.HybridNet(fnet_path, thick_min, thick_max, hybnet_size, device_train)

LossFcn = HybridNet.HybnetLoss()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, hybnet.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

loss = torch.tensor([0], device=device_train)
loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))

os.makedirs(path, exist_ok=True)
log_file = open(path + 'TrainingLog.txt', 'w+')
time_start = time.time()
time_epoch0 = time_start
for epoch in range(EpochNum):
    Specs_train = Specs_train[torch.randperm(TrainingDataSize), :]
    for i in range(0, TrainingDataSize // BatchSize):
        Specs_batch = Specs_train[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
        Output_pred = hybnet(Specs_batch)
        DesignParams = hybnet.show_design_params()
        loss = LossFcn(Specs_batch, Output_pred, DesignParams, thick_min, thick_max, beta_range)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    scheduler.step()
    if epoch % TestInterval == 0:
        hybnet.to(device_test)
        hybnet.eval()
        Out_test_pred = hybnet(Specs_test)
        hybnet.to(device_train)
        hybnet.train()
        hybnet.eval_fnet()
        loss_train[epoch // TestInterval] = loss.data
        loss_t = HybridNet.MatchLossFcn(Specs_test, Out_test_pred)
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

hybnet.eval()
torch.save(hybnet, path + 'hybnet.pkl')
hybnet.to(device_test)

HWweights = hybnet.show_hw_weights()
TargetCurves = HWweights.detach().cpu().numpy()
scio.savemat(path + 'TargetCurves.mat', mdict={'TargetCurves': TargetCurves})

DesignParams = hybnet.show_design_params()
print(DesignParams[0, :])
TargetCurves_FMN = hybnet.run_fnet(DesignParams).detach().cpu().numpy()
scio.savemat(path + 'TargetCurves_FMN.mat', mdict={'TargetCurves_FMN': TargetCurves_FMN})
Params = DesignParams.detach().cpu().numpy()
scio.savemat(path + 'TrainedParams.mat', mdict={'Params': Params})

plt.figure(1)
plt.clf()
for i in range(TFNum):
    plt.subplot(math.ceil(math.sqrt(TFNum)), math.ceil(math.sqrt(TFNum)), i + 1)
    plt.plot(WL, TargetCurves[i, :], WL, TargetCurves_FMN[i, :])
    plt.ylim(0, 1)
plt.savefig(path + 'ROFcurves')
plt.show()

Output_train = hybnet(Specs_train[0, :].to(device_test).unsqueeze(0)).squeeze(0)
FigureTrainLoss = HybridNet.MatchLossFcn(Specs_train[0, :].to(device_test), Output_train)
plt.figure(2)
plt.plot(WL, Specs_train[0, :].cpu().numpy())
plt.plot(WL, Output_train.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='upper right')
plt.savefig(path + 'train')
plt.show()

Output_test = hybnet(Specs_test[0, :].to(device_test).unsqueeze(0)).squeeze(0)
FigureTestLoss = HybridNet.MatchLossFcn(Specs_test[0, :].to(device_test), Output_test)
plt.figure(3)
plt.plot(WL, Specs_test[0, :].cpu().numpy())
plt.plot(WL, Output_test.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='upper right')
plt.savefig(path + 'test')
plt.show()

print('Training finished!',
      '| loss in figure \'train.png\': %.5f' % FigureTrainLoss.data.item(),
      '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item())
print('Training finished!',
      '| loss in figure \'train.png\': %.5f' % FigureTrainLoss.data.item(),
      '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item(), file=log_file)
log_file.close()

plt.figure(4)
plt.plot(range(0, EpochNum, TestInterval), loss_train.detach().cpu().numpy())
plt.plot(range(0, EpochNum, TestInterval), loss_test.detach().cpu().numpy())
plt.semilogy()
plt.legend(['Loss_train', 'Loss_test'], loc='upper right')
plt.savefig(path + 'loss')
plt.show()
