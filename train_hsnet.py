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
device_data = torch.device("cpu")
device_train = torch.device("cuda:0")
device_test = torch.device("cpu")

Material = 'Meta'
TrainingDataSize = 1000000
TestingDataSize = 100000
BatchSize = 2000
BatchEnable = True
EpochNum = 501
TestInterval = 10
lr = [1e-4]
lr_decay_step = 50
lr_decay_gamma = 0.8
beta_Constrain = 1e-3
beta_Smooth = 7e-6
TFNum = 4
param_min = torch.tensor([200, 100, 50, 300])
param_max = torch.tensor([400, 200, 200, 400])


StartWL = 400
EndWL = 701
Resolution = 2
WL = np.arange(StartWL, EndWL, Resolution)
SpectralSliceNum = WL.size

path_data = 'data/'
specs_train = torch.zeros([TrainingDataSize, SpectralSliceNum], device=device_data, dtype=dtype)
specs_test = torch.zeros([TestingDataSize, SpectralSliceNum], device=device_test, dtype=dtype)
data = h5py.File(path_data + 'ICVL/SpectralCurves/ICVLSpecs_PchipInterp.mat', 'r')
Specs_all = np.array(data['Specs'])
np.random.shuffle(Specs_all)
specs_train[0:TrainingDataSize//2, :] = torch.tensor(Specs_all[0:TrainingDataSize//2, :])
specs_test[0:TestingDataSize//2, :] = torch.tensor(
    Specs_all[TrainingDataSize//2:TrainingDataSize//2 + TestingDataSize//2, :])
data = h5py.File(path_data + 'CAVE/SpectralCurves/ColumbiaSpecs_PchipInterp.mat', 'r')
Specs_all = np.array(data['Specs'])
np.random.shuffle(Specs_all)
specs_train[TrainingDataSize//2:TrainingDataSize, :] = torch.tensor(Specs_all[0:TrainingDataSize//2, :])
specs_test[TestingDataSize//2:TestingDataSize, :] = torch.tensor(
    Specs_all[TrainingDataSize//2:TrainingDataSize//2 + TestingDataSize//2, :])

data.close()
del Specs_all, data
assert SpectralSliceNum == WL.size

MatchLossFcn = nn.MSELoss(reduction='mean')


class hsnetLoss(nn.Module):
    def __init__(self):
        super(hsnetLoss, self).__init__()

    def forward(self, t1, t2, params, beta_Range, beta_Smooth):
        MatchLoss = MatchLossFcn(t1, t2)

        # Spectral response range regularization (limited between 0-1).
        # U-shaped function，U([param_min + delta, param_max - delta]) = 0, U(param_min) = U(param_max) = 1。
        delta = 0.01
        res = torch.max((params - delta) / (-delta), (params + delta - 1) / delta)
        RangeLoss = torch.mean(torch.max(res, torch.zeros_like(res)))

        # KL-Loss function, f(rho)=0, f(0, 1)=Inf.
        # Because the KL-Loss is not defined at the entire real number domain, it often lets the gradient vanish.
        # Thus the KL-Loss is not suitable here.
        # rho = 0.5
        # RangeLoss = sum(sum(rho * torch.log(rho / params) + (1-rho) * torch.log((1-rho)/(1-params))))

        # L2-norm-based smoothness regularization.
        shift_diff = params - params.roll(1)
        shift_diff[:, 0] = 0
        SmoothLoss = torch.norm(shift_diff)

        return MatchLoss + beta_Range * RangeLoss + beta_Smooth * SmoothLoss


LossFcn = hsnetLoss()

for k in range(len(lr)):
    folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = 'nets/hsnet/' + folder_name + '/'
    rnet_path = 'nets/rnet/Meta/rnet.pkl'
    fnet_path = 'nets/fnet/Meta/fnet.pkl'

    hsnet = nn.Sequential()
    hsnet.add_module('HardwareLayer', nn.Linear(SpectralSliceNum, TFNum))
    hsnet.add_module('LReLU1', nn.LeakyReLU())
    hsnet.add_module('Linear2', nn.Linear(TFNum, 500))
    hsnet.add_module('LReLU2', nn.LeakyReLU())
    hsnet.add_module('Linear3', nn.Linear(500, 500))
    hsnet.add_module('LReLU3', nn.LeakyReLU())
    hsnet.add_module('Linear4', nn.Linear(500, SpectralSliceNum))
    hsnet = hsnet.to(device_train)

    hsnetParams = hsnet.named_parameters()
    for name, params in hsnetParams:
        if name == 'HardwareLayer.bias':
            params.requires_grad = False
            nn.init.constant_(params, 0)
        if name == 'HardwareLayer.weight':
            nn.init.uniform_(params, a=0.1, b=0.9)


    optimizer = torch.optim.Adam(hsnet.parameters(), lr=lr[k])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    loss = torch.tensor([0], device=device_train)
    loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
    loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))

    os.makedirs(path, exist_ok=True)
    log_file = open(path + 'TrainingLog.txt', 'w+')
    time_start = time.time()
    time_epoch0 = time_start
    for epoch in range(EpochNum):
        specs_train = specs_train[torch.randperm(TrainingDataSize), :]
        for i in range(0, TrainingDataSize // BatchSize):
            Specs_batch = specs_train[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
            Output_pred = hsnet(Specs_batch)
            hsnetParams = hsnet.named_parameters()
            HWWeights = torch.tensor([])
            for name, params in hsnetParams:
                if name == 'HardwareLayer.weight':
                    HWWeights = params
                    break
            assert HWWeights.size() == torch.Size([TFNum, SpectralSliceNum])
            loss = LossFcn(Specs_batch, Output_pred, HWWeights, beta_Constrain, beta_Smooth)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
        if epoch % TestInterval == 0:
            hsnet.to(device_test)
            out_test_pred = hsnet(specs_test)
            hsnet.to(device_train)
            loss_train[epoch // TestInterval] = loss.data
            loss_t = MatchLossFcn(specs_test, out_test_pred)
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

    hsnet.eval()
    torch.save(hsnet, path + 'hsnet.pkl')
    hsnet.to(device_test)

    hsnetParams = hsnet.named_parameters()
    HWWeights = torch.tensor([])
    for name, params in hsnetParams:
        if name == 'HardwareLayer.weight':
            HWWeights = params
            break
    assert HWWeights.size() == torch.Size([TFNum, SpectralSliceNum])
    TargetCurves = HWWeights.detach().cpu().numpy()
    scio.savemat(path + 'TargetCurves.mat', mdict={'TargetCurves': TargetCurves})

    rnet = torch.load(rnet_path)
    rnet.to(device_test)
    fnet = torch.load(fnet_path)
    fnet.to(device_test)
    DesignParams = (param_max - param_min) * rnet(HWWeights) + param_min
    print(DesignParams[0, :])
    TargetCurves_FMN = fnet(DesignParams).detach().cpu().numpy()
    scio.savemat(path + 'TrainedCurves_check.mat', mdict={'TargetCurves_FMN': TargetCurves_FMN})
    Params = DesignParams.detach().cpu().numpy()
    scio.savemat(path + 'TrainedParams.mat', mdict={'Params': Params})

    plt.figure()
    for i in range(TFNum):
        plt.subplot(math.ceil(math.sqrt(TFNum)), math.ceil(math.sqrt(TFNum)), i + 1)
        plt.plot(WL, TargetCurves[i, :], WL, TargetCurves_FMN[i, :])
        plt.ylim(0, 1)
    plt.savefig(path + 'ROFcurves')
    plt.show()

    Output_temp = hsnet(specs_train[0, :].to(device_test).unsqueeze(0)).squeeze(0)
    FigureTrainLoss = MatchLossFcn(specs_train[0, :].to(device_test), Output_temp)
    plt.figure()
    plt.plot(WL, specs_train[0, :].cpu().numpy())
    plt.plot(WL, Output_temp.detach().cpu().numpy())
    plt.ylim(0, 1)
    plt.legend(['GT', 'pred'], loc='upper right')
    plt.savefig(path + 'train')
    plt.show()

    Output_temp = hsnet(specs_test[0, :].to(device_test).unsqueeze(0)).squeeze(0)
    FigureTestLoss = MatchLossFcn(specs_test[0, :].to(device_test), Output_temp)
    plt.figure()
    plt.plot(WL, specs_test[0, :].cpu().numpy())
    plt.plot(WL, Output_temp.detach().cpu().numpy())
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

    plt.figure()
    plt.plot(range(0, EpochNum, TestInterval), loss_train.detach().cpu().numpy())
    plt.plot(range(0, EpochNum, TestInterval), loss_test.detach().cpu().numpy())
    plt.semilogy()
    plt.legend(['Loss_train', 'Loss_test'], loc='upper right')
    plt.savefig(path + 'loss')
    plt.show()
