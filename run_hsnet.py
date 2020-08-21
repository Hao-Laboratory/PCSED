import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np

dtype = torch.float
device = torch.device("cpu")

TrainingDataSize = 1000000
TestingDataSize = 100000

StartWL = 400
EndWL = 701
Resolution = 2
WL = np.arange(StartWL, EndWL, Resolution)
SpectralSliceNum = WL.size

path_data = 'data/'
specs_train = torch.zeros([TrainingDataSize, SpectralSliceNum], device=device, dtype=dtype)
specs_test = torch.zeros([TestingDataSize, SpectralSliceNum], device=device, dtype=dtype)
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

cmp = ['designed', 'noised']
path = 'nets/hsnet/Meta/'
hsnet = torch.load(path + 'hsnet.pkl')
hsnet.to(device)
hsnet.eval()

HWWeights_designed = torch.tensor(scio.loadmat(path + 'DesignedCurves.mat')['TFs'], device=device, dtype=dtype)
HWWeights_noised = torch.tensor(scio.loadmat(path + 'FabedCurves_noised_6nm.mat')['TFs_noised'], device=device, dtype=dtype)
TFNum = HWWeights_designed.size(0)

MatchLossFcn = nn.MSELoss(reduction='mean')

output_train = hsnet(specs_train.to(device))
loss_train = MatchLossFcn(specs_train.to(device), output_train)
output_test = hsnet(specs_test.to(device))
loss_test = MatchLossFcn(specs_test.to(device), output_test)

hsnet.HardwareLayer.weight.data = HWWeights_designed

output_train_designed = hsnet(specs_train.to(device))
loss_train_designed = MatchLossFcn(specs_train.to(device), output_train_designed)
output_test_designed = hsnet(specs_test.to(device))
loss_test_designed = MatchLossFcn(specs_test.to(device), output_test_designed)

hsnet.HardwareLayer.weight.data = HWWeights_noised

output_train_noised = hsnet(specs_train.to(device))
loss_train_noised = MatchLossFcn(specs_train.to(device), output_train_noised)
output_test_noised = hsnet(specs_test.to(device))
loss_test_noised = MatchLossFcn(specs_test.to(device), output_test_noised)

log_file = open(path + 'RunningLog.txt', 'w+')
print('Running finished!')
print('| train loss using trained curves: %.5f' % loss_train.data.item(),
      '| train loss using designed curves: %.5f' % loss_train_designed.data.item(),
      '| train loss using noised curves: %.5f' % loss_train_noised.data.item())
print('| test loss using trained curves: %.5f' % loss_test.data.item(),
      '| test loss using designed curves: %.5f' % loss_test_designed.data.item(),
      '| test loss using noised curves: %.5f' % loss_test_noised.data.item())
print('Running finished!', file=log_file)
print('| train loss using trained curves: %.5f' % loss_train.data.item(),
      '| train loss using designed curves: %.5f' % loss_train_designed.data.item(),
      '| train loss using noised curves: %.5f' % loss_train_noised.data.item(), file=log_file)
print('| test loss using trained curves: %.5f' % loss_test.data.item(),
      '| test loss using designed curves: %.5f' % loss_test_designed.data.item(),
      '| test loss using noised curves: %.5f' % loss_test_noised.data.item(), file=log_file)
log_file.close()