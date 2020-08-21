import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import math

dtype = torch.float
device = torch.device("cpu")

Material = 'Meta'  # 'Meta' or 'TF'
# Material == 'TF'
TrainingDataSize = 1000000
TestingDataSize = 100000
TFNum = 4
if Material == 'TF':
    thick_min = 100
    thick_max = 300
else:
    thick_min = torch.tensor([200, 100, 50, 300])
    thick_max = torch.tensor([400, 200, 200, 400])

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

MatchLossFcn = nn.MSELoss(reduction='mean')

folder_name = 'Meta'  # If the Material is 'TF', change the folder_name to 'TF_100-300nm' or 'TF_0-150nm'
# folder_name = 'TF_100-300nm'
# folder_name = 'TF_0-150nm'
path = 'nets/hsnet/' + folder_name + '/'
rnet_path = 'nets/rnet/' + folder_name + '/rnet.pkl'
fnet_path = 'nets/fnet/' + folder_name + '/fnet.pkl'

hsnet = torch.load(path + 'hsnet.pkl')
hsnet.eval()
hsnet.to(device)

hsnetParams = hsnet.named_parameters()
HWWeights = torch.tensor([])
for name, params in hsnetParams:
    if name == 'HardwareLayer.weight':
        HWWeights = params
        break
assert HWWeights.size() == torch.Size([TFNum, SpectralSliceNum])
TargetCurves = HWWeights.detach().cpu().numpy()

rnet = torch.load(rnet_path)
rnet.to(device)
fnet = torch.load(fnet_path)
fnet.to(device)
DesignParams = (thick_max - thick_min) * rnet(HWWeights) + thick_min
print(DesignParams)
TargetCurves_FMN = fnet(DesignParams).detach().cpu().numpy()
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

Output_temp = hsnet(specs_train[0, :].to(device).unsqueeze(0)).squeeze(0)
FigureTrainLoss = MatchLossFcn(specs_train[0, :].to(device), Output_temp)
plt.figure(2)
plt.plot(WL, specs_train[0, :].cpu().numpy())
plt.plot(WL, Output_temp.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='upper right')
plt.savefig(path + 'train')
plt.show()

Output_temp = hsnet(specs_test[0, :].to(device).unsqueeze(0)).squeeze(0)
FigureTestLoss = MatchLossFcn(specs_test[0, :].to(device), Output_temp)
plt.figure(3)
plt.plot(WL, specs_test[0, :].cpu().numpy())
plt.plot(WL, Output_temp.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='upper right')
plt.savefig(path + 'test')
plt.show()
