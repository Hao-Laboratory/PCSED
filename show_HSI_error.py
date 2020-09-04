from matplotlib import cm, colors
from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import scipy.interpolate as scinterp
import h5py
import torch
import torch.nn as nn
import os
import PSNR

dtype = torch.float
device = torch.device("cpu")

path_data = 'data/'

Material = 'Meta'
folder_name_hybnet = 'Meta'
folder_name_hsnet = 'Meta'
# Material = 'TF'
# folder_name_hybnet = 'TF'
# folder_name_hsnet = 'TF_100-300nm'
# folder_name_hsnet = 'TF_0-150nm'
path_hybnet = 'nets/hybnet/' + folder_name_hybnet + '/'
path_hsnet = 'nets/hsnet/' + folder_name_hsnet + '/'

cmp_list = ['HSI_error_target', 'HSI_error_designed', 'HSI_error_noised']

for cmp in cmp_list:
    if cmp == 'HSI_error_target':
        path_HSI_error = path_hybnet + 'HSI_error_target/'
        os.makedirs(path_HSI_error, exist_ok=True)
        HWWeights_hybnet = torch.tensor(scio.loadmat(path_hybnet + 'TargetCurves.mat')['TFCurves'], device=device,
                                        dtype=dtype)
        HWWeights_hsnet = torch.tensor(scio.loadmat(path_hsnet + 'TargetCurves.mat')['TFCurves'], device=device,
                                       dtype=dtype)
    elif cmp == 'HSI_error_designed':
        path_HSI_error = path_hybnet + 'HSI_error_designed/'
        os.makedirs(path_HSI_error, exist_ok=True)
        HWWeights_hybnet = torch.tensor(scio.loadmat(path_hybnet + 'DesignedCurves.mat')['TFs'], device=device,
                                        dtype=dtype)
        HWWeights_hsnet = torch.tensor(scio.loadmat(path_hsnet + 'DesignedCurves.mat')['TFs'], device=device,
                                       dtype=dtype)
    elif cmp == 'HSI_error_noised':
        path_HSI_error = path_hybnet + 'HSI_error_noised/'
        os.makedirs(path_HSI_error, exist_ok=True)
        HWWeights_hybnet = torch.tensor(scio.loadmat(path_hybnet + 'FabedCurves_noised_6nm.mat')['TFs_noised'],
                                        device=device, dtype=dtype)
        HWWeights_hsnet = torch.tensor(scio.loadmat(path_hsnet + 'FabedCurves_noised_6nm.mat')['TFs_noised'],
                                       device=device, dtype=dtype)

    HSI_file_name = [['CAVE', 'fake_and_real_lemon_slices_ms'],
                     ['CAVE', 'feathers_ms'],
                     ['CAVE', 'flowers_ms'],
                     ['CAVE', 'oil_painting_ms'],
                     ['ICVL', 'ARAD_HS_0016'],
                     ['ICVL', 'ARAD_HS_0156'],
                     ['ICVL', 'ARAD_HS_0389']]
    curve1_pos_x = [136, 122, 324, 59, 450, 181, 420]
    curve1_pos_y = [125, 339, 211, 122, 30, 166, 229]
    curve2_pos_x = [351, 370, 186, 226, 145, 241, 238]
    curve2_pos_y = [158, 269, 342, 270, 249, 300, 424]
    ErrorFcn = nn.MSELoss(reduction='none')

    WL_orign = np.arange(400, 701, 10)
    WL = np.arange(400, 701, 2)

    Curve1_GT = torch.zeros([len(WL), len(HSI_file_name)])
    Curve1_hybnetd = torch.zeros([len(WL), len(HSI_file_name)])
    Curve1_hsnetd = torch.zeros([len(WL), len(HSI_file_name)])
    Curve2_GT = torch.zeros([len(WL), len(HSI_file_name)])
    Curve2_hybnetd = torch.zeros([len(WL), len(HSI_file_name)])
    Curve2_hsnetd = torch.zeros([len(WL), len(HSI_file_name)])
    HSI = []
    MSE_hybnetd = []
    MSE_hsnetd = []
    RGB = {}


    # Set the first image as the master, with all the others
    # observing it for changes in cmap or norm.

    class ImageFollower(object):
        'update image in response to changes in clim or cmap on another image'

        def __init__(self, follower):
            self.follower = follower

        def __call__(self, leader):
            self.follower.set_cmap(leader.get_cmap())
            self.follower.set_clim(leader.get_clim())


    for k in range(len(HSI_file_name)):
        if HSI_file_name[k][0] == 'CAVE':
            HSI = torch.tensor(np.array(h5py.File(path_data + 'CAVE/HSI/' + HSI_file_name[k][1] + '_interp.mat', 'r')
                                        ['HSI_interp']), device=device, dtype=dtype)
            HSI = HSI.transpose(0, 2)
            HSI /= 65535
            RGB[HSI_file_name[k][1]] = imread(path_data + 'CAVE/HSI/' + HSI_file_name[k][1] + '.png')
            MSE_hybnetd.append(torch.zeros_like(HSI))
            MSE_hsnetd.append(torch.zeros_like(HSI))
        elif HSI_file_name[k][0] == 'ICVL':
            if os.path.exists(path_data + 'ICVL/HSI/' + HSI_file_name[k][1] + '_interp.mat'):
                HSI = torch.tensor(scio.loadmat(path_data + 'ICVL/HSI/' + HSI_file_name[k][1] + '_interp.mat')
                                   ['cube'], device=device, dtype=dtype)
                HSI = HSI[:, 15:(512-15), :]  # For a better image alignment, cut the ICVL images into a square (482*482)
            else:
                HSI_orign = torch.tensor(np.array(scio.loadmat(path_data + 'ICVL/HSI/' + HSI_file_name[k][1] + '.mat')
                                                  ['cube']), device=device, dtype=dtype)
                HSI_orign = HSI_orign.reshape([482 * 512, HSI_orign.size(2)])
                # HSI = nn.functional.interpolate(HSI_orign, size=[HSI_orign.size(0), HSI_orign.size(0), len(WL)])
                HSI = torch.zeros(HSI_orign.size(0), len(WL))
                for i in range(HSI_orign.size(0)):
                    HSI[i, :] = torch.tensor(scinterp.pchip_interpolate(WL_orign, HSI_orign[i, :].numpy(), WL))
                HSI = HSI.reshape([482, 512, len(WL)])
                scio.savemat(path_data + 'ICVL/HSI/' + HSI_file_name[k][1] + '_interp.mat', {'cube': HSI.numpy()})
                HSI = HSI[:, 15:(512 - 15), :]  # For a better image alignment, cut the ICVL images into a square (482*482)
            RGB[HSI_file_name[k][1]] = imread(path_data + 'ICVL/HSI/' + HSI_file_name[k][1] + '_clean.png')[:, 15:(512-15), :]
            MSE_hybnetd.append(torch.zeros_like(HSI))
            MSE_hsnetd.append(torch.zeros_like(HSI))

        Curve1_GT[:, k] = HSI[curve1_pos_y[k], curve1_pos_x[k], :]
        Curve2_GT[:, k] = HSI[curve2_pos_y[k], curve2_pos_x[k], :]

        net = torch.load(path_hybnet + 'hybnet.pkl', map_location=device)
        net.to(device)
        net.eval()
        HSI_hybnetd = net.run_swnet(HSI.to(device), HWWeights_hybnet).detach()
        Curve1_hybnetd[:, k] = HSI_hybnetd[curve1_pos_y[k], curve1_pos_x[k], :]
        Curve2_hybnetd[:, k] = HSI_hybnetd[curve2_pos_y[k], curve2_pos_x[k], :]

        net = torch.load(path_hsnet + 'hsnet.pkl', map_location=device)
        net.to(device)
        net.eval()
        net.HardwareLayer.weight.data = HWWeights_hsnet
        HSI_hsnetd = net(HSI.to(device)).detach()
        Curve1_hsnetd[:, k] = HSI_hsnetd[curve1_pos_y[k], curve1_pos_x[k], :]
        Curve2_hsnetd[:, k] = HSI_hsnetd[curve2_pos_y[k], curve2_pos_x[k], :]

        MSE_hybnetd[k] = ErrorFcn(HSI_hybnetd, HSI).mean(2)
        MSE_hsnetd[k] = ErrorFcn(HSI_hsnetd, HSI).mean(2)
        MSE_pc_hybnetd = ErrorFcn(HSI_hybnetd, HSI)
        MSE_pc_hsnetd = ErrorFcn(HSI_hsnetd, HSI)

        WL_pick = np.arange(400, 701, 50)
        index = []
        for i in WL_pick:
            index.append(np.argwhere(WL == i)[0, 0])
        datadict = {
            0: HSI[:, :, index].detach().cpu().numpy(),
            1: HSI_hybnetd[:, :, index].detach().cpu().numpy(),
            2: HSI_hsnetd[:, :, index].detach().cpu().numpy(),
            3: MSE_pc_hybnetd[:, :, index].detach().cpu().numpy(),
            4: MSE_pc_hsnetd[:, :, index].detach().cpu().numpy(),
        }

        Nr = 5
        Nc = WL_pick.size

        fig = plt.figure() # Per Channel Error

        figtitle = 'Per Channel Error'
        fig.text(0.5, 0.95, figtitle,
                 horizontalalignment='center',
                 fontproperties=FontProperties(size=16))
        ylabel = ['           GT',
                  '         PCSED',
                  '       SED-inv',
                  '     MSE_PCSED',
                  '    MSE_SED-inv']

        cax = fig.add_axes([0.2, 0.08, 0.6, 0.04])

        w = 0.105
        h = 0.14
        ax = []
        images = []
        errors = []
        ivmin = 1e40
        evmin = 1e40
        ivmax = -1e40
        evmax = -1e40
        for i in range(Nr):
            for j in range(Nc):
                pos = [0.12 + j * 1.05 * w, 0.75 - i * 1.05 * h, w, h]
                a = fig.add_axes(pos)
                a.axis('off')
                data = np.abs(datadict[i][:, :, j])
                dd = np.ravel(data)
                if i > 2:
                    # Manually find the min and max of all colors for
                    # use in setting the color scale.
                    evmin = min(evmin, np.amin(dd))
                    evmax = max(evmax, np.amax(dd))
                    errors.append(a.imshow(data, cmap=cm.viridis))
                else:
                    # Manually find the min and max of all colors for
                    # use in setting the color scale.
                    ivmin = min(ivmin, np.amin(dd))
                    ivmax = max(ivmax, np.amax(dd))
                    images.append(a.imshow(data, cmap=cm.gray))
                if i == 0:
                    fig.text(0.14 + j * 1.05 * w, 0.9, str(WL_pick[j]) + 'nm',
                             fontproperties=FontProperties(size=10))
                if j == 0:
                    fig.text(0.095, 0.75 - i * 1.05 * h, ylabel[i],
                             fontproperties=FontProperties(size=6),
                             rotation=90)

                ax.append(a)

        norm = colors.Normalize(vmin=ivmin, vmax=ivmax)
        for i, im in enumerate(images):
            im.set_norm(norm)
            if i > 0:
                images[0].callbacksSM.connect('changed', ImageFollower(im))

        norm = colors.Normalize(vmin=evmin, vmax=evmax)
        for i, im in enumerate(errors):
            im.set_norm(norm)
            if i > 0:
                errors[0].callbacksSM.connect('changed', ImageFollower(im))

        # The colorbar is also based on this master image.
        fig.colorbar(errors[0], cax, orientation='horizontal')

        # We need the following only if we want to run this interactively and
        # modify the colormap:

        plt.axes(ax[0])  # Return the current axes to the first one,
        plt.sci(images[0])  # because the current image must be in current axes.
        plt.axes(ax[21])  # Return the current axes to the first one,
        plt.sci(errors[0])  # because the current image must be in current axes.

        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        plt.savefig(path_HSI_error + 'PerChannelError_' + HSI_file_name[k][1])
        plt.show()

    Nr = 5
    Nc = len(HSI_file_name)

    fig = plt.figure() # HSI Reconstruction Error

    if cmp == 'HSI_error_designed':
        figLabels = ['(a)', '(b)', '(c)', '(d)', '(e)']
        for i in range(5):
            fig.text(0.095, 0.885 - i * 0.147, figLabels[i],
                    horizontalalignment='center',
                    fontproperties=FontProperties(size=10))

    ylabel = ['       RGB\n     images',
              '       MSE\n      PCSED',
              '       MSE\n     SED-inv',
              '   red patch',
              'green patch']

    cax = fig.add_axes([0.2, 0.1, 0.6, 0.02])

    w = 0.105
    h = 0.14
    ax = []
    images = []
    ivmin = 1e40
    ivmax = -1e40
    for i in range(Nr):
        for j in range(Nc):
            pos = [0.12 + j * 1.05 * w, 0.77 - i * 1.05 * h, w, h]
            a = fig.add_axes(pos)
            if i == 0:
                a.axis('off')
                a.imshow(RGB[HSI_file_name[j][1]])
                if j <= 4:
                    a.add_patch(plt.Rectangle((curve1_pos_x[j] - 15, curve1_pos_y[j] - 15), width=31, height=31, linewidth=1,
                                              edgecolor='r', facecolor='none'))
                    a.add_patch(plt.Rectangle((curve2_pos_x[j] - 15, curve2_pos_y[j] - 15), width=31, height=31, linewidth=1,
                                              edgecolor='lime', facecolor='none'))
                else:  # Coordinates of cutted ICVL images has 15 pixels right shift
                    a.add_patch(plt.Rectangle((curve1_pos_x[j] - 15 - 15, curve1_pos_y[j] - 15), width=31, height=31, linewidth=1,
                                              edgecolor='r', facecolor='none'))
                    a.add_patch(plt.Rectangle((curve2_pos_x[j] - 15 - 15, curve2_pos_y[j] - 15), width=31, height=31, linewidth=1,
                                              edgecolor='lime', facecolor='none'))
            if i == 1:
                a.axis('off')
                data = MSE_hybnetd[j]
                dd = np.ravel(data)
                ivmin = min(ivmin, np.amin(dd))
                ivmax = max(ivmax, np.amax(dd))
                images.append(a.imshow(data, cmap=cm.viridis))
                if j <= 4:
                    a.add_patch(plt.Rectangle((curve1_pos_x[j] - 15, curve1_pos_y[j] - 15), width=31, height=31, linewidth=1,
                                              edgecolor='r', facecolor='none'))
                    a.add_patch(plt.Rectangle((curve2_pos_x[j] - 15, curve2_pos_y[j] - 15), width=31, height=31, linewidth=1,
                                              edgecolor='lime', facecolor='none'))
                else:  # Coordinates of cutted ICVL images has 15 pixels right shift
                    a.add_patch(plt.Rectangle((curve1_pos_x[j] - 15 - 15, curve1_pos_y[j] - 15), width=31, height=31, linewidth=1,
                                              edgecolor='r', facecolor='none'))
                    a.add_patch(plt.Rectangle((curve2_pos_x[j] - 15 - 15, curve2_pos_y[j] - 15), width=31, height=31, linewidth=1,
                                              edgecolor='lime', facecolor='none'))
                a.text(40, 70, 'PSNR=' + str('%.2f' % PSNR.mse2psnr(np.mean(dd), bitdepth=1)) + 'dB',
                       fontproperties=FontProperties(size=5), color='w')
            if i == 2:
                a.axis('off')
                data = MSE_hsnetd[j]
                dd = np.ravel(data)
                ivmin = min(ivmin, np.amin(dd))
                ivmax = max(ivmax, np.amax(dd))
                images.append(a.imshow(data, cmap=cm.viridis))
                if j <= 4:
                    a.add_patch(plt.Rectangle((curve1_pos_x[j] - 15, curve1_pos_y[j] - 15), width=31, height=31, linewidth=1,
                                              edgecolor='r', facecolor='none'))
                    a.add_patch(plt.Rectangle((curve2_pos_x[j] - 15, curve2_pos_y[j] - 15), width=31, height=31, linewidth=1,
                                              edgecolor='lime', facecolor='none'))
                else:  # Coordinates of cutted ICVL images has 15 pixels right shift
                    a.add_patch(plt.Rectangle((curve1_pos_x[j] - 15 - 15, curve1_pos_y[j] - 15), width=31, height=31, linewidth=1,
                                              edgecolor='r', facecolor='none'))
                    a.add_patch(plt.Rectangle((curve2_pos_x[j] - 15 - 15, curve2_pos_y[j] - 15), width=31, height=31, linewidth=1,
                                              edgecolor='lime', facecolor='none'))
                a.text(40, 70, 'PSNR=' + str('%.2f' % PSNR.mse2psnr(np.mean(dd), bitdepth=1)) + 'dB',
                       fontproperties=FontProperties(size=5), color='w')
            if i == 3:
                a.plot(WL, Curve1_GT[:, j], 'r', linewidth=1)
                a.plot(WL, Curve1_hybnetd[:, j], '--', linewidth=0.8)
                a.plot(WL, Curve1_hsnetd[:, j], '--', linewidth=0.8)
                a.set_ylim([0, 1])
                a.set_yticklabels(['0', None, '.5'])
                a.tick_params(labelsize=6)
                if j == 0:
                    a.legend(['ground truth', 'PCSED', 'SED-inv'], loc='upper left', frameon=False, fontsize=4.5)
            if i == 4:
                a.plot(WL, Curve2_GT[:, j], c='lime', linewidth=1)
                a.plot(WL, Curve2_hybnetd[:, j], '--', linewidth=0.8)
                a.plot(WL, Curve2_hsnetd[:, j], '--', linewidth=0.8)
                a.set_ylim([0, 1])
                a.set_yticklabels(['0', None, '.5'])
                a.tick_params(labelsize=6)
                if j == 0:
                    a.legend(['ground truth', 'PCSED', 'SED-inv'], loc='upper left', frameon=False, fontsize=4.5)
            if j == 0:
                fig.text(0.075, 0.77 - i * 1.05 * h, ylabel[i],
                         fontproperties=FontProperties(size=6),
                         rotation=90)
            if i < 4:
                a.set_xticks([])
                a.set_xticklabels([])
            if j > 0:
                a.set_yticks([])
                a.set_yticklabels([])

            ax.append(a)

    norm = colors.Normalize(vmin=ivmin, vmax=ivmax)
    for i, im in enumerate(images):
        im.set_norm(norm)
        if i > 0:
            images[0].callbacksSM.connect('changed', ImageFollower(im))

    # The colorbar is also based on this master image.
    cb = fig.colorbar(images[0], cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=7)

    # We need the following only if we want to run this interactively and
    # modify the colormap:

    plt.axes(ax[7])  # Return the current axes to the first one,
    plt.sci(images[0])  # because the current image must be in current axes.

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(path_HSI_error + 'HSI Reconstruction Error')
    plt.show()
