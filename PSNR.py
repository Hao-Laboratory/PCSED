import math

def mse2psnr(mse, bitdepth=8):
    return 10 * math.log10((2**bitdepth - 1)**2 / mse)
def rmse2psnr(rmse, bitdepth=8):
    return 20 * math.log10((2**bitdepth - 1) / rmse)