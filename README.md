# Introduction
This is the implementation of the "parameter constrained spectral encoder and decoder (PCSED)" introduced in the article "Deep-Learned Random Optical Filters for Spectroscopic Instruments".
# Instructions
## Variables:
The names of the variables are not consistant with those concepts in the paper. Here are the matchups (the variable name in the left corresponds to the concept in paper in the right):  
fnet -- FMN  
hsnet -- SED  
hybnet -- PCSED  
rnet -- IDN (inverse design network)  
## Folders:
data -- dataset folder. We did not upload the dataset because it exceeds the Github repository storage limit. You can download the demonstration dataset from https://xxx and copy it to this folder, then the scripts should work correctly.  
nets -- the folder for storaging the networks and the generated data.
## Files:
HybridNet.py -- PCSED definition.  
PSNR.py -- MSE to PSNR transformation function definitions.  
run_fnet.py -- run the trained FMN.  
run_hsnet.py -- run the trained SED.  
run_hybnet.py -- run the trained PCSED.  
show_HSI_error.py -- plot the hyperspectral image (HSI) reconstruction results of PCSED and SED.  
train_fnet.py -- train an FMN.  
train_hsnet.py -- train a SED.  
train_hybnet_Meta.py -- train a PCSED.  
train_tnet.py -- train an IDN using a tandem neural network architecture (proposed in Ref. [18] in the main text).  
redesign_hsnet.py -- re-design the ROFs of the SED. This script is for designing the ROFs using a different material (metasurface or thin-film) to match the same target spectral responses.
