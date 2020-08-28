# Introduction
This is the implementation of the "parameter constrained spectral encoder and decoder (PCSED)" introduced in the article "Deep-Learned Random Optical Filters for Spectroscopic Instruments".
# Instructions
## Variables:
The names of the variables are not consistant with the concepts in the paper. Here are the matchups (the variable name in the left corresponds to the concept in the right):  
fnet -- FMN  
hsnet -- SED  
hybnet -- PCSED  
rnet -- IDN (inverse design network)  
## Folders:
data -- 
nets -- 
## Files:
HybridNet.py -- PCSED definition.  
PSNR.py -- MSE and PSNR transformation function definitions.  
redesign_hsnet.py -- 
run_fnet.py -- 
run_hsnet.py -- 
run_hybnet.py -- 
show_HSI_error.py -- 
train_fnet.py -- 
train_hsnet.py -- 
train_hybnet_Meta.py -- 
train_hybnet_TF.py -- 
train_tnet.py -- 
