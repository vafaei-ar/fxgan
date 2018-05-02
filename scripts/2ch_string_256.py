import os
import random
import numpy as np
import pylab as plt
import cv2

import sys
import csgan as cs

def canny(d_in,R,meth,edd):
    d = d_in
    if (R!=0):
        dt = np.fft.fft2(d)
        if meth=='g':
            for i in range(sz):
                for j in range(sz):
                    k2 = 1.*(i*i+j*j)/d.shape[0]
                    dt[i,j]=dt[i,j]*np.exp(-k2*R*R/2)

        if meth=='tp':
            for i in range(sz):
                for j in range(sz):
                    k = np.sqrt(0.001+i*i+j*j)/sz
                    dt[i,j]=dt[i,j]* 3*(np.sin(k*R)-k*R*np.cos(k*R))/(k*R)**3

        d = np.fft.ifft2(dt)
        d = abs(d)

    if edd=='lap':
        d = cv2.Laplacian(d,cv2.CV_64F)

    if edd=='sob':
        sobelx = cv2.Sobel(d,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(d,cv2.CV_64F,0,1,ksize=3)
        d =np.sqrt(sobelx**2+sobely**2)

    if edd=='sch':
        scharrx = cv2.Scharr(d,cv2.CV_64F,1,0)
        scharry = cv2.Scharr(d,cv2.CV_64F,0,1)
        d =np.sqrt(scharrx**2+scharry**2)
        
    return d

def filt_all(maps,func):
    out1 = []
    for m in maps:
        out1.append(func(m))
        
    return np.stack([maps,np.array(out1)],axis=3)

def func(dt):
    return canny(dt,0,'none','sch')


file_list = ['../../dataset/map1n_allz_rtaapixlw_2048_'+str(i)+'.fits' for i in range(1,4)]
dp = cs.Data_Provider(file_list,preprocess_mode=2)

#dt = filt_all(dp(10,128),func)
#dt.shape
#fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,18))
#ax1.imshow(dt[0,:,:,0])
#ax2.imshow(dt[0,:,:,1])


batch_size = 64
image_size = 256
checkpoint_dir = './checkpoint/'+sys.argv[0][:-3]
sample_dir = './samples/'+sys.argv[0][:-3]

def dpp(n):
#    return dp(n,image_size).reshape(n,image_size,image_size,1)
    return filt_all(dp(n,image_size),func)

# defult_model_build lets you to define your own generator and discriminator. 
# Set it to 1, if you want to use default DCGAN architecture.
defult_model_build=1

dcgan = cs.DCGAN(
    data_provider = dpp,
    data_postprocess = dp.postprocess,
    batch_size=64, gf_dim=64, df_dim=64,
    label_real_lower=.9, label_fake_upper=.1,
    z_dim=2048,checkpoint_dir=checkpoint_dir,
    save_per = 100, defult_model_build=defult_model_build)

dcgan.train(num_epoch=100000,batch_per_epoch=50,verbose=10,\
learning_rate=1e-4,D_update_per_batch=1,G_update_per_batch=1,\
sample_dir=sample_dir,checkpoint_dir=checkpoint_dir,time_limit=600)
