import matplotlib as mpl
mpl.use('agg')

import os
import numpy as np
import pylab as plt

import cv2

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


file_list = ['../../dataset/map1n_allz_rtaapixlw_2048_'+str(i)+'.fits' for i in range(1,2)]
dp = cs.Data_Provider(file_list)

#dt = filt_all(dp(10,128),func)
#dt.shape
#fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,18))
#ax1.imshow(dt[0,:,:,0])
#ax2.imshow(dt[0,:,:,1])


batch_size = 512

image_size = 128
gf_dim = 8

z_dim = 4096
#(image_size/16)**2*gf_dim*8==z_dim

def dpp(n):
#    return dp(n,image_size).reshape(n,image_size,image_size,1)
    return filt_all(dp(n,image_size),func)

# run_config.gpu_options.allow_growth = True
dcgan = cs.DCGAN(
    data_provider = dpp,
    data_denormalizer = dp.denormalize,
    batch_size=batch_size,
    gf_dim=gf_dim, df_dim=64,
    gfc_dim=1024, dfc_dim=1024,
    z_dim=z_dim,
    checkpoint_dir='checkpoint')

dcgan.train(num_epoch=500,batch_per_epoch=50)

fig,ax=plt.subplots(1,5,figsize=(14,3))

sample_z = np.random.normal(size=(5, dcgan.z_dim))

for i in range(5):
	ax[i].imshow(dcgan.generate(sample_z)[i,:,:,0])
	ax[i].set_xticks([])
	ax[i].set_yticks([])

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig('plot_'+str(image_size)+'.jpg')




