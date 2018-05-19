import os
import sys
import cv2
import random
import numpy as np
import csgan as cs 
import pylab as plt
from time import time
from skimage.draw import polygon
from scipy.ndimage.interpolation import rotate

def canny(d,edege):

    if edege=='lap':
	    d = cv2.Laplacian(d,cv2.CV_64F)

    if edege=='sob':
	    sobelx = cv2.Sobel(d,cv2.CV_64F,1,0,ksize=3)
	    sobely = cv2.Sobel(d,cv2.CV_64F,0,1,ksize=3)
	    d =np.sqrt(sobelx**2+sobely**2)

    if edege=='sch':
	    scharrx = cv2.Scharr(d,cv2.CV_64F,1,0)
	    scharry = cv2.Scharr(d,cv2.CV_64F,0,1)
	    d =np.sqrt(scharrx**2+scharry**2)

    return d

class Simple_String(object):

    def __init__(self,nx=200,ny=200,num=100,augment=True,
                reinit=0,ns=20,l_min=None,l_max=None):
        self.nx = nx
        self.ny = ny
        if augment:
            num = 8*num
        self.num = num
        self.augment = augment
        self.reinit = reinit
        self.ns = ns
        if l_min is None:
            self.l_min = min(nx,ny)//10
        else:
            self.l_min = l_min
        if l_max is None:
            self.l_max = min(nx,ny)//3
        else:
            self.l_max = l_max
        if reinit:
            self.season = 0
        
        self.strings = np.zeros((num,nx,ny,1))
        self.reinitiate()
        
    def augmentation(self,m,i):
        i_rot = i%4
        i_flp = i//4
        if i_rot:
            m = np.rot90(m,i_rot)
        if i_flp:
            m = np.fliplr(m)
        return m
    
    def reinitiate(self):
        if self.augment:
            for i in range(0,self.num,8):
                m = self.string()
                for j in range(8):
                    self.strings[i+j,:,:,0] = self.augmentation(m,j)    
        else:
            for i in range(self.num):
                self.strings[i,:,:,0] = self.string()
        
    def string(self,ns=None):
        
        if ns is None:
            ns = self.ns
    
        imx = self.nx+100
        imy = self.ny+100
        
        rec = np.zeros((imx,imy))
        for i in range(ns):
            width = 2*np.random.randint(self.l_min,self.l_max)
            height = width/2
            rand = np.random.randint((imx-width)*(imy-height))
            
            r0 = rand%(imx-width)
            c0 = rand//(imx-width)
            angle = np.random.uniform(0,180)
        
            rr, cc = [r0, r0 + width//2, r0 + width//2., r0], [c0, c0, c0 + height, c0 + height]
            rr, cc = polygon(rr, cc)    
            rec[rr, cc] += 1
            
            rr, cc = [r0 + width//2, r0 + width, r0 + width, r0 + width//2], [c0, c0, c0 + height, c0 + height]
            rr, cc = polygon(rr, cc)    
            rec[rr, cc] = -1
        
            rec = rotate(rec, angle, axes=(1, 0), reshape=0) 
        return canny(rec[50:-50,50:-50],'sob')
            
    def __call__(self,n):
    
        if self.reinit:
            if self.season%self.reinit==self.reinit-1:
                print 'Simulation reinitiation...'
                self.reinitiate()
            self.season += 1
            
        idx = np.arange(self.num)
        random.shuffle(idx)
        idx = idx[:n]
        return self.strings[idx]
    
#t0 = time()
#ss = Simple_String(nx=100,ny=100,num=100,augment=True,reinit=500)
#t1 = time()
#print t1-t0
#print ss.strings.shape
#sims = ss(10)
#for i in range(10):
#    img = sims[i,:,:,0]
#    plt.imshow(img,origin='lower',cmap='gray')
#    plt.show()
#    plt.close()
#    
#exit()

n_side = 4

if os.path.exists('./'+sys.argv[0][:-3]+'/sample_z'):
    sample_z = np.loadtxt('./'+sys.argv[0][:-3]+'/sample_z')
else:
    cs.ch_mkdir('./'+sys.argv[0][:-3])
    sample_z = np.random.normal(size=(n_side**2, 512))
    np.savetxt('./'+sys.argv[0][:-3]+'/sample_z',sample_z)

batch_size = 64
dp = Simple_String(nx=100,ny=100,num=100,augment=True,reinit=500)

checkpoint_dir = './'+sys.argv[0][:-3]+'/checkpoint'
sample_dir = './'+sys.argv[0][:-3]+'/samples'
log_dir = './'+sys.argv[0][:-3]+'/logs'


dcgan = cs.DCGAN(
    data_provider = dp,
    batch_size=batch_size, 
    n_side=n_side,sample_z = sample_z,
    gf_dim=32, df_dim=32,
    label_real_lower=.9, label_fake_upper=.1,
    z_dim=512,save_per = 100)

dcgan.train(num_epoch=100000,batch_per_epoch=10,sample_per=1,verbose=1,\
learning_rate=1e-4,D_update_per_batch=1,G_update_per_batch=1,\
sample_dir=sample_dir,checkpoint_dir=checkpoint_dir,\
log_dir=log_dir,time_limit=15)

cs.movie(sample_dir,output=sys.argv[0][:-3]+'/movie')









