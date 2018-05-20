import cv2
import random
import numpy as np
from skimage.draw import polygon
from scipy.ndimage.interpolation import rotate

def normal(x):
    xmax = x.max()
    if xmax!=0:
        x = x/xmax
    return x

def canny(d,edege):

    if edege=='lap':
        d = cv2.Laplacian(d,cv2.CV_64F)

    elif edege=='sob':
        sobelx = cv2.Sobel(d,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(d,cv2.CV_64F,0,1,ksize=3)
        d =np.sqrt(sobelx**2+sobely**2)

    elif edege=='sch':
        scharrx = cv2.Scharr(d,cv2.CV_64F,1,0)
        scharry = cv2.Scharr(d,cv2.CV_64F,0,1)
        d =np.sqrt(scharrx**2+scharry**2)
        
    else:
        assert 0,'Mode have to be lap, sob or sch!' 

    return d

class Simple_String(object):

    def __init__(self,nx=200,ny=200,num=100,augment=True,
                reinit=0,ns=20,l_min=None,l_max=None,mode=None,supervised=False):
                
        if supervised:
            assert mode is not None,'Supervised GAN have to use a filter, please choose a filter mode!'
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
        self.mode = mode
        self.strings = np.zeros((num,nx,ny,1))
        if supervised:
            self.featured = np.zeros((num,nx,ny,1))
        self.supervised = supervised
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
                    self.strings[i+j,:,:,0] = normal(self.strings[i+j,:,:,0])
                    if self.supervised:
                        self.featured[i+j,:,:,0] = canny(self.strings[i+j,:,:,0],self.mode)
        else:
            for i in range(self.num):
                self.strings[i,:,:,0] = self.string()
                self.strings[i,:,:,0] = normal(self.strings[i,:,:,0])
                if self.supervised:
                    self.featured[i,:,:,0] = canny(self.strings[i,:,:,0],self.mode)
        

        
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
            
        if self.mode is None or self.supervised:
            return rec[50:-50,50:-50]
        else:
            return canny(rec[50:-50,50:-50],self.mode)
            
    def __call__(self,n):
    
        if self.reinit:
            if self.season%self.reinit==self.reinit-1:
                print 'Reinitiation...'
                self.reinitiate()
            self.season += 1
            
        idx = np.arange(self.num)
        random.shuffle(idx)
        idx = idx[:n]
        if self.supervised:
            return self.strings[idx],self.featured[idx]
        else:
            return self.strings[idx]
    







