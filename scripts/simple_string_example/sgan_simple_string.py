import os
import sys
import argparse
import numpy as np
import csgan as cs 
import pylab as plt
from time import time
from utils import *
    
parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--show', action="store_true", default=False)
parser.add_argument('--nside', action="store", type=int, default=100)
parser.add_argument('--mode', action="store", type=str, default='none')
parser.add_argument('--time_limit', action="store", type=int, default=60)
parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
args = parser.parse_args()
show = args.show
nside = args.nside
if args.mode == 'none':
    mode = None
else:
    mode = args.mode
time_limit = args.time_limit
learning_rate = args.learning_rate

n_side = 4
batch_size = 64
if mode is None:
    save_dir = sys.argv[0][:-3]
else:
    save_dir = sys.argv[0][:-3]+'_'+mode

if os.path.exists('./'+save_dir+'/sample_z'):
    sample_z = np.loadtxt('./'+save_dir+'/sample_z')
else:
    cs.ch_mkdir('./'+save_dir)
    sample_z = np.random.normal(size=(n_side**2, 512))
    np.savetxt('./'+save_dir+'/sample_z',sample_z)

dp = Simple_String(nx=nside,ny=nside,num=100,augment=True,reinit=100,mode=mode)

if show:
    sims = dp(10)
    for i in range(10):
        img = sims[i,:,:,0]
        plt.imshow(img,origin='lower',cmap='gray')
        plt.show()
        plt.close()

checkpoint_dir = './'+save_dir+'/checkpoint'
sample_dir = './'+save_dir+'/samples'
log_dir = './'+save_dir+'/logs'

sgan = cs.SGAN(
    data_provider = dp,
    batch_size=batch_size, 
    n_side=n_side,sample_z = sample_z,
    gf_dim=32, df_dim=32,
    label_real_lower=.9, label_fake_upper=.1,
    z_dim=512,save_per = 100)
    
sgan.train(num_epoch=100000,batch_per_epoch=10,sample_per=1,verbose=1,\
learning_rate=learning_rate,D_update_per_batch=2,G_update_per_batch=1,\
sample_dir=sample_dir,checkpoint_dir=checkpoint_dir,\
log_dir=log_dir,time_limit=time_limit)

cs.movie(sample_dir,output=save_dir+'/movie')









