import os
import random
import numpy as np
import pylab as plt

import sys
import csgan as cs

file_list = ['../../dataset/map1n_allz_rtaapixlw_2048_'+str(i)+'.fits' for i in range(1,4)]

class Sample_Data_Provider(cs.Data_Provider):
	def preprocess(self,inp):
		scl = float(self.max - self.min)
		return 2. * ((inp - self.min) / scl) - 1.
		
	def postprocess(self,inp):
		scl = float (self.max - self.min)
		return (inp + 1.) * .5 * scl + self.min

batch_size = 64
image_size = 256
dp = Sample_Data_Provider(file_list,image_size)

checkpoint_dir = './'+sys.argv[0][:-3]+'/checkpoint'
sample_dir = './'+sys.argv[0][:-3]+'/samples'
log_dir = './'+sys.argv[0][:-3]+'/logs'

dcgan = cs.DCGAN(
    data_provider = dp,
    batch_size=batch_size, gf_dim=64, df_dim=64,
    label_real_lower=.9, label_fake_upper=.1,
    z_dim=2048,save_per = 100)

dcgan.train(num_epoch=100000,batch_per_epoch=50,verbose=10,\
learning_rate=1e-4,D_update_per_batch=1,G_update_per_batch=1,\
sample_dir=sample_dir,checkpoint_dir=checkpoint_dir,log_dir=log_dir,time_limit=10)

