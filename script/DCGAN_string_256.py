import os
import random
import numpy as np
import pylab as plt

import sys
import csgan as cs

file_list = ['../../dataset/map1n_allz_rtaapixlw_2048_'+str(i)+'.fits' for i in range(1,4)]
dp = cs.Data_Provider(file_list,preprocess_mode=2)

batch_size = 64
image_size = 256
checkpoint_dir = './checkpoint/'+sys.argv[0][:-3]
sample_dir = './samples/'+sys.argv[0][:-3]

def dpp(n):
    return dp(n,image_size).reshape(n,image_size,image_size,1)

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

