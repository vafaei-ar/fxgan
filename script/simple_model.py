import os
import numpy as np
import pylab as plt

import sys
sys.path.insert(0, "./")

import csgan as cs

if os.path.exists('../dataset/'):
	prefix = '../dataset/'
elif os.path.exists('../../dataset/'):
	prefix = '../../dataset/'
else:
	raise Exception ('Dataset not found!')

dataset_files = ['map1n_allz_rtaapixlw_2048_1.fits', 'map1n_allz_rtaapixlw_2048_2.fits', 'map1n_allz_rtaapixlw_2048_3.fits']
dp = cs.Data_Provider ([prefix + file_name for file_name in dataset_files], preprocess_mode=2)

# dt = filt_all(dp(10,128),func)
# dt.shape
# fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,18))
# ax1.imshow(dt[0,:,:,0])
# ax2.imshow(dt[0,:,:,1])


batch_size = 16
image_size = 128
gf_dim = 64
df_dim = 64
z_dim = 200


def dpp(n):
	return dp(n, image_size).reshape(n, image_size, image_size, 1)

# run_config.gpu_options.allow_growth = True

dcgan = cs.DCGAN (
	data_provider=dpp,
	data_postprocess=dp.postprocess,
	batch_size=batch_size,
	gf_dim=gf_dim, df_dim=df_dim,
	label_real_lower=.9, label_fake_upper=.1,
	z_dim=z_dim,
	checkpoint_dir='checkpoint')

dcgan.train (num_epoch=500, batch_per_epoch=50, learning_rate=1e-4, D_update_per_batch=1, G_update_per_batch=1, verbose=100)

