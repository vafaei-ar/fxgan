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

batch_size = 16
image_size = 64
gf_dim = 32
df_dim = 32
z_dim = 128


def dpp(n):
	return dp(n, image_size).reshape(n, image_size, image_size, 1)

##################################################################
from csgan.ops import lrelu, conv2d, linear
import tensorflow as tf

class NewDCGAN(cs.DCGAN):
	def discriminator(self, image, reuse=False):
		with tf.variable_scope ("discriminator") as scope:
			if reuse:
				scope.reuse_variables ()

			h0 = lrelu (conv2d (image, self.df_dim, name='d_h0_conv'))
			h1 = lrelu (self.d_bn1 (conv2d (h0, self.df_dim * 2, name='d_h1_conv')))
			h2 = lrelu (self.d_bn2 (conv2d (h1, self.df_dim * 4, name='d_h2_conv')))
			h3 = lrelu (self.d_bn3 (conv2d (h2, self.df_dim * 8, name='d_h3_conv')))
			h4 = linear (tf.reshape (h3, [1, -1]), self.batch_size * self.df_dim, 'd_h4_lin')
			h5 = linear (tf.reshape (h4, [1, -1]), self.batch_size, 'd_h5_lin')
			h6 = linear (tf.reshape (h5, [1, -1]), 1, 'd_h6_lin')

			return tf.nn.sigmoid (h6), h6

dcgan = NewDCGAN (
	data_provider=dpp,
	data_postprocess=dp.postprocess,
	batch_size=batch_size,
	gf_dim=gf_dim, df_dim=df_dim,
	label_real_lower=.9, label_fake_upper=.1,
	z_dim=z_dim,
	checkpoint_dir='checkpoint')

dcgan.train (num_epoch=2000, batch_per_epoch=50, learning_rate=1e-4, D_update_per_batch=1, G_update_per_batch=1, verbose=100, sample_per=10000)
sample_z = np.random.normal(size=(200, dcgan.z_dim))
ress = dcgan.generate(sample_z)
from PIL import Image
for i in range(200):
	res= ress[i, :, :, 0]
	visual = (res - res.min()) / (res.max() - res.min())
	result = Image.fromarray((visual * 255).astype(np.uint8))
	result.save("./samples/%d.jpg"%i)

