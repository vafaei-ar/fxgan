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

batch_size = 64
image_size = 128
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

			h00 = lrelu (conv2d (image, self.df_dim, name='d_h0_conv'))
			h01 = lrelu (self.d_bn1 (conv2d (h00, self.df_dim * 2, name='d_h01_conv')))
			h02 = lrelu (self.d_bn2 (conv2d (h01, self.df_dim * 4, name='d_h02_conv')))
			h03 = lrelu (self.d_bn3 (conv2d (h02, self.df_dim * 8, name='d_h03_conv')))
			h04 = linear (tf.reshape (h03, [self.batch_size, -1]), 1, 'd_h04_lin') # bs, 1
			h05 = linear (tf.reshape (h04, [1, -1]), 1, 'd_h05_lin')

			inp = tf.transpose (image, [3, 1, 2, 0])
			h10 = lrelu (conv2d (inp, 4, name='d_h10_conv'))
			h11 = lrelu (conv2d (h10, 2, name='d_h11_conv'))
			h12 = lrelu (conv2d (h11, 1, name='d_h12_conv'))
			h13 = linear (tf.reshape (h12, [1, -1]), 1, 'd_h13_lin') # 1, 1

			o = linear (tf.concat ([h05, h13], axis=1), 1, 'd_output')

			return tf.nn.sigmoid (o), o

dcgan = NewDCGAN (
	data_provider=dpp,
	data_postprocess=dp.postprocess,
	batch_size=batch_size,
	gf_dim=gf_dim, df_dim=df_dim,
	label_real_lower=.9, label_fake_upper=.1,
	z_dim=z_dim,
	checkpoint_dir='checkpoint')

dcgan.train (num_epoch=4000, batch_per_epoch=50, learning_rate=8e-5, D_update_per_batch=1, G_update_per_batch=1, verbose=100, sample_per=10000)
sample_z = np.random.normal(size=(200, dcgan.z_dim))
ress = dcgan.generate(sample_z)
from PIL import Image
for i in range(200):
	res= ress[i, :, :, 0]
	visual = (res - res.min()) / (res.max() - res.min())
	result = Image.fromarray((visual * 255).astype(np.uint8))
	result.save("./samples/%d.jpg"%i)

