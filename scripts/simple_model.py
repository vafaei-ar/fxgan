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
image_size = 128
gf_dim = 64
df_dim = 64
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
			h4 = lrelu (linear (tf.reshape (h3, [self.batch_size, -1]), 4, 'd_h4_lin'))
			o =  linear (tf.reshape (h4, [1, -1]), 1, 'd_out_lin')

			# h_moment_analyse = lrelu (conv2d (h3, 4, name='d_m0_conv',  k_h=1, k_w=1, d_h=1, d_w=1))
			# h_moment_analyse_size = tf.shape(h_moment_analyse)
			# moments1 = tf.concat(tf.nn.moments(h_moment_analyse, axes=[0]), axis=-1) #x y 16df
			# moments1_asbatch = tf.reshape (moments1, [h_moment_analyse_size[1] * h_moment_analyse_size[2], 4 * 2])
			# m10 = lrelu (linear(moments1_asbatch, 2, 'd_m10_lin'))
			# m11 = lrelu (linear(tf.reshape(m10, [1, -1]), 1, 'd_m11_lin'))
			# m12 = tf.tile(m11, [self.batch_size, 1])

			# moments2 = tf.concat(tf.nn.moments(h_moment_analyse, axes=[1, 2]), axis=-1) #bs 16df
			# m20 = lrelu (linear (moments2, 4, 'd_m20_lin'))
			# m21 = lrelu (linear (m20, 1, 'd_m21_lin'))

			# h5 = tf.concat([h4, m12, m21], axis=-1)
			# h5 = tf.concat ([h4, m12], axis=-1)
			# o = linear (h5, 1, 'd_output_lin')

			return tf.nn.sigmoid (o), o

dcgan = NewDCGAN (
	data_provider=dpp,
	data_postprocess=dp.postprocess,
	batch_size=batch_size,
	gf_dim=gf_dim, df_dim=df_dim,
	label_real_lower=.9, label_fake_upper=.1,
	z_dim=z_dim,
	checkpoint_dir='checkpoint')

dcgan.train (num_epoch=10000, batch_per_epoch=50, learning_rate=2e-4, D_update_per_batch=1, G_update_per_batch=1, verbose=100, sample_per=10000)
sample_z = np.random.normal(size=(200, dcgan.z_dim))
ress = dcgan.generate(sample_z)
from PIL import Image
for i in range(200):
	res= ress[i, :, :, 0]
	visual = (res - res.min()) / (res.max() - res.min())
	result = Image.fromarray((visual * 255).astype(np.uint8))
	result.save("./samples/final_%d.jpg"%i)

