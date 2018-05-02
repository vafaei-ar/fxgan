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

defult_model_build=0 

dcgan = cs.DCGAN(
    data_provider = dpp,
    data_postprocess = dp.postprocess,
    batch_size=64, gf_dim=64, df_dim=64,
    label_real_lower=.9, label_fake_upper=.1,
    z_dim=2048,checkpoint_dir=checkpoint_dir,
    save_per = 100, defult_model_build=defult_model_build)

if ~defult_model_build:
    import tensorflow as tf

    def discriminator(image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            trainable = 1

            h0 = tf.layers.conv2d(inputs=image, filters=dcgan.df_dim, kernel_size=[5,5], 
                     strides=(2,2),padding='same',
                     activation=tf.nn.relu, name='d_h0_conv')

            h1 = tf.layers.conv2d(inputs=h0, filters=2*dcgan.df_dim, kernel_size=[5,5], 
                     strides=(2,2),padding='same',
                     activation=None, name='d_h1_conv')
            h1 = tf.contrib.layers.batch_norm(h1,decay=0.9,updates_collections=None,
                            epsilon=1e-5,scale=True,is_training=trainable,scope='d_bn1')
            h1 = tf.nn.relu(h1)

            h2 = tf.layers.conv2d(inputs=h1, filters=4*dcgan.df_dim, kernel_size=[5,5], 
                     strides=(2,2),padding='same',
                     activation=None, name='d_h2_conv')
            h2 = tf.contrib.layers.batch_norm(h2,decay=0.9,updates_collections=None,
                            epsilon=1e-5,scale=True,is_training=trainable,scope='d_bn2')
            h2 = tf.nn.relu(h2)

            h3 = tf.layers.conv2d(inputs=h2, filters=8*dcgan.df_dim, kernel_size=[5,5], 
                     strides=(2,2),padding='same',
                     activation=None, name='d_h3_conv')
            h3 = tf.contrib.layers.batch_norm(h3,decay=0.9,updates_collections=None,
                            epsilon=1e-5,scale=True,is_training=trainable,scope='d_bn3')
            h3 = tf.nn.relu(h3)

            h4 = tf.layers.dense(inputs=tf.reshape(h3, [dcgan.batch_size, -1]),
                                 units=1,kernel_initializer=tf.random_normal_initializer(stddev=0.02), 
                                 bias_initializer=tf.constant_initializer (0.01),
                                 use_bias=1,activation=None,name='d_h4_lin')

        return tf.nn.sigmoid(h4), h4

    def generator(z, batch_size, mode='train'):
        with tf.variable_scope("generator") as scope:
            if mode=='train':
                trainable = True
                pass
            elif mode=='sampler':
                trainable = False
                scope.reuse_variables()
            else:
                assert 0,'Unkown mode for generator.'

            s_h, s_w = dcgan.output_height, dcgan.output_width
            s_h2, s_w2 = cs.conv_out_size_same(s_h, 2),  cs.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = cs.conv_out_size_same(s_h2, 2),  cs.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = cs.conv_out_size_same(s_h4, 2),  cs.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = cs.conv_out_size_same(s_h8, 2),  cs.conv_out_size_same(s_w8, 2)

            # assert s_h16*s_w16*self.gf_dim*8==z.shape[1],str(s_h16*s_w16*self.gf_dim*8)+' != '+str(z.shape[1])

            # project `z` and reshape
            dcgan.z_ = tf.layers.dense(z,dcgan.gf_dim * 8 * s_h16 * s_w16,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.02), 
                                bias_initializer=tf.constant_initializer (0.01),
                                use_bias=1,activation=None,name='g_h0_lin')
            with tf.variable_scope('g_h0_lin', reuse=True):
                dcgan.h0_w = tf.get_variable('kernel')
                dcgan.h0_b = tf.get_variable('bias')
            dcgan.h0 = tf.reshape(dcgan.z_, [batch_size, s_h16, s_w16, dcgan.gf_dim * 8])
            h0 = tf.contrib.layers.batch_norm(dcgan.h0,decay=0.9,updates_collections=None,
                            epsilon=1e-5,scale=True,is_training=trainable,scope='g_bn0')
            h0 = tf.nn.relu(h0)

            dcgan.h1, dcgan.h1_w, dcgan.h1_b = cs.deconv2d(h0, [batch_size, s_h8, s_w8, dcgan.gf_dim * 4], 
                                                    k_h=9, k_w=9, name='g_h1', with_w=True)
            h1 = tf.contrib.layers.batch_norm(dcgan.h1,decay=0.9,updates_collections=None,
                            epsilon=1e-5,scale=True,is_training=trainable,scope='g_bn1')
            h1 = tf.nn.relu(h1) 

            h2, dcgan.h2_w, dcgan.h2_b = cs.deconv2d(h1, [batch_size, s_h4, s_w4, dcgan.gf_dim * 2],
                                                     k_h=9, k_w=9, name='g_h2', with_w=True)
            h2 = tf.contrib.layers.batch_norm(h2,decay=0.9,updates_collections=None,
                            epsilon=1e-5,scale=True,is_training=trainable,scope='g_bn2')
            h2 = tf.nn.relu(h2) 

            h3, dcgan.h3_w, dcgan.h3_b = cs.deconv2d(h2, [batch_size, s_h2, s_w2, dcgan.gf_dim * 1],
                                                     k_h=7, k_w=7, name='g_h3', with_w=True)
            h3 = tf.contrib.layers.batch_norm(h3,decay=0.9,updates_collections=None,
                            epsilon=1e-5,scale=True,is_training=trainable,scope='g_bn3')
            h3 = tf.nn.relu(h3) 

            h4, dcgan.h4_w, dcgan.h4_b = cs.deconv2d(h3, [batch_size, s_h, s_w, dcgan.c_dim],
                                                     k_h=5, k_w=5, name='g_h4', with_w=True)

            return tf.nn.tanh(h4)
        
    dcgan.discriminator = discriminator
    dcgan.generator = generator
    dcgan.build_model()

dcgan.train(num_epoch=100000,batch_per_epoch=50,verbose=10,\
learning_rate=1e-4,D_update_per_batch=1,G_update_per_batch=1,\
sample_dir=sample_dir,checkpoint_dir=checkpoint_dir,time_limit=600)

