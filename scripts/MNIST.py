import os
import sys
import random
import numpy as np

import csgan as cs

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../../dataset/mnist/")

x_train = mnist.train.images[:55000,:]
x_train.shape

class data_provider(object):
    def __init__(self,x_train):
        
        x_train = x_train/np.max(x_train,axis=1)[:,None]
        self.x_train = x_train
        self.num = x_train.shape[0]
        
    def __call__(self,n):
        n_list = np.arange(self.num)
        random.shuffle(n_list)
        return self.x_train[n_list[:n]].reshape(n,28,28,1)
        


batch_size = 64
image_size = 256


dp = data_provider(x_train)
    
#image = dp(10)[0].reshape([28,28])
#plt.imshow(image, cmap=plt.get_cmap('gray_r'))
#plt.show()

checkpoint_dir = './'+sys.argv[0][:-3]+'/checkpoint'
sample_dir = './'+sys.argv[0][:-3]+'/samples'
log_dir = './'+sys.argv[0][:-3]+'/logs'

dcgan = cs.DCGAN(
    data_provider = dp,
    batch_size=64, gf_dim=64, df_dim=64,
    z_dim=100, save_per = 100)

dcgan.train(num_epoch=100000, batch_per_epoch = 10, verbose=10, learning_rate=1e-4, sample_per=5, sample_dir=sample_dir, checkpoint_dir=checkpoint_dir, log_dir=log_dir, time_limit=60)















