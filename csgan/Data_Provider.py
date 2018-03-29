import os
import numpy as np
import healpy as hp
import pylab as plt

if not os.path.exists('./sky2face.so'):
	os.system('f2py -c -m sky2face ./f90_src/sky2face.f90')

from sky2face import sky_to_patch

class Data_Provider(object):
	
	"""
	CLASS GeneralDataProvider: This class will provide data to feed network.

	--------
	METHODS:

	__init__:
	| arguments:
	|		files_list: list of paths to the maps. 
	|		dtype (default=np.float16): 
	|		nest (default=1): 
	|		lp (default=4096):  
	
	__call__:
	This method provides 
	| Arguments:
	|		num: number of returned patches.
	|		l: number of returned patches.

	| Returns:
	|		Image, Demand map, coordinates (if coord is true)
	"""

	def __init__(self,files_list,
				 dtype = np.float16,
				 nest = 1,
				 lp = None,
				 preprocess_mode=2):

		self.preprocess_mode = preprocess_mode

		npatch = 1
		numpa = 12

		if type(files_list) is not list:
			files_list = [files_list]

		if lp is None:
			fits_hdu = hp.fitsfunc._get_hdu(files_list[0], hdu=1, memmap=False)
			lp = fits_hdu.header.get('NSIDE')

		self.files_list = files_list
		n_files = len(files_list)
		self.patchs = np.zeros((12*n_files,lp,lp))
		for i in range(n_files):
			file_ = files_list[i]
			m = hp.read_map(file_,dtype=dtype,verbose=0,nest=nest)
			self.patchs[i*12:(i+1)*12,:,:] = sky_to_patch(m,npatch,numpa,lp)

		self.n_patch = self.patchs.shape[0]

		self.mean = self.patchs.mean()
		self.std = self.patchs.std()
		self.min = self.patchs.min()
		self.max = self.patchs.max()

		print("Data Loaded:\n\tpatch number=%d\n\tsize in byte=%d" % (self.n_patch, self.patchs.nbytes))
		print("\tmin value=%f\n\tmax value=%f\n\tmean value=%f\n\tSTD value=%f" % (self.min, self.max, self.mean, self.std))

		if self.preprocess_mode == 0:
			pass
		elif self.preprocess_mode == 1:
			# Normalize data
			self.patchs = (self.patchs - self.mean) / self.std
		elif self.preprocess_mode == 2:
			scl = float(self.max - self.min)
			self.patchs = 2. * ((self.patchs - self.min) / scl) - 1.
		elif self.preprocess_mode == 3:
			self.patchs = np.tanh(self.patchs - self.mean)
		else:
			raise Exception("invalid normalization mode")


	def postprocess(self, inp):
		if self.preprocess_mode == 0:
			return inp
		elif self.preprocess_mode == 1:
			return inp * self.std + self.mean
		elif self.preprocess_mode == 2:
			scl = float (self.max - self.min)
			return (inp + 1.) * .5 * scl + self.min
		elif self.preprocess_mode == 3:
			return np.arctanh(self.patchs) + self.mean
		else:
			raise Exception("invalid normalization mode")

		 
	def __call__(self,num,l):
					 
		l_max = self.patchs.shape[1]
		assert l<l_max,'ERROR!'
		x = np.zeros((num,l,l))

		for i in range(num):
			face = np.random.randint(self.n_patch)
			i0 = np.random.randint(l_max-l)
			j0 = np.random.randint(l_max-l)
			xx = self.patchs[face,i0:i0+l,j0:j0+l]

			xx = np.rot90(xx,np.random.randint(4))
			if 0 == np.random.randint(2):
				xx = np.flip(xx,0)

			x[i,:,:] = xx

		return x
