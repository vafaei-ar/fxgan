#!/usr/bin/env python

import os
import sys
import shutil
from setuptools.command.test import test as TestCommand
from setuptools import find_packages

def remove_dir(dirpath):
	if os.path.exists(dirpath) and os.path.isdir(dirpath):
		  shutil.rmtree(dirpath)

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from numpy.distutils.core import Extension
from numpy.distutils.core import setup

extensions = Extension(name = 'sky2face',
#                 extra_compile_args = ['-O3'],
                 sources = ['f90_src/sky2face.f90'],
                 libraries=['gfortran'] 
                 )

extensions.extra_f77_compile_args = []
extensions.extra_f90_compile_args = []

requires = [] #during runtime
tests_require=['pytest>=2.3'] #for testing

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
	name='csgan',
	version='0.1.0',
	description='Generative adveserial network for cosmic string simulation.',
	author='Alireza',
	url='https://github.com/vafaeiar/csgan',
	packages=find_packages(PACKAGE_PATH, "csgan"),
	package_dir={'csgan': 'csgan'},
	include_package_data=True,
	install_requires=requires,
	license='GPLv3',
	zip_safe=False,
	keywords='csgan',
	classifiers=[
		  'Development Status :: 2 - Pre-Alpha',
		  "Intended Audience :: Science/Research",
		  'Intended Audience :: Developers',
		  "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
		  'Natural Language :: English',
		  'Programming Language :: Python :: 2',
		  'Programming Language :: Python :: 2.6',
		  'Programming Language :: Python :: 2.7',
		  'Programming Language :: Python :: 3',
		  'Programming Language :: Python :: 3.3',
		  'Programming Language :: Fortran',
	],
	ext_modules=[extensions]
)

remove_dir('build')
remove_dir('csgan.egg-info')
remove_dir('dist')
