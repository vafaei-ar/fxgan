modules = ['Data_Provider','model','sgan']

for module in modules:
	exec('from '+module+' import *')
#
# import sys
# sys.path.insert(0,'./')
#
