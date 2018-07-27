from __future__ import print_function

import numpy as np
##import matplotlib.pyplot as plt
import sys
#from .optimizer import Optimizer

class _scheduler_learning_rate(object):
    
	def __init__(self, optimizer, epoch=-1):

		self.optimizer 	= optimizer
		self.epoch 		= epoch	

	def step(self, epoch=None):
	
		if epoch is None:
            
			epoch = self.epoch + 1
		
		self.epoch	= epoch
		lr 			= self.get_lr()

		for param_group in self.optimizer.param_groups:
        
			param_group['lr'] = lr

class scheduler_learning_rate_sigmoid(_scheduler_learning_rate): 

	def __init__(self, optimizer, lr_initial, lr_final, numberEpoch, alpha=10, beta=0, epoch=-1):

		_index 		= np.linspace(-1, 1, numberEpoch)
		_sigmoid	= 1 / (1 + np.exp(alpha * _index + beta))

		val_initial = _sigmoid[0]
		val_final	= _sigmoid[-1]

		a = (lr_initial - lr_final) / (val_initial - val_final)
		b = lr_initial - a * val_initial 

		self.schedule		= a * _sigmoid + b
		self.numberEpoch	= numberEpoch

		super(scheduler_learning_rate_sigmoid, self).__init__(optimizer, epoch)

	def get_lr(self, epoch=None):

		if epoch is None:

			epoch = self.epoch

		lr = self.schedule[epoch]

		return lr
"""
	def plot(self):

		fig 	= plt.figure()
		ax		= fig.add_subplot(111)	
		
		ax.plot(self.schedule)
		
		plt.xlim(0, self.numberEpoch + 1)
		plt.xlabel('epoch')
		#plt.ylabel('learning rate')
		plt.grid(linestyle='dotted')
		plt.tight_layout()
		plt.show()
"""
