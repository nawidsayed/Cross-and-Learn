import os
import numpy as np
from PIL import Image
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from experiment import utils
from experiment import Base
from compvis.models import Container

__all__ = ['Connect']

class Connect(Base):
	def __init__(self,
			name,
		 	exp_1, 
		 	exp_2, 
			learning_rate = 0.01,
			weight_decay = 0.0005,
			loss_mode = 0
		):
		super(Connect, self).__init__(name=name)
		self.exp_1 = exp_1
		self.exp_2 = exp_2
		self.learning_rate = learning_rate 
		self.weight_decay = weight_decay
		self.loss_mode = loss_mode

		self.list_infos += [
			('name_exp_1', exp_1.name),
			('name_exp_2', exp_2.name),
			('learning_rate', learning_rate),
			('weight_decay', weight_decay), 
			('loss_mode', loss_mode)]


		self.net = Container(self.exp_1.net, self.exp_2.net)

		self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate,
			momentum=0.9, weight_decay=self.weight_decay)

	def run(self, split_batch=1, strict=True):
		self._write_infos()	
		self._print_infos()	
		self.net.cuda()
		t0_tot = time()
		while True:
			loss = self.get_loss()
			if loss is None:
				break
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
		t1_tot = time()
		self.net.cpu()
		print('total runtime run: %f' %(t1_tot-t0_tot))

	def get_loss(self):
		if self.loss_mode == 0:
			loss_1 = self.exp_1.push_loss()
			loss_2 = self.exp_2.push_loss()
			return (loss_1 + loss_2) / 2
		if self.loss_mode == 1:
			loss_1 = self.exp_1.push_loss()
			loss_2 = self.exp_2.push_loss()
			return (loss_1 + loss_2)
		if self.loss_mode == 2:
			if not hasattr(self, 'cmod2'):
				self.cmod2 = 0
			self.cmod2 += 1
			if self.cmod2 % 2 == 0:
				return self.exp_1.push_loss()
			else:
				return self.exp_2.push_loss()





	