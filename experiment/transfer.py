import os
import numpy as np
from PIL import Image
from time import time
import _pickle as pickle
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from experiment import utils
from experiment import Base, Base_experiment_pretraining, Base_experiment
from experiment import Experiment_pretraining_fm
from experiment.tracker import Tracker_similarity

import compvis.data as data
from compvis import transforms_det as transforms 
from compvis.datasets import Dataset_OF, Dataset_RGB
from compvis.models import get_network, AlexNet, CaffeNet_BN

__all__ = ['Transfer_fm']

eps = 0.00001

def cos_sim(f_1, f_2):
	len_1 = torch.sqrt(torch.sum(f_1 ** 2, dim=1) + eps)
	len_2 = torch.sqrt(torch.sum(f_2 ** 2, dim=1) + eps)
	return torch.sum(f_1 * f_2, dim=1) / (len_1 * len_2)

def euc_sim(f_1, f_2):
	sim = cos_sim(f_1, f_2)
	if not isinstance(sim, float):
		return 1 - torch.sqrt(2-2*sim + eps)
	else:
		return 1 - np.sqrt(2-2*sim + eps)

def lin_sim(f_1, f_2):
	sim = cos_sim(f_1, f_2)
	if not isinstance(sim, float):
		return 1 - 2*torch.acos(sim) / np.pi
	else:
		return 1 - 2*np.arccos(sim) / np.pi

def own_sim(f_1, f_2):
	sim = cos_sim(f_1, f_2)
	return 1 - 0.5 * (1-sim)**2

class Transfer_fm(Base_experiment):
	arch_dict = {'alex':AlexNet, 'caffe_bn':CaffeNet_BN}
	dataloader = None
	tracker = None
	save_intervall = None
	test_intervall = None
	net = None
	batch_size_test = None
	optimizer = None
	def __init__(self,
			exp,
			name_transfer,
			epochs = 100,
			learning_rate = 0.005,
			lr_decay_scheme = 0,
			weight_decay = 0.0005,
			data_key = 'all',
			source = 'l',
			norm = 'caffe_bn',
			load_epoch_pt = -1,
			layer = 'fc6',
			transfer_scheme = 'rgb',
			dropout = 0.5,
			similarity = 'cosine'
		):
		name = exp.name
		batch_size = exp.batch_size
		super(Transfer_fm, self).__init__(name=name, batch_size=batch_size, epochs=epochs,
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay,
		 	data_key=data_key, source=source)
		self.exp = exp
		self.results_dir = os.path.join(self.results_dir, name_transfer)
		if not utils.mkdir(self.results_dir):
			print('Existing transfer for given name')
		self.name_transfer = name_transfer
		self.norm = norm
		self.load_epoch_pt = load_epoch_pt
		self.layer = layer
		self.transfer_scheme = transfer_scheme
		self.dropout = dropout
		self.similarity = similarity
		self.list_infos += [('load_epoch_pt', load_epoch_pt), ('name_transfer', name_transfer),
			('norm', norm), ('layer', layer), ('transfer_scheme', transfer_scheme),
			('dropout', dropout), ('similarity', similarity)]
		self.exp._load_net(load_epoch=self.load_epoch_pt)
		self.net_pt = self.exp.net
		arch = self.arch_dict[self.norm]
		self.net = arch(dropout=self.dropout)
		self.Tracker = Tracker_similarity()
		self.save_intervall = 50
		self.test_intervall = 10
		self.batch_size_test = self.exp.batch_size_test
		self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate,
			momentum=0.9, weight_decay=self.weight_decay)
		self.sim_func = cos_sim

	def run(self, resume_training=0):
		self.net_pt.cuda()
		super(Transfer_fm, self).run(resume_training=resume_training)
		self.net_pt.cpu()

	def _training(self, split_batch=1):
		self.net_pt.train(mode=True)
		super(Transfer_fm, self)._training(split_batch=split_batch)

	def _testing(self, split_batch=1):
		self.net_pt.train(mode=False)
		super(Transfer_fm, self)._testing(split_batch=split_batch)

	def _get_data(self, iterator):
		self.transfer_scheme_cur = self.transfer_scheme
		if self.transfer_scheme == 'random_rgb_of':
			l = ['rgb', 'of']
			np.random.shuffle(l)
			self.transfer_scheme_cur = l[0]
		samples = next(iterator, None)
		if samples is None:
			return None, None
		data = []
		for field in samples:
			field = torch.cat(field, dim=0).cuda()
			field = Variable(field)
			data.append(field)
		return data, None

	def _forward(self, data):
		# var [30x3 30x20 30x3 30x20]
		data_train = self._prepare_data(data)
		dismissed = self._prepare_dismissed()
		features = self.net_pt.get_feature_output(*data, layer=self.layer, dismissed=dismissed)
		# features [30x4096, None, 30x4096, None]
		features_pt = self._prepare_features(features)
		# features_pt [60x4096]
		features_train = self.net.get_feature_output(*data_train, layer=self.layer)
		sim_true = self.sim_func(features_pt, features_train)
		sim_false = sim_true * 0
		lengths = sim_false
		return sim_true, sim_false, lengths

	def _get_loss(self, output, labels):
		norms = output[-1]
		output = output[:-1]
		half = int(len(output) / 2)
		sim_true = 0
		sim_false = 0
		for i in range(half):
			# sim_true += output[i]
			# sim_false += output[half + i]

			sim_true += self._distance_transformation(output[i])
			sim_false += self._distance_transformation(output[half + i])
		loss_sim = torch.mean(sim_false - sim_true, dim=0) / half
		return loss_sim

	def _distance_transformation(self, sim):
		if self.similarity == 'cosine':
			return sim
		if self.similarity == 'euclidean':
			return 1 - torch.sqrt(2-2*sim + eps)

	def _set_dataloader(self):
		print('_set_dataloader deprecated here')

	# should make use of _reconfigure_dataloader()
	def _reconfigure_dataloader_train(self):
		self.exp._reconfigure_dataloader_train()
		self.dataloader = self.exp.dataloader

	# should make use of _reconfigure_dataloader()
	def _reconfigure_dataloader_test(self):
		self.exp._reconfigure_dataloader_test()
		self.dataloader = self.exp.dataloader

	def _reconfigure_tracker_train(self):
		names = ['p', 'n']
		self.tracker = Tracker_similarity(names)

	def _reconfigure_tracker_test(self):
		names = ['p', 'n']
		self.tracker = Tracker_similarity(names)	

	def _prepare_data(self, data):
		if self.transfer_scheme_cur == 'rgb':
			return [torch.cat([data[0], data[2]], dim=0)]
		if self.transfer_scheme_cur == 'of':
			return [torch.cat([data[0], data[2]], dim=0)]
		if self.transfer_scheme_cur == 'centroid_rgb_of':
			return [torch.cat([data[0], data[2]], dim=0)]

	def _prepare_dismissed(self):
		if self.transfer_scheme_cur == 'rgb':
			return ['of']
		if self.transfer_scheme_cur == 'of':
			return ['rgb']
		if self.transfer_scheme_cur == 'centroid_rgb_of':
			return []

	def _prepare_features(self, features):
		if self.transfer_scheme_cur == 'rgb':
			return torch.cat([features[0], features[2]], dim=0)
		if self.transfer_scheme_cur == 'of':
			return torch.cat([features[1], features[3]], dim=0)
		if self.transfer_scheme_cur == 'centroid_rgb_of':
			features_norm = []
			for feature in features:
				norm = torch.sqrt(torch.sum(feature**2, dim=1, keepdim=True)) + 0.000001
				feature_norm = feature / norm
				features_norm.append(feature_norm)
			return torch.cat([features_norm[0]+features_norm[1], features_norm[2]+features_norm[3]])

			
