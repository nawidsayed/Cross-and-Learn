import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import yaml

import torch
from time import time
from experiment import utils

import compvis.data as data
from compvis import transforms_det as transforms

from compvis.datasets import UCF101_i #, HMDB51_i, ACT_i, Leeds_i, OlympicSports_i
from compvis.models import get_network, Net_ar


# Net doesnt need get_setup, every experiment gets his own name and own file, run is executed only once
# preloading nets require name of other experiment and epoch

# TODO list_infos could have substructure for net_info, data_info, training_info 
# TODO inherit experiment names from experiment structure
# TODO inherit data_keys
# TODO where to put feature mat generation? What is an axperiment actaully?
# -> every experiment has one dataset and one net! but not necessarily one folder... what about epoch?

# Every name has an unique experiment and vice versa
# Every name can have multiple evals even from same subclass, 
# Source is now a general attribute of a dataset_info 

path_config = 'config.yml'

def split_item(item, split_batch):
	item_split = []	
	if item is not None:
		if isinstance(item, (tuple, list)):
			bs = item[0].size(0)
		else:
			bs = item.size(0)
		if bs % split_batch != 0:
			raise Exception('split_batch doesnt divide batch_size') 
		length = int(bs / split_batch)
		for i in range(split_batch):
			if isinstance(item, (tuple, list)):
				item_mini = []
				for field in item:
					item_mini.append(field[i*length:(i+1)*length])
			else:
				item_mini = item[i*length:(i+1)*length]	
			item_split.append(item_mini)
	else:
		for i in range(split_batch):
			item_split.append(None)
	return item_split

def split_data_labels(data, labels, split_batch):
 data_split = split_item(data, split_batch)
 labels_split = split_item(labels, split_batch)
 return data_split, labels_split

class Base(object):
	max_digits = 4
	mean = [0.485, 0.456, 0.406] 
	std = [0.229, 0.224, 0.225]
	dict_dataset_info_types = {'ucf':[UCF101_i]} #, 'hmdb':[HMDB51_i], 'act':[ACT_i], 
		# 'all':[UCF101_i, HMDB51_i, ACT_i], 'leeds':[Leeds_i], 'olympic':[OlympicSports_i],
		# 'all_olympic':[UCF101_i, HMDB51_i, ACT_i, OlympicSports_i]}
	def __init__(self,
			name,
			data_key='ucf',
		):
		self.name = name
		self.data_key = data_key

		with open(path_config, 'r') as ymlfile:
			cfg = yaml.load(ymlfile)	
		self.results_path = cfg['path_results']
		utils.mkdir(self.results_path)

		self.results_dir = os.path.join(self.results_path, self.name)
		utils.mkdir(self.results_dir)

		self.list_infos = [('name', self.name), ('data_key', data_key)]
		self.dataset_info_types = self.dict_dataset_info_types[data_key] 

	def _reconfigure_dataloader(self, dataset, batch_size, shuffle):
		if self.dataloader is None:
			self._set_dataloader(dataset)
		else:
			self.dataloader.reconfigure_dataset(dataset)
		self.dataloader.reconfigure_shuffle(shuffle)
		self.dataloader.reconfigure_batch_size(batch_size)

	def _write_infos(self):
		path = os.path.join(self.results_dir, 'infos.txt')
		f = open(path, 'a')
		f.write(utils.print_iterable(self.list_infos, delimiter='\n'))
		f.close()

	def _print_infos(self):
		path = os.path.join(self.results_dir, 'infos.txt')
		print(utils.print_iterable(self.list_infos, delimiter='\n'))

	def _write_progress(self, name, result):
		delimiter = ' '
		path = os.path.join(self.results_dir, '%s.txt' %name)
		f = open(path, 'a')
		f.write(utils.print_iterable(result, delimiter=' ', max_digits=self.max_digits, print_keys=False))
		f.write('\n')
		f.close()	

	# This is done once for every experiment to init dataloader, 
	# choices for num_workers, drop_last, shuffle, ... are made here
	def _set_dataloader(self, dataset):
		raise NotImplementedError('_set_dataloader() in Base not implemented')		

	@property
	def dataloader(self):
		raise NotImplementedError('Base_experiment should implement dataloader (compvis.data.dataloader)')

class Base_experiment(Base):
	test_epoch_zero = False
	def __init__(self,
			name,
			batch_size,
			epochs,
			learning_rate=0.01,
			lr_decay_scheme=0,
			weight_decay=0.0005,
			data_key='ucf',
		):
		super(Base_experiment, self).__init__(name=name, data_key=data_key)
		self.batch_size = batch_size
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.lr_decay_scheme = lr_decay_scheme
		self.weight_decay = weight_decay

		self.list_infos += [('batch_size', batch_size), ('epochs', epochs), 
			('learning_rate', learning_rate), ('lr_decay_scheme', lr_decay_scheme), 
			('weight_decay', weight_decay)]

		self.is_training = False

	def push_loss(self):
		if not hasattr(self, 'epoch'):
			self._before_training()
		if self.iterator is None:
			self._before_epoch()
		_, loss = self._iteration()
		if loss is None:
			return self._after_epoch()
		return loss

	def push_data(self):
		if not hasattr(self, 'epoch'):
			self._before_training()
		if self.iterator is None:
			self._before_epoch()
		data, _ = self._iteration()
		if data is None:
			return self._after_epoch()
		return data

	def _before_training(self):
		self.iterator = None
		self.epoch = 0
		self._write_infos()	
		self._print_infos()	
		self._save_net_infos()
		self._reconfigure_dataloader_tracker_train()

	def _before_epoch(self):
		self._apply_per_epoch()
		self.epoch += 1
		self.iterator = iter(self.dataloader)

	def _iteration(self):
		data, labels = self._get_data(self.iterator)
		if data is None:
			return None, None
		output = self._forward(data)
		loss = self._get_loss(output, labels)	
		loss_copy = loss.clone()
		self.tracker.update(output, labels, loss_copy)
		return data, loss

	def _after_epoch(self):
		result = self.tracker.result()
		result = [('epoch', self.epoch)] + result
		print('train ' + utils.print_iterable(result, max_digits=self.max_digits))
		self._write_progress('train', result)
		if self.epoch % 10 == 0:
			self._save_net_infos(latest=True)
		if self.epoch % self.save_intervall == 0:
			self._save_net_infos()
		if self.epoch != self.epochs:
			self.iterator = None
			return self.push_loss()
		else:
			return None

	def run(self, resume_training=0, split_batch=1, strict=True, final_test_runs=1):
		if resume_training != 0:
			self._load_net(resume_training, strict=strict)
		self.net.cuda()
		self.epoch = 0
		t0_tot = time()
		self._write_infos()	
		self._print_infos()	
		self._reconfigure_dataloader_tracker_train()
		for _ in range(resume_training):
			self._apply_per_epoch()
			self.epoch += 1
		self._save_net_infos()
		if self.test_epoch_zero and self.epoch == 0:
			self._testing(split_batch=split_batch)
		for _ in range(self.epochs-resume_training):
			self._apply_per_epoch()
			self.epoch += 1
			self._training(split_batch=split_batch)
			if self.epoch % 10 == 0:
				self._save_net_infos(latest=True)
			if self.epoch % self.save_intervall == 0 or self.epoch == self.epochs:
				self._save_net_infos()
			if self.epoch == self.epochs:
				self._save_net_infos(latest=True)
			if self.epoch % self.test_intervall == 0 or self.epoch == self.epochs:
				self._reconfigure_dataloader_tracker_test()
				self._testing(split_batch=split_batch)
				self._reconfigure_dataloader_tracker_train()
		for _ in range(final_test_runs-1):
			self.epoch += 1
			self._training(split_batch=split_batch)
			self.epoch += 1
			self._training(split_batch=split_batch)
			self._reconfigure_dataloader_tracker_test()
			self._testing(split_batch=split_batch)
			self._reconfigure_dataloader_tracker_train()		
		self.net.cpu()
		t1_tot = time()
		print('total runtime run: %f' %(t1_tot-t0_tot))

	def _training(self, split_batch=1):
		self.is_training = True
		t0 = time()
		self.net.train(mode=True)
		iterator = iter(self.dataloader)
		loss_cum = 0
		counter_cum = 0
		while(True):
			data, labels = self._get_data(iterator)
			if data is None:
				break
			self.optimizer.zero_grad()
			# data_split, labels_split = split_data_labels(data, labels, split_batch)
			# for i in range(split_batch):
			# 	data_mini = data_split[i]
			# 	labels_mini = labels_split[i]
			# 	output = self.net(*data_mini)
			# 	loss = self._get_loss(output, labels_mini) / split_batch
			# 	loss.backward()
			# 	self.tracker.update(output, labels_mini, loss)
			output = self._forward(data)
			loss = self._get_loss(output, labels)
			loss.backward()
			self.optimizer.step()
			self.tracker.update(output, labels, loss)
			self._track_per_iteration(output, labels, loss)
			loss_cum += float(loss)
			counter_cum += 1
		loss_cum /= counter_cum
		if self._interrupt_training(loss_cum):
			raise Exception('Training interrupted loss: %f' %loss_cum)
		result = self.tracker.result()
		t1 = time()
		runtime = t1-t0
		result = [('epoch', self.epoch), ('runtime', runtime)] + result
		print('train ' + utils.print_iterable(result, max_digits=self.max_digits))
		self._write_progress('train', result)

	def _testing(self, split_batch=1):
		self.is_training = False
		t0 = time()
		self.net.train(mode=False)
		iterator = iter(self.dataloader)
		while(True):
			data, labels = self._get_data(iterator)
			if data is None:
				break
			data_split, labels_split = split_data_labels(data, labels, split_batch)
			output_minis = []
			labels_minis = []
			if split_batch != 1:
				for i in range(split_batch):
					data_mini = data_split[i]
					labels_mini = labels_split[i]
					output_mini = self._forward(data_mini)
					output_minis.append(output_mini.detach())
				output = torch.cat(output_minis)
			else:
				output = self._forward(data)
			loss = self._get_loss(output, labels)
			self.tracker.update(output, labels, loss)
		result = self.tracker.result()
		t1 = time()
		runtime = t1-t0
		result = [('epoch', self.epoch), ('runtime', runtime)] + result
		print('-------------------------------------------------------------------')
		print('test ' + utils.print_iterable(result, max_digits=self.max_digits))
		print('-------------------------------------------------------------------')
		self._write_progress('test', result)

	def _save_net_infos(self, latest=False):
		save_epoch = self.epoch
		if latest is True:
			save_epoch = -1
		self._save_net(save_epoch)
		self._save_filters(save_epoch)

	def _save_filters(self, save_epoch):
		dict_filters = self.net.get_filters()
		for name, filters in dict_filters.items():
			name = '%s_%d.png' %(name, save_epoch)
			path = os.path.join(self.results_dir, name)
			if len(filters.shape) == 3:
				plt.imsave(path, filters)
			else:
				plt.imsave(path, filters, cmap='gray')
			plt.close() 

	def _save_net(self, save_epoch):
		path = os.path.join(self.results_dir, 'net_%i.pkl' %(save_epoch))
		torch.save(self.net.state_dict(), path)
		path_info = os.path.join(self.results_dir, 'net_info.pkl')
		utils.pickle_dump(self.net.get_net_info(), path_info)

	def _load_net(self, load_epoch, strict=True):
		path = os.path.join(self.results_dir, 'net_%i.pkl' %(load_epoch))
		print('loading net from: ', path)
		new_sd = torch.load(path)
		if strict:
			self.net.load_state_dict(new_sd)		
		else:
			utils.load_sd(self.net, new_sd)

	def _reconfigure_dataloader_tracker_train(self):
		self._reconfigure_tracker_train()
		self._reconfigure_dataloader_train()

	def _reconfigure_dataloader_tracker_test(self):
		self._reconfigure_tracker_test()
		self._reconfigure_dataloader_test()

	def _decay_learning_rate(self):
		scheme_0 = []
		scheme_1 = [(3/4)]
		scheme_2 = [(5/8), (7/8)]
		schemes = [scheme_0, scheme_1, scheme_2]
		for frac in schemes[self.lr_decay_scheme]:
			if self.epoch == int(self.epochs * frac):
				self.learning_rate *= 0.1
				utils.adjust_learning_rate(self.optimizer, self.learning_rate)	
				print('decay lr to %f' %self.learning_rate)

	def _apply_per_epoch(self):
		self._decay_learning_rate()

	def _track_per_iteration(self, output, labels, loss):
		pass

	def _forward(self, data):
		return self.net(*data)

	def _interrupt_training(self, loss_cum):
		return False

	# This func puts data fields into var and cuda 
	# and returns it together with labels, if iterator is at end, return None
	def _get_data(self, iterator):
		raise NotImplementedError('_get_data in Base_experiment not implemented')	

	def _get_loss(self, output, labels):
		raise NotImplementedError('_get_loss() in Base_experiment not implemented')

	# should make use of _reconfigure_dataloader()
	def _reconfigure_dataloader_train(self):
		raise NotImplementedError('_reconfigure_dataloader_train() in Base_experiment not implemented')

	# should make use of _reconfigure_dataloader()
	def _reconfigure_dataloader_test(self):
		raise NotImplementedError('_reconfigure_dataloader_test() in Base_experiment not implemented')		

	def _reconfigure_tracker_train(self):
		raise NotImplementedError('_reconfigure_dataloader_train() in Base_experiment not implemented')

	def _reconfigure_tracker_test(self):
		raise NotImplementedError('_reconfigure_dataloader_test() in Base_experiment not implemented')	

	@property
	def net(self):
		raise NotImplementedError('Base_experiment should implement net (VGG or Siam*)')

	@property
	def tracker(self):
		raise NotImplementedError('Base_experiment should implement tracker (Tracker)')

	@property
	def optimizer(self):
		raise NotImplementedError('Base_experiment should implement optimizer (torch.optim.*)')

	@property
	def save_intervall(self):
		raise NotImplementedError('Base_experiment should implement save_intervall (int)')

	@property
	def test_intervall(self):
		raise NotImplementedError('Base_experiment should implement test_intervall (int)')

	@property
	def batch_size_test(self):
		raise NotImplementedError('Base_experiment should implement batch_size_test (int)')

class Base_experiment_pretraining(Base_experiment):
	save_intervall = None
	test_intervall = None
	dataloader = None
	batch_size_test = None
	def __init__(self,
			name,
			batch_size,
			epochs = 200,
			learning_rate = 0.01,
			lr_decay_scheme = 0,
			weight_decay = 0.0005,
			data_key = 'all',
			norm = 'BN',
			split_channels = False,
			dropout = 0.5,
		):
		super(Base_experiment_pretraining, self).__init__(name=name, batch_size=batch_size, epochs=epochs, 
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay, 
			data_key=data_key)

		self.results_dir = os.path.join(self.results_dir, 'experiment')
		if not utils.mkdir(self.results_dir):
			print('Existing experiment for given name')

		self.norm = norm
		self.split_channels = split_channels
		self.dropout = dropout
		self.list_infos += [('norm', norm), ('split_channels', split_channels), 
			('dropout', dropout)]
		self.save_intervall = 50
		self.test_intervall = 10
		self.batch_size_test = int(batch_size / 2)
		if self.split_channels:
			self.transform_color = transforms.SplitChannels()
		else:
			self.transform_color = transforms.Smoothen(kernel_size=0)
	
	def _set_dataloader(self, dataset):
		self.dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=20, 
			shuffle=True, drop_last=True)	

	@property
	def dataset_type(self):
		raise NotImplementedError('Base_experiment_pretraining should implement dataset_type (type)')

class Base_experiment_finetuning(Base_experiment):
	save_intervall = None
	test_intervall = None
	batch_size_test = None
	def __init__(self,
			name,
			batch_size = 128,
			epochs = 200,
			learning_rate = 0.01,
			lr_decay_scheme = 1,
			weight_decay = 0.0005,
			data_key = 'ucf',
			dropout = 0.5,
			load_epoch_pt = -1,
			name_finetuning = None,
			name_experiment = None,
			reset_fc7 = False,
			reset_fc6 = False,
			freeze_layer = 'input', 
			split = 1
		):
		super(Base_experiment_finetuning, self).__init__(name=name, batch_size=batch_size, epochs=epochs, 
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay, 
			data_key=data_key)
		
		self.results_dir = os.path.join(self.results_dir, name_finetuning)
		if not utils.mkdir(self.results_dir):
			print('Existing finetuning for given name')

		self.dropout = dropout
		self.load_epoch_pt = load_epoch_pt
		self.name_finetuning = name_finetuning
		if name_experiment is None:
			name_experiment = 'experiment'
		self.name_experiment = name_experiment
		self.reset_fc7 = reset_fc7
		self.reset_fc6 = reset_fc6
		self.freeze_layer = freeze_layer
		self.split = split
		self.list_infos += [('dropout', dropout), ('load_epoch_pt', load_epoch_pt), 
			('name_finetuning', name_finetuning), ('name_experiment', name_experiment), 
			('reset_fc7', reset_fc7), ('freeze_layer', freeze_layer), ('split', split),
			('reset_fc6', reset_fc6)]
		self.save_intervall = 50
		# Currently there is a massive bug in which training loss resets after testing the first time
		self.test_intervall = self.epochs
		self.batch_size_test = 1
		self._load_pretraining()

	def evaluate_net(self, num_test=5, load_epoch=-1, final_test_runs=5, split_batch=1):
		self._load_net(load_epoch)
		t0_tot = time()
		self.net.cuda()
		self._reconfigure_dataloader_tracker_train()
		num_test_before = self.num_test
		self.num_test = num_test
		self._print_infos()	
		print('num_test: %d' %self.num_test)
		self.epoch = 0
		for _ in range(load_epoch):
			self._apply_per_epoch()
			self.epoch += 1
		for _ in range(final_test_runs-1):
			self.epoch += 1
			# self._training()
			self.epoch += 1
			# self._training()
			self._reconfigure_dataloader_tracker_test()
			self._testing(split_batch=split_batch)
			self._reconfigure_dataloader_tracker_train()	
		self.net.cpu()
		t1_tot = time()
		self.num_test = num_test_before
		print('total runtime evaluate_net: %f' %(t1_tot-t0_tot))		

	def _load_pretraining(self):
		results_dir_pt = os.path.join(self.results_path, self.name, self.name_experiment)
		path_info = os.path.join(results_dir_pt, 'net_info.pkl')
		net_pt = get_network(path_info)
		if self.load_epoch_pt != -2:
			path_params = os.path.join(results_dir_pt, 'net_%i.pkl' %(self.load_epoch_pt))
			new_sd = torch.load(path_params)
			utils.load_sd(net_pt, new_sd)
		feature_net = self._get_pretrained_subnet(net_pt)
		if self.reset_fc7:
			print('reset_fc7')
			feature_net.reset_fc7()
		if self.reset_fc6:
			print('reset_fc6')
			feature_net.reset_fc6()
		self.net = Net_ar(feature_net, dropout=self.dropout, data_key=self.data_key)	
		self.net.freeze_layers(self.freeze_layer)

	def _set_dataloader(self, dataset):
		self.dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=20, 
			shuffle=True, drop_last=False)	

	# Choose the right subnet in net_pt for the finetuning procedure
	def _get_pretrained_subnet(self, net_pt):
		raise NotImplementedError('_get_pretrained_subnet() in Base_experiment_finetuning not implemented')

