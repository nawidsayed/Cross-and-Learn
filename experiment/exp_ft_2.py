import _pickle as pickle
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from experiment import utils
from experiment import Base_experiment_finetuning
from experiment.tracker import Tracker_classification
from compvis import transforms_det as transforms 
from compvis.datasets import Dataset_RGB, Dataset_OF, Dataset_COD
from compvis.models import get_network, Net_ar, Two_Stream, get_arch
from time import time


__all__ = ['Experiment_finetuning_cluster', 'Experiment_finetuning_twostream']


class Experiment_finetuning_cluster(Base_experiment_finetuning):
	net = None
	tracker = None
	dataloader = None
	optimizer = None
	def __init__(self,
			name,
			name_cluster,
			name_labels = None,
			batch_size = 128,
			epochs = 200,
			learning_rate = 0.01,
			lr_decay_scheme = 1,
			weight_decay = 0.0005,
			data_key = 'all',
			source = 'l',
			dropout = 0.5,
			load_epoch_pt = -1,
			name_finetuning = None,
			reset_fc7 = False,
			reset_fc6 = False,
			freeze_layer = 'input',
			rgb = 0.3,
			num_test = -30,
			num_frames = 12,
			num_clusters = 1000,
			condense = False,
			norm = None
		):
		self.num_clusters = num_clusters
		self.norm = norm
		super(Experiment_finetuning_cluster, self).__init__(name=name,batch_size=batch_size,epochs=epochs, 
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay, 
			data_key=data_key, source=source, dropout=dropout, name_finetuning=name_finetuning, 
			reset_fc7=reset_fc7, load_epoch_pt=load_epoch_pt,freeze_layer=freeze_layer,reset_fc6=reset_fc6)
		if condense:
			num_test = 1
		self.rgb = rgb
		self.num_test = num_test
		self.num_frames = num_frames
		self.name_cluster = name_cluster
		if name_labels is None:
			name_labels = 'labels_old'
		self.name_labels = name_labels
		self.list_infos += [('rgb', rgb), ('num_test', num_test), ('num_frames', num_frames), 
			('name_cluster', name_cluster), ('name_labels', name_labels),
			('num_clusters', num_clusters), ('norm', norm)]
		self.dataset_type = Dataset_RGB
		self.path_labels = os.path.join(self.results_path, self.name, self.name_cluster, 
			'%s.pkl' %self.name_labels)
		self.path_labels_test = os.path.join(self.results_path, self.name, self.name_cluster, 
			'%s_test.pkl' %self.name_labels)
		self.tracker = Tracker_classification()
		self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), 
			lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
		self.criterion = nn.CrossEntropyLoss()
		self.num_test_frames = 5

	def evaluate_net(self, num_test=5, load_epoch=-1, final_test_runs=5, split_batch=1):
		self.num_test_frames = num_test
		super(Experiment_finetuning_cluster, self).evaluate_net(num_test=num_test,
			load_epoch=load_epoch, final_test_runs=final_test_runs, split_batch=split_batch)

	def _get_data(self, iterator):
		images, labels = next(iterator, (None, None))
		if images is None:
			return None, None
		images = torch.cat(images, dim=0)
		labels = torch.cat(labels, dim=0)	
		images, labels = Variable(images).cuda(), Variable(labels).cuda()
		return [images], labels

	def _get_loss(self, output, labels):
		return self.criterion(output, labels)

	def _reconfigure_dataloader_train(self):
		if not self.name_labels == 'labels_old':
			labels = pickle.load(open(self.path_labels, 'rb'))
			if len(labels) != len(self.dataset_info_types):
				print(len(labels), len(self.dataset_info_types))
				raise Exception('labels and dataset_info types have different length')
		rgb = self.rgb
		transform = transforms.Compose([
			transforms.Scale(256), 
			transforms.SplitChannels(use_rand=False),
			transforms.RandomCrop(self.net.input_spatial_size),
			transforms.RandomHorizontalFlip(), 
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		dataset_infos = []
		for i, dataset_info_type in enumerate(self.dataset_info_types):
			dataset_info = dataset_info_type(train=True, source=self.source, num_frames=self.num_frames)
			if not self.name_labels == 'labels_old':
				dataset_info.set_labels(labels[i], num_test=self.num_test)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=True, transform=transform)
		self._reconfigure_dataloader(dataset, self.batch_size, shuffle=True)

	def _reconfigure_dataloader_test(self):
		if not self.name_labels == 'labels_old':
			labels = pickle.load(open(self.path_labels_test, 'rb'))
			if len(labels) != len(self.dataset_info_types):
				print(len(labels), len(self.dataset_info_types))
				raise Exception('labels and dataset_info types have different length')
		rgb = self.rgb
		transform = transforms.Compose([
			transforms.Scale(256), 
			transforms.SplitChannels(use_rand=False, train=False),
			transforms.TenCrop(self.net.input_spatial_size),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		dataset_infos = []
		for i, dataset_info_type in enumerate(self.dataset_info_types):
			dataset_info = dataset_info_type(train=False, source=self.source, num_frames=self.num_frames)
			if not self.name_labels == 'labels_old':
				dataset_info.set_labels(labels[i], num_test=self.num_test)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=False, transform=transform, 
			num_test=self.num_test_frames)
		self._reconfigure_dataloader(dataset, 1, shuffle=False)

	def _reconfigure_tracker_train(self):
		self.tracker = Tracker_classification()

	def _reconfigure_tracker_test(self):
		self.tracker = Tracker_classification(mode='multi_frame')

	def _get_pretrained_subnet(self, net_pt):
		if hasattr(net_pt, 'app_net'):
			return net_pt.app_net
		else:
			return net_pt

	def _check_index_list(self, dataset):
		if not hasattr(self, 'index_list'):
			length = len(dataset)
			self.index_list = np.arange(length).astype(int)
			np.random.shuffle(self.index_list)
			self.train_test_split = int(length * 0.8)
		if len(dataset) != len(self.index_list):
			raise Exception('dataset and index_list missmatch: %d, %d'%(len(dataset),len(self.index_list)))

	def _load_pretraining(self):
		if self.norm is None:
			results_dir_pt = os.path.join(self.results_path, self.name, 'experiment')
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
		else:
			arch = get_arch(self.norm)
			groups = 1
			if 'g2' in self.norm:
				groups = 2
			feature_net = arch(input_dim=3, groups=groups)
		self.net = Net_ar(feature_net, dropout=self.dropout, data_key=self.num_clusters)	

class Experiment_finetuning_twostream(Base_experiment_finetuning):
	net = None
	tracker = None
	dataloader = None
	optimizer = None
	def __init__(self,
			name,
			name_1,
			name_2,
			name_finetuning,
			name_finetuning_1,
			name_finetuning_2,
			name_experiment = None,
			load_epoch_ft_1 = -1,
			load_epoch_ft_2 = -1,
			batch_size = 128,
			epochs = 1000,
			learning_rate = 0.01,
			lr_decay_scheme = 1,
			weight_decay = 0.0005,
			data_key = 'ucf',
			source = 'l',
			dropout = 0.5,
			load_epoch_pt = -1,
			reset_fc7 = False,
			freeze_layer = 'input',
			rgb = 0.3,
			num_test = 5,
			split = 1,
			fusion_scheme = 'avg'
		):
		super(Experiment_finetuning_twostream, self).__init__(name=name, batch_size=batch_size, 
			epochs=epochs, 
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay, 
			data_key=data_key, source=source, dropout=dropout, name_finetuning=name_finetuning, 
			name_experiment=name_experiment, reset_fc7=reset_fc7, load_epoch_pt=load_epoch_pt, 
			freeze_layer=freeze_layer, split=split)
		self.name_1 = name_1
		self.name_2 = name_2
		self.name_finetuning_1 = name_finetuning_1
		self.name_finetuning_2 = name_finetuning_2
		self.load_epoch_ft_1 = load_epoch_ft_1
		self.load_epoch_ft_2 = load_epoch_ft_2
		self.rgb = rgb
		self.num_test = num_test
		self.fusion_scheme = fusion_scheme
		self.list_infos += [('rgb', rgb), ('num_test', num_test), ('name_1', name_1), ('name_2', name_2), 
			('name_finetuning_1', name_finetuning_1), ('name_finetuning_2', name_finetuning_2),
			('load_epoch_ft_1', load_epoch_ft_1), ('load_epoch_ft_2', load_epoch_ft_2), 
			('fusion_scheme', fusion_scheme)]
		self.dataset_type = Dataset_RGB
		self.tracker = Tracker_classification()
		self._load_models()
		self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), 
			lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
		self.criterion = nn.CrossEntropyLoss()

		

	def run(self):
		raise Exception('Cant run this')


	def evaluate_net(self, num_test=5, load_epoch=200, final_test_runs=5):
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
			list_ir = self._evaluating()
			self._reconfigure_dataloader_tracker_train()	
			path_list_ir = os.path.join(self.results_dir, 'list_ir.pkl')
		self.net.cpu()
		t1_tot = time()
		self.num_test = num_test_before
		print('total runtime evaluate_net: %f' %(t1_tot-t0_tot))	

	def _evaluating(self, split_batch=1):
		t0 = time()
		self.net.train(mode=False)
		iterator = iter(self.dataloader)
		while(True):
			data, labels = self._get_data(iterator)
			if data is None:
				break
			output, out_1, out_2 = self._forward(data)
			self.tracker.update(output, labels)
			self.tracker_1.update(out_1, labels)
			self.tracker_2.update(out_2, labels)
		result = self.tracker.result()[1:]
		result_1 = self.tracker_1.result()[1:]
		result_2 = self.tracker_2.result()[1:]
		list_ir, list_pred = self.tracker.list_individual_results()
		list_ir_1, list_pred_1 = self.tracker_1.list_individual_results()
		list_ir_2, list_pred_2 = self.tracker_2.list_individual_results()
		t1 = time()
		runtime = t1-t0
		result = [('epoch', self.epoch)] + result + result_1 + result_2
		print('-------------------------------------------------------------------')
		print('test ' + utils.print_iterable(result, max_digits=self.max_digits))
		print('-------------------------------------------------------------------')
		self._write_progress('eval', result)
		self._evaluate_list_irs(list_ir, list_ir_1, list_ir_2)
		self._evaluate_list_preds(list_pred, list_pred_1, list_pred_2)

	def _get_data(self, iterator):
		images, labels = next(iterator, (None, None))
		if images is None:
			return None, None
		images = torch.cat(images, dim=0)
		labels = torch.cat(labels, dim=0)	
		images, labels = Variable(images).cuda(), Variable(labels).cuda()
		return [images], labels

	#	This code is for testing the speed of the network 
	# def _get_data(self, iterator):
	# 	if not hasattr(self, 'batch_data'):
	# 		self.num_images_epoch = 0
	# 		images, labels = next(iterator, (None, None))
	# 		images = torch.cat(images, dim=0)
	# 		labels = torch.cat(labels, dim=0)	
	# 		self.batch_data = Variable(images).cuda(), Variable(labels).cuda()
	# 	self.num_images_epoch += 1
	# 	if self.num_images_epoch > len(self.dataloader):
	# 		self.num_images_epoch = 0
	# 		return None, None
	# 	images, labels = self.batch_data
	# 	return [images], labels

	def _evaluate_list_irs(self, list_ir, list_ir_1, list_ir_2):
		length = len(list_ir)
		acc = sum(list_ir) / length
		acc_1 = sum(list_ir_1) / length
		acc_2 = sum(list_ir_2) / length
		union_rand = 1 - (1-acc_1) * (1-acc_2)
		inter_rand = acc_1 * acc_2
		union = 0
		inter = 0
		for i in range(length):
			if list_ir_1[i] == 1 and list_ir_2[i] == 1:
				inter += 1
			if list_ir_1[i] == 1 or list_ir_2[i] == 1:
				union += 1
		union /= length
		inter /= length
		accurcies = [('acc', acc), ('acc_1', acc_1), ('acc_2', acc_2)]
		int_uni = [('inter', inter),('union', union), ('iou', inter/union)]
		print(utils.print_iterable(accurcies, max_digits=self.max_digits))
		print(utils.print_iterable(int_uni, max_digits=self.max_digits))
		self._write_progress('eval', accurcies)
		self._write_progress('eval', int_uni)

	def _evaluate_list_preds(self, list_ir, list_ir_1, list_ir_2):
		length = len(list_ir)
		inter = 0
		for i in range(length):
			if list_ir_1[i] == list_ir_2[i]:
				inter += 1

		inter /= length
		int_uni = [('inter_pred', inter)]
		print(utils.print_iterable(int_uni, max_digits=self.max_digits))
		self._write_progress('eval', int_uni)


	def _get_loss(self, output, labels):
		return self.criterion(output, labels)

	def _reconfigure_dataloader_train(self):
		rgb = self.rgb
		transform = transforms.Compose([
			transforms.Scale(256), 
			transforms.RandomCrop(self.net.input_spatial_size),
			transforms.RandomHorizontalFlip(), 
			transforms.RandomColorJitter(brightness=rgb, contrast=rgb, saturation=rgb, hue=rgb),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=True, source=self.source, num_frames=1, min_msd=0,
				split=self.split)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=True, transform=transform)
		self._reconfigure_dataloader(dataset, self.batch_size, shuffle=True)

	def _reconfigure_dataloader_test(self):
		transform = transforms.Compose([
			transforms.Scale(256), 
			transforms.TenCrop(self.net.input_spatial_size),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, source=self.source, num_frames=1, min_msd=0,
				split=self.split)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=False, transform=transform, 
			num_test=self.num_test)
		self._reconfigure_dataloader(dataset, self.batch_size_test, shuffle=False)

	def _load_pretraining(self):
		pass

	def _load_net(self, epoch):
		pass

	def _load_models(self, strict=True):
		result_dirs = []
		result_dirs.append(os.path.join(self.results_path, self.name_1, self.name_finetuning_1))
		result_dirs.append(os.path.join(self.results_path, self.name_2, self.name_finetuning_2))
		load_epoch_fts = [self.load_epoch_ft_1, self.load_epoch_ft_2]
		nets = []
		for i in range(2):
			path_info = os.path.join(result_dirs[i], 'net_info.pkl')
			net_ft = get_network(path_info)
			if load_epoch_fts[i] != -2:
				path_params = os.path.join(result_dirs[i], 'net_%i.pkl' %(load_epoch_fts[i]))
				new_sd = torch.load(path_params)
				if strict:
					net_ft.load_state_dict(new_sd)		
				else:
					utils.load_sd(net_ft, new_sd)
			nets.append(net_ft)
		self.net = Two_Stream(nets[0], nets[1], dropout=self.dropout)

	def _reconfigure_tracker_train(self):
		self.tracker = Tracker_classification()
		self.tracker.track_individual_results()
		self.tracker_1 = Tracker_classification()
		self.tracker_1.track_individual_results()
		self.tracker_2 = Tracker_classification()
		self.tracker_2.track_individual_results()

	def _reconfigure_tracker_test(self):
		self.tracker = Tracker_classification(mode='multi_frame')
		self.tracker.track_individual_results()
		self.tracker_1 = Tracker_classification(mode='multi_frame')
		self.tracker_1.track_individual_results()
		self.tracker_2 = Tracker_classification(mode='multi_frame')
		self.tracker_2.track_individual_results()

	def _forward(self, data):
		out_1, out_2 = self.net(*data)
		if self.fusion_scheme == 'avg':
			output = (out_1 + out_2) / 2
		elif self.fusion_scheme == 'avg0.9':
			output = out_1 * 0.9 + out_2 * 0.1
		elif self.fusion_scheme == 'avg0.8':
			output = out_1 * 0.8 + out_2 * 0.2
		return output, out_1, out_2
		# elif self.fusion_scheme == 'mc':
		# 	softmax_1 = self.softmax(out_1)
		# 	softmax_2 = self.softmax(out_2)
		# 	pred_1, predind_1 = torch.max(softmax_1.data, 1)
		# 	pred_2, predind_2 = torch.max(softmax_2.data, 1)
		# 	t = torch.Tensor(out_1.data.size()).zero_().cuda()
		# 	out = Variable(t)
		# 	for i in range(out.size(0)):
		# 	   if pred_1[i] > pred_2[i]:
		# 	       out[i] = out_1[i]
		# 	   else:
		# 	       out[i] = out_2[i]
		# 	return out