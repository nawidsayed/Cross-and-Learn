import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from experiment import utils
from experiment import Base_experiment_finetuning
from experiment.tracker import Tracker_classification
from compvis import transforms_det as transforms 
from compvis.datasets import Dataset_RGB, Dataset_OF, Dataset_COD
from time import time


__all__ = ['Experiment_finetuning_ar_RGB','Experiment_finetuning_ar_OF','Experiment_finetuning_ar_COD',
	'Experiment_finetuning_ar_multi']

class Experiment_finetuning_ar_RGB(Base_experiment_finetuning):
	net = None
	tracker = None
	dataloader = None
	optimizer = None
	def __init__(self,
			name,
			batch_size = 128,
			epochs = 1000,
			learning_rate = 0.01,
			lr_decay_scheme = 1,
			weight_decay = 0.0005,
			data_key = 'ucf',
			source = 'l',
			dropout = 0.5,
			load_epoch_pt = -1,
			name_finetuning = None,
			name_experiment = None,
			reset_fc7 = False,
			reset_fc6 = False,
			freeze_layer = 'input',
			rgb = 0.3,
			num_test = 5,
			split = 1
		):
		super(Experiment_finetuning_ar_RGB, self).__init__(name=name, batch_size=batch_size, epochs=epochs, 
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay, 
			data_key=data_key, source=source, dropout=dropout, name_finetuning=name_finetuning, 
			name_experiment=name_experiment, reset_fc7=reset_fc7, load_epoch_pt=load_epoch_pt,
			freeze_layer=freeze_layer, split=split, reset_fc6=reset_fc6)

		self.rgb = rgb
		self.num_test = num_test
		self.list_infos += [('rgb', rgb), ('num_test', num_test)]
		self.dataset_type = Dataset_RGB
		self.tracker = Tracker_classification()
		self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), 
			lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)

		self.criterion = nn.CrossEntropyLoss()

	def run(self, resume_training=0):
		super(Experiment_finetuning_ar_RGB, self).run(final_test_runs=1, resume_training=resume_training)


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
		print(len(dataset))
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

	def _reconfigure_tracker_train(self):
		self.tracker = Tracker_classification()

	def _reconfigure_tracker_test(self):
		self.tracker = Tracker_classification(mode='multi_frame')

	def _get_pretrained_subnet(self, net_pt):
		if hasattr(net_pt, 'app_net'):
			return net_pt.app_net
		elif hasattr(net_pt, 'feature_net'):
			return net_pt.feature_net
		else:
			return net_pt

class Experiment_finetuning_ar_OF(Base_experiment_finetuning):
	net = None
	tracker = None
	dataloader = None
	optimizer = None
	def __init__(self,
			name,
			batch_size = 128,
			epochs = 1000,
			learning_rate = 0.01,
			lr_decay_scheme = 1,
			weight_decay = 0.0005,
			data_key = 'ucf',
			source = 'l',
			dropout = 0.5,
			load_epoch_pt = -1,
			name_finetuning = None,
			reset_fc7 = False,
			reset_fc6 = False,
			freeze_layer = 'input',
			num_test = 5,
			time_flip = False,
			split = 1, 
			remove_motion = False
		):
		super(Experiment_finetuning_ar_OF, self).__init__(name=name, batch_size=batch_size, epochs=epochs, 
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay, 
			data_key=data_key, source=source, dropout=dropout, name_finetuning=name_finetuning, 
			reset_fc7=reset_fc7, load_epoch_pt=load_epoch_pt, freeze_layer=freeze_layer, split=split, 
			reset_fc6=reset_fc6)

		self.num_test = num_test
		self.num_frames = int(self.net.input_dim / 2)
		self.time_flip = time_flip
		self.remove_motion = remove_motion
		self.list_infos += [('num_test', num_test), ('num_frames', self.num_frames), 
			('time_flip', time_flip), ('remove_motion', remove_motion)]
		self.dataset_type = Dataset_OF
		self.tracker = Tracker_classification()
		self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), 
			lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
		self.criterion = nn.CrossEntropyLoss()

	def run(self):
		super(Experiment_finetuning_ar_OF, self).run(final_test_runs=1)

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
		transform = transforms.Compose([
			transforms.Scale(256), 
			transforms.RandomCrop(self.net.input_spatial_size),
			transforms.RandomHorizontalFlip(), 
			transforms.ToTensor(),
			transforms.SubMeanDisplacement()])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=True, source=self.source, num_frames=self.num_frames, 
				min_msd=0, split=self.split)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=True, transform=transform,
			num_frames=self.num_frames, time_flip=self.time_flip, remove_motion=self.remove_motion)
		self._reconfigure_dataloader(dataset, self.batch_size, shuffle=True)

	def _reconfigure_dataloader_test(self):
		transform = transforms.Compose([
			transforms.Scale(256), 
			transforms.TenCrop(self.net.input_spatial_size),
			transforms.ToTensor(),
			transforms.SubMeanDisplacement()])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, source=self.source, num_frames=self.num_frames, 
				min_msd=0, split=self.split)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=False, transform=transform, 
			num_test=self.num_test, num_frames=self.num_frames, remove_motion=self.remove_motion)
		self._reconfigure_dataloader(dataset, self.batch_size_test, shuffle=False)

	def _reconfigure_tracker_train(self):
		self.tracker = Tracker_classification()

	def _reconfigure_tracker_test(self):
		self.tracker = Tracker_classification(mode='multi_frame')

	def _get_pretrained_subnet(self, net_pt):
		if hasattr(net_pt, 'mot_net'):
			return net_pt.mot_net
		else:
			return net_pt

class Experiment_finetuning_ar_COD(Base_experiment_finetuning):
	net = None
	tracker = None
	dataloader = None
	optimizer = None
	def __init__(self,
			name,
			batch_size = 128,
			epochs = 1000,
			learning_rate = 0.01,
			lr_decay_scheme = 1,
			weight_decay = 0.0005,
			data_key = 'ucf',
			source = 'l',
			dropout = 0.5,
			load_epoch_pt = -1,
			name_finetuning = None,
			reset_fc7 = False,
			reset_fc6 = False,
			freeze_layer = 'input',
			num_test = 5,
			nodiff = False,
			time_flip = False,
			split = 1
		):
		super(Experiment_finetuning_ar_COD, self).__init__(name=name, batch_size=batch_size, epochs=epochs, 
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay, 
			data_key=data_key, source=source, dropout=dropout, name_finetuning=name_finetuning, 
			reset_fc7=reset_fc7, load_epoch_pt=load_epoch_pt, freeze_layer=freeze_layer, split=split,
			reset_fc6=reset_fc6)

		self.num_test = num_test
		self.num_frames = int(self.net.input_dim / 3)
		self.nodiff = nodiff
		self.time_flip = time_flip
		self.list_infos += [('num_test', num_test), ('num_frames', self.num_frames), ('nodiff', nodiff),
			('time_flip', time_flip)]
		self.dataset_type = Dataset_COD
		self.tracker = Tracker_classification()
		self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), 
			lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)

		self.criterion = nn.CrossEntropyLoss()

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
		transform = transforms.Compose([
			transforms.Scale(256), 
			transforms.RandomCrop(self.net.input_spatial_size),
			transforms.RandomHorizontalFlip(), 
			transforms.ToTensor()])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=True, source=self.source, num_frames=self.num_frames, 
				min_msd=0, split=self.split)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=True, transform=transform,
			num_frames=self.num_frames, nodiff=self.nodiff, time_flip=self.time_flip)
		self._reconfigure_dataloader(dataset, self.batch_size, shuffle=True)

	def _reconfigure_dataloader_test(self):
		transform = transforms.Compose([
			transforms.Scale(256), 
			transforms.TenCrop(self.net.input_spatial_size),
			transforms.ToTensor()])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, source=self.source, num_frames=self.num_frames, 
				min_msd=0, split=self.split)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=False, transform=transform, 
			num_test=self.num_test, num_frames=self.num_frames, nodiff=self.nodiff)
		self._reconfigure_dataloader(dataset, self.batch_size_test, shuffle=False)

	def _reconfigure_tracker_train(self):
		self.tracker = Tracker_classification()

	def _reconfigure_tracker_test(self):
		self.tracker = Tracker_classification(mode='multi_frame')

	def _get_pretrained_subnet(self, net_pt):
		if hasattr(net_pt, 'cod_net'):
			return net_pt.cod_net
		else:
			return net_pt

class Experiment_finetuning_ar_multi(Base_experiment_finetuning):
	net = None
	tracker = None
	dataloader = None
	optimizer = None
	def __init__(self,
			name,
			batch_size = 128,
			epochs = 500,
			learning_rate = 0.01,
			lr_decay_scheme = 1,
			weight_decay = 0.0005,
			data_key = 'all',
			source = 'l',
			dropout = 0.5,
			load_epoch_pt = -1,
			name_finetuning = None,
			name_experiment = None,
			reset_fc7 = False,
			freeze_layer = 'input',
			rgb = 0.3,
			num_test = 5,
			split = 1
		):
		super(Experiment_finetuning_ar_multi, self).__init__(name=name,batch_size=batch_size,epochs=epochs, 
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay, 
			data_key=data_key, source=source, dropout=dropout, name_finetuning=name_finetuning, 
			name_experiment=name_experiment, reset_fc7=reset_fc7, load_epoch_pt=load_epoch_pt, 
			freeze_layer=freeze_layer, split=split)
		if data_key != 'all':
			raise Exception('Multitask is only for all implemented')
		self.slices = [0,101,152,195]
		print(self.slices)
		self.rgb = rgb
		self.num_test = num_test
		self.list_infos += [('rgb', rgb), ('num_test', num_test)]
		self.dataset_type = Dataset_RGB
		self.tracker = Tracker_classification()
		self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), 
			lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)

		self.criterion = nn.CrossEntropyLoss()

	def run(self, resume_training=0):
		super(Experiment_finetuning_ar_multi, self).run(final_test_runs=1, resume_training=resume_training)


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

	def _get_loss(self, output, labels):
		length = output.size(0)
		loss = 0
		for i in range(length):
			k = int(labels[i] / 100000)
			lower = self.slices[k]
			upper = self.slices[k+1]
			out = output[i:i+1,lower:upper]
			lab = labels[i:i+1] - k*100000
			loss += self.criterion(out, lab)
		return loss / length
		# return self.criterion(output, labels)
		
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

	def _reconfigure_tracker_train(self):
		self.tracker = Tracker_classification(only_loss=True)

	def _reconfigure_tracker_test(self):
		self.tracker = Tracker_classification(mode='multi_frame', only_loss=True)

	def _get_pretrained_subnet(self, net_pt):
		if hasattr(net_pt, 'app_net'):
			return net_pt.app_net
		elif hasattr(net_pt, 'feature_net'):
			return net_pt.feature_net
		else:
			return net_pt









if __name__ == "__main__":
	e = Experiment_finetuning_ar_RGB('test_ft', batch_size=20, source='l', 
		load_epoch_pt=400, name_finetuning='test_def')
	e.run()
