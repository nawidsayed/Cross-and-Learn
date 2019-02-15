import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from experiment import utils
from experiment import Base_experiment_pretraining
from experiment.tracker import Tracker_classification, Tracker_similarity, Tracker_similarity_rec
from compvis.models import Concat, Cross_and_Learn
from compvis.datasets import Dataset_def, Dataset_fm, Dataset_RGB, Dataset_OF

from compvis import transforms_det as transforms 

__all__ = ['Pretraining_Concat', 'Pretraining_Cross_and_Learn']

class Pretraining_Concat(Base_experiment_pretraining):
	net = None
	tracker = None
	optimizer = None
	dataset_type = None
	def __init__(self,
			name,
			batch_size = 30,
			epochs = 200,
			learning_rate = 0.01,
			lr_decay_scheme = 0,
			weight_decay = 0.0005,
			norm = 'BN',
			data_key = 'all',
			modalities = ['rgb', 'of'],
			num_frames = 10,
			num_frames_cod = 4,
			split_channels = False,
			dropout = 0.5,
			high_motion = 1,
			time_flip = True
		):
		super(Pretraining_Concat, self).__init__(name=name, batch_size=batch_size, epochs=epochs, 
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay, 
			norm=norm, data_key=data_key, split_channels=split_channels, 
			dropout=dropout)
		self.num_frames = num_frames
		self.num_frames_cod = num_frames_cod
		self.high_motion = high_motion	
		self.time_flip = time_flip
		self.modalities = modalities
		self.list_infos += [('num_frames', num_frames), ('num_frames_cod', num_frames_cod),
			('high_motion', high_motion), ('time_flip', time_flip), ('modalities', modalities)]

		self.net = Concat(norm=self.norm, num_frames=self.num_frames, dropout=self.dropout,
			modalities=self.modalities)
		self.tracker = Tracker_classification()
		self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate,
			momentum=0.9, weight_decay=self.weight_decay)
		self.dataset_type = Dataset_fm
		self.criterion = nn.CrossEntropyLoss()

	def _get_data(self, iterator):
		samples = next(iterator, None)
		if samples is None:
			return None, None
		data = []
		for field in samples:
			field = torch.cat(field, dim=0).cuda()
			field = Variable(field)
			data.append(field)

		batch_size = data[0].size()[0]
		labels_zero_1 = Variable(torch.LongTensor(batch_size).zero_().cuda())
		labels_zero_2 = Variable(torch.LongTensor(batch_size).zero_().cuda())
		labels_one_1 = Variable(torch.LongTensor(batch_size).zero_().cuda() + 1)
		labels_one_2 = Variable(torch.LongTensor(batch_size).zero_().cuda() + 1)
		labels = [labels_one_1, labels_one_2, labels_zero_1, labels_zero_2]
		return data, labels

	def _get_loss(self, output, labels):
		norms = output[-1]
		output = output[:-1]
		loss = 0
		for i in range(4):
			loss += self.criterion(output[i], labels[i])
		loss /= 4
		return loss

	def _reconfigure_dataloader_train(self):
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			transforms.RandomCrop(self.net.input_spatial_size),
			transforms.RandomHorizontalFlip(), 
			self.transform_color,
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		transform_of = transforms.Compose([
			transforms.Scale(256), 
			transforms.RandomCrop(self.net.input_spatial_size),
			transforms.RandomHorizontalFlip(), 
			transforms.ToTensor(), 
			transforms.SubMeanDisplacement()])
		transform_cod = transforms.Compose([
			transforms.Scale(256), 
			transforms.RandomCrop(self.net.input_spatial_size),
			transforms.RandomHorizontalFlip(), 
			transforms.ToTensor()])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=True, num_frames=self.num_frames)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=True, transform_rgb=transform_rgb,
		  transform_of=transform_of, transform_cod=transform_cod, num_frames=self.num_frames, 
		  num_frames_cod=self.num_frames_cod, high_motion=self.high_motion, modalities=self.modalities,
		  time_flip=self.time_flip)
		self._reconfigure_dataloader(dataset, self.batch_size, shuffle=True)

	def _reconfigure_dataloader_test(self):
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(self.net.input_spatial_size),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		transform_of = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(self.net.input_spatial_size),
			transforms.ToTensor(), 
			transforms.SubMeanDisplacement()])
		transform_cod = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(self.net.input_spatial_size),
			transforms.ToTensor()])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, num_frames=self.num_frames)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=False, transform_rgb=transform_rgb,
		  transform_of=transform_of, transform_cod=transform_cod, num_frames=self.num_frames,
		  num_frames_cod=self.num_frames_cod, modalities=self.modalities)
		self._reconfigure_dataloader(dataset, self.batch_size_test, shuffle=True)

	def _reconfigure_tracker_train(self):
		self.tracker = Tracker_classification(with_nonzeros=True)

	def _reconfigure_tracker_test(self):
		self.tracker = Tracker_classification(with_nonzeros=True)

class Pretraining_Cross_and_Learn(Base_experiment_pretraining):
	test_epoch_zero = True
	net = None
	tracker = None
	optimizer = None
	dataset_type = None
	def __init__(self,
			name,
			batch_size = 45,
			epochs = 200,
			learning_rate = 0.01,
			lr_decay_scheme = 0,
			weight_decay = 0.0005,
			norm = 'BN',
			data_key = 'all',
			num_frames = 10,
			num_frames_cod = 4,
			split_channels = False,
			dropout = 0.5,
			layer = 'fc6',
			modalities = ['rgb', 'of'], 
			high_motion = False,
			time_flip = True,
			similarity_scheme = 'cosine',
			split = 1,
			weight_pos = 0.5,
			leaky_relu = False,
			eps = 0.001,
			gradient_dot = 'balanced'
		):
		super(Pretraining_Cross_and_Learn, self).__init__(name=name, batch_size=batch_size, epochs=epochs, 
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay, 
			norm=norm, data_key=data_key, split_channels=split_channels, 
			dropout=dropout)
		self.num_frames = num_frames
		self.num_frames_cod = num_frames_cod
		self.layer = layer
		self.modalities = modalities
		self.high_motion = high_motion
		self.time_flip = time_flip
		self.similarity_scheme = similarity_scheme
		self.split = split
		self.weight_pos = weight_pos
		self.leaky_relu = leaky_relu
		self.eps = eps
		self.gradient_dot = gradient_dot
		self.list_infos += [('num_frames', num_frames), ('num_frames_cod', num_frames_cod), 
			('layer', layer),  
			('modalities', modalities), ('high_motion', high_motion), 
			('time_flip', time_flip),('similarity_scheme', similarity_scheme),
			('split', split), ('weight_pos', weight_pos),
			('leaky_relu', leaky_relu), ('eps', eps), 
			('gradient_dot', gradient_dot)]
		self.net = Cross_and_Learn(norm=self.norm, num_frames=self.num_frames,num_frames_cod=self.num_frames_cod,
			dropout=self.dropout, layer=self.layer, modalities=self.modalities, 
			similarity_scheme=self.similarity_scheme, leaky_relu=leaky_relu, eps=self.eps)
		self.tracker = Tracker_similarity()
		self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate,
			momentum=0.9, weight_decay=self.weight_decay)
		self.dataset_type = Dataset_fm
		self.criterion_norm = nn.MSELoss()

	def _get_data(self, iterator):
		samples = next(iterator, None)
		if samples is None:
			return None, None
		data = []
		for field in samples:
			field = torch.cat(field, dim=0).cuda()
			field = Variable(field)
			data.append(field)
		return data, None
	
	# This code removes CPU bottleneck 
	# def _get_data(self, iterator):
	# 	if not hasattr(self, 'length_iterator'):
	# 		self.length_iterator = 0
	# 		print(len(iterator))
	# 	self.length_iterator += 1
	# 	if self.length_iterator > len(iterator):
	# 		self.length_iterator = 0
	# 		return None, None
	# 	data_3 = torch.Tensor(30,3,224,224).cuda()
	# 	data_20 = torch.Tensor(30,20,224,224).cuda()
	# 	var_3_1 = Variable(data_3)
	# 	var_3_2 = Variable(data_3)
	# 	var_20_1 = Variable(data_20)
	# 	var_20_2 = Variable(data_20)
	# 	data = [var_3_1, var_20_1, var_3_2, var_20_2]
	# 	return data, None

	def _get_loss(self, output, labels):
		norms = output[-1]
		output = output[:-1]
		loss_norm = 0
		half = int(len(output) / 2)
		sim_true = 0
		sim_false = 0
		for i in range(half):
			sim_true += self._distance_transformation(output[i], True)
			sim_false += self._distance_transformation(output[half + i], False)
		loss_sim = 2*torch.mean(sim_false*(1-self.weight_pos) - sim_true*self.weight_pos, dim=0) / half
		return loss_sim

	def _distance_transformation(self, sim, close):
		if self.gradient_dot == 'balanced':
			return sim
		if self.gradient_dot == 'euclidean':
			return 1 - torch.sqrt(2-2*sim + self.eps)
		if self.gradient_dot == 'high_close':
			return sim ** 2
		if self.gradient_dot == 'high_far':
			return 1-(1-sim)**2
		if self.gradient_dot == 'mixed':
			if close:
				return sim ** 2
			else:
				return 1-(1-sim)** 2
		if self.gradient_dot == 'mixed_anti':
			if not close:
				return sim ** 2
			else:
				return 1-(1-sim)** 2

	def _gen_feature_mat(self, mode):
		raise Exception('deprecated')
		print('generating feature_mat')
		self.net.cuda()
		self.net.train(mode=False)
		self._reconfigure_dataloader_fm(mode)
		iterator = iter(self.dataloader)
		feature_mat = []
		while(True):
			data, labels = self._get_data_fm(iterator)
			if data is None:
				break
			# import ipdb; ipdb.set_trace()
			output = self._get_subnet(mode).get_fc6_features(*data)
			output = output.view(output.size(0), -1).data
			output = output.mean(dim=0, keepdim=True)
			feature_mat.append(output)
		feature_mat = torch.cat(feature_mat, dim=0)
		self.net.cpu()

	def _gen_nn_mat(self, mode):
		feature_mat = self.gen_feature_mat(mode)
		trust_region = 100
		length = self.feature_mat.size()[0]
		nn_mat = torch.LongTensor(length, trust_region)
		for i in range(length):
			values, indices = utils.cos_sim(feature_mat[i:i+1], feature_mat).sort(descending=True)
			nn_mat[i] = indices[:trust_region].cpu()
		return nn_mat

	def save_nn_mat(self):
		for mode in ['RGB', 'OF']:
			nn_mat = self._gen_nn_mat(mode)
			path = os.path.join(self.results_dir, 'similarity_matrices_%d.mat' %self.load_epoch)
			self._savemat(self.similarity_matrices, path)

	def _get_data_fm(self, iterator):
		images, labels = next(iterator, (None, None))
		if images is None:
			return None, None
		images = torch.cat(images, dim=0)
		labels = torch.cat(labels, dim=0)	
		images, labels = Variable(images).cuda(), Variable(labels).cuda()
		return [images], None

	def _reconfigure_dataloader_train(self):
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			transforms.RandomCrop(self.net.input_spatial_size),
			transforms.RandomHorizontalFlip(), 
			self.transform_color,
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		transform_of = transforms.Compose([
			transforms.Scale(256), 
			transforms.RandomCrop(self.net.input_spatial_size),
			transforms.RandomHorizontalFlip(), 
			transforms.ToTensor(), 
			transforms.SubMeanDisplacement()])
		transform_cod = transforms.Compose([
			transforms.Scale(256), 
			transforms.RandomCrop(self.net.input_spatial_size),
			transforms.RandomHorizontalFlip(), 
			transforms.ToTensor()])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=True, num_frames=self.num_frames,
				split=self.split)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=True, transform_rgb=transform_rgb,
		  transform_of=transform_of, transform_cod=transform_cod, num_frames=self.num_frames, 
		  num_frames_cod=self.num_frames_cod, 
		  modalities=self.modalities, high_motion=self.high_motion,
		  time_flip=self.time_flip)
		self._reconfigure_dataloader(dataset, self.batch_size, shuffle=True)

	def _reconfigure_dataloader_test(self):
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(self.net.input_spatial_size),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		transform_of = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(self.net.input_spatial_size),

			transforms.ToTensor(), 
			transforms.SubMeanDisplacement()])
		transform_cod = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(self.net.input_spatial_size),
			transforms.ToTensor()])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, num_frames=self.num_frames,
				split=self.split)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=False, transform_rgb=transform_rgb,
		  transform_of=transform_of, transform_cod=transform_cod, num_frames=self.num_frames, 
		  num_frames_cod=self.num_frames_cod,
		  modalities=self.modalities)
		self._reconfigure_dataloader(dataset, self.batch_size_test, shuffle=True)

	def _reconfigure_dataloader_fm(self, mode):
		self.dataloader = None
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, num_frames=self.num_frames,
				split=self.split)
			dataset_infos.append(dataset_info)
		if mode == 'RGB':
			transform_rgb = transforms.Compose([
				transforms.Scale(256), 
				transforms.TenCrop(self.net.input_spatial_size),
				transforms.ToTensor(),
				transforms.Normalize(self.mean, self.std)])
			dataset = Dataset_RGB(infos=dataset_infos, train=False, transform=transform_rgb, num_test=5)
		elif mode == 'OF':	
			transform_of = transforms.Compose([
				transforms.Scale(256), 
				transforms.TenCrop(self.net.input_spatial_size),
				transforms.ToTensor(), 
				transforms.SubMeanDisplacement()])
			dataset = Dataset_OF(infos=dataset_infos, train=False, transform=transform_of, num_test=5,
				num_frames=self.num_frames, num_frames_cod=self.num_frames_cod)
		self._reconfigure_dataloader(dataset, 1, shuffle=True)	

	def _get_subnet(self, mode):
		if mode == 'RGB':
			return self.net.app_net
		elif mode == 'OF':
			return self.net.mot_net

	def _apply_per_epoch(self):
		super(Pretraining_Cross_and_Learn, self)._apply_per_epoch()

	def _reconfigure_tracker_train(self):
		if len(self.modalities) == 2:
			names = ['p', 'n']
		elif len(self.modalities) == 3:
			names = ['p', 'p', 'p', 'n', 'n', 'n']	
		self.tracker = Tracker_similarity(names)

	def _reconfigure_tracker_test(self):
		if len(self.modalities) == 2:
			names = ['p', 'n']
		elif len(self.modalities) == 3:
			names = ['p', 'p', 'p', 'n', 'n', 'n']	
		self.tracker = Tracker_similarity(names)

	def _track_per_iteration(self, output, labels, loss):
		norms = output[-1]
		output = output[:-1]
		sum_norm = 0
		for norm in norms:
			sum_norm += norm
		sum_norm = float(sum_norm / len(norms))
		norm = sum_norm
		similarities = np.zeros(2)
		for i in range(2):
			if len(output) != 2:
				similarity = (output[2*i] + output[2*i+1]) / 2
			else:
				similarity = output[i]
			if i == 0:
				length = similarity.size()[0]
			similarities[i] += torch.sum(similarity, dim=0).data.cpu().numpy()
		if loss is not None:
			loss = float(loss.data.cpu())
		norm /= length
		similarities /= length
		result = [('loss', loss), ('norm', norm), ('p', similarities[0]), ('n', similarities[1])]
		delimiter = ' '
		path = os.path.join(self.results_dir, 'train_iter.txt')
		f = open(path, 'a')
		f.write(utils.print_iterable(result, delimiter=' ', max_digits=self.max_digits, print_keys=False))
		f.write('\n')
		f.close()	

if __name__ == "__main__":
	e = Pretraining_Concat('test_def', batch_size=10)
	e.run()