import numpy as np
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from experiment import utils
from experiment import Base_experiment_pretraining
from experiment.tracker import Tracker_classification, Tracker_similarity
from compvis.models import Single_def, Single_fm
from compvis.datasets import Dataset_def, Dataset_fm, Dataset_RGB, Dataset_OF, Dataset_sl

from compvis import transforms_det as transforms 

__all__ = ['Experiment_pretraining_sl_def' ,'Experiment_pretraining_sl_fm']

class Experiment_pretraining_sl_def(Base_experiment_pretraining):
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
			source = 'l',
			rgb = 0.3,
			split_channels = False,
			dropout = 0.5,
			num_frames = 20,
			min_spacing = 3,
			bottleneck = 1024,
			early_cat = True,
			max_shift = 0,
			use_rand = True,
			high_motion = False,
			time_flip = False,
			same_video = True,
			min_msd = 100, 
			max_spacing = 10,
			union = False
		):

		super(Experiment_pretraining_sl_def, self).__init__(name=name,batch_size=batch_size, epochs=epochs, 
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay, 
			norm=norm, data_key=data_key, source=source, rgb=rgb, split_channels=split_channels, 
			dropout=dropout, use_rand=use_rand)
		self.num_frames = num_frames
		self.min_spacing = min_spacing
		self.bottleneck = bottleneck
		self.early_cat = early_cat
		self.max_shift = max_shift
		self.high_motion = high_motion
		self.time_flip = time_flip
		self.same_video = same_video
		self.min_msd = min_msd
		self.max_spacing = max_spacing
		self.union = union
		self.list_infos += [('num_frames', num_frames), ('min_spacing',min_spacing), 
			('bottleneck', bottleneck), ('early_cat', early_cat), ('max_shift', max_shift), 
			('high_motion', high_motion), ('time_flip', time_flip), ('same_video', same_video),
			('min_msd', min_msd), ('max_spacing', max_spacing), ('union', union)]
		self.net = Single_def(norm=self.norm, dropout=self.dropout, bottleneck=self.bottleneck, 
			early_cat=self.early_cat, union=self.union)
		self.tracker = Tracker_classification()
		self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate,
			momentum=0.9, weight_decay=self.weight_decay)
		self.dataset_type = Dataset_sl
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
		labels_zero_3 = Variable(torch.LongTensor(batch_size).zero_().cuda())
		labels_one = Variable(torch.LongTensor(batch_size).zero_().cuda() + 1)
		labels = [labels_one, labels_zero_1, labels_zero_2]
		# self._save_sample(data, labels)
		return data, labels

	def _save_sample(self, data, label):
		# data = list, labels = list
		path_images_dir = os.path.join(self.results_dir, 'samples')
		utils.mkdir(path_images_dir)
		counter_field = 0
		for image in data:
			image = image.data.cpu().numpy()
			for i in range(image.shape[0]):
				arr, mini, maxi = utils.normalize(image[i])
				arr = arr.transpose(1,2,0)
				img = utils.get_color_img(arr)
				img = Image.fromarray(img)
				image_name = '%03d_%d_%.4f_%.4f.png' %(i, counter_field, mini, maxi)
				path_image = os.path.join(path_images_dir, image_name)
				img.save(path_image)
			counter_field += 1
		raise Exception
			
	# Todo remove weighting
	def _get_loss(self, output, labels):
		loss = 0
		for i in range(3):
			loss += self.criterion(output[i], labels[i])
		loss /= 3
		return loss

	def _reconfigure_dataloader_train(self):
		rgb = self.rgb
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			transforms.RandomCrop(self.net.input_spatial_size, max_shift=self.max_shift),
			transforms.RandomHorizontalFlip(), 
			self.transform_color,
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])

		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=True, source=self.source, num_frames=self.num_frames,
				min_msd=self.min_msd)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=True, transform_rgb=transform_rgb,
			num_frames=self.num_frames, min_spacing=self.min_spacing, high_motion=self.high_motion,
			time_flip=self.time_flip, same_video=self.same_video, 
			min_msd=self.min_msd, max_spacing=self.max_spacing)
		self._reconfigure_dataloader(dataset, self.batch_size, shuffle=True)

	def _reconfigure_dataloader_test(self):
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(self.net.input_spatial_size),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])

		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, source=self.source, num_frames=self.num_frames,
				min_msd=self.min_msd)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=False, transform_rgb=transform_rgb,
			num_frames=self.num_frames, min_spacing=self.min_spacing, high_motion=self.high_motion, 
			same_video=self.same_video, 
			min_msd=self.min_msd, max_spacing=self.max_spacing)
		self._reconfigure_dataloader(dataset, self.batch_size_test, shuffle=True)

	def _reconfigure_tracker_train(self):
		self.tracker = Tracker_classification()

	def _reconfigure_tracker_test(self):
		self.tracker = Tracker_classification()


class Experiment_pretraining_sl_fm(Base_experiment_pretraining):
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
			source = 'l',
			rgb = 0.3,
			split_channels = False,
			dropout = 0.5,
			num_frames = 20,
			min_spacing = 3,
			layer = 'fc6',
			max_shift = 0,
			use_rand = True,
			high_motion = False,
			time_flip = False,
			min_msd = 100,
			max_spacing = 10,
			scheme = 0,
			similarity_scheme = 'cosine'
		):
		super(Experiment_pretraining_sl_fm, self).__init__(name=name, batch_size=batch_size, epochs=epochs, 
			learning_rate=learning_rate, lr_decay_scheme=lr_decay_scheme, weight_decay=weight_decay, 
			norm=norm, data_key=data_key, source=source, rgb=rgb, split_channels=split_channels, 
			dropout=dropout, use_rand=use_rand)
		self.num_frames = num_frames
		self.min_spacing = min_spacing
		self.layer = layer
		self.max_shift = max_shift
		self.high_motion = high_motion
		self.time_flip = time_flip
		self.min_msd = min_msd
		self.max_spacing = max_spacing
		self.scheme = scheme
		self.similarity_scheme = similarity_scheme
		self.list_infos += [('num_frames', num_frames), ('min_spacing',min_spacing), ('layer', layer), 
			('max_shift', max_shift), ('high_motion', high_motion), ('time_flip', time_flip), 
			('min_msd', min_msd), ('max_spacing', max_spacing), ('scheme', scheme), 
			('similarity_scheme', similarity_scheme)]
		self.net = Single_fm(norm=self.norm, dropout=self.dropout, layer=self.layer, scheme=self.scheme,
			similarity_scheme = self.similarity_scheme)
		self.tracker = Tracker_similarity()
		self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate,
			momentum=0.9, weight_decay=self.weight_decay)
		self.dataset_type = Dataset_sl

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

	def _get_loss(self, output, labels):
		half = int(len(output) / 2)
		sim_true = 0
		sim_false = 0
		for i in range(half):
			sim_true += output[i]
			sim_false += output[half + i]
		loss_sim = torch.mean(sim_false - sim_true, dim=0) / half
		return loss_sim

	def _reconfigure_dataloader_train(self):
		rgb = self.rgb
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			transforms.RandomCrop(self.net.input_spatial_size, max_shift=self.max_shift),
			transforms.RandomHorizontalFlip(), 
			self.transform_color,
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])

		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=True, source=self.source, num_frames=self.num_frames,
				min_msd=self.min_msd)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=True, transform_rgb=transform_rgb,
			num_frames=self.num_frames, min_spacing=self.min_spacing, high_motion=self.high_motion,
			time_flip=self.time_flip, min_msd=self.min_msd, max_spacing=self.max_spacing)
		self._reconfigure_dataloader(dataset, self.batch_size, shuffle=True)

	def _reconfigure_dataloader_test(self):
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(self.net.input_spatial_size),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])

		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, source=self.source, num_frames=self.num_frames,
				min_msd=self.min_msd)
			dataset_infos.append(dataset_info)
		dataset = self.dataset_type(infos=dataset_infos, train=False, transform_rgb=transform_rgb,
			num_frames=self.num_frames, min_spacing=self.min_spacing, high_motion=self.high_motion, 
			min_msd=self.min_msd, max_spacing=self.max_spacing)
		self._reconfigure_dataloader(dataset, self.batch_size_test, shuffle=True)

	def _reconfigure_tracker_train(self):
		self.tracker = Tracker_similarity(['p', 'n'])

	def _reconfigure_tracker_test(self):
		self.tracker = Tracker_similarity(['p', 'n'])