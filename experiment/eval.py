import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import h5py
from scipy.io import savemat
import _pickle as pickle

import torch
from torch.autograd import Variable
from time import time

from experiment import utils
from experiment import Base

import compvis.data as data
from compvis import transforms_det as transforms

from compvis.datasets import Dataset_RGB, Dataset_OF, Dataset_Image
from compvis.models import get_network, Siamese, Siamese_fm

__all__ = ['Eval_feature_mat_RGB', 'Eval_feature_mat_OF', 'Eval_feature_mat_Image',
 'Eval_CNN_Visualization_Image', 'Eval_CNN_Visualization_RGB', 'Eval_CNN_Visualization_OF']

# TODO currently sim_mats are really bad, check for bugs, check network filters, 
# maybe try one of fm networks, or try one pretrained with supervision
# the pipeline used after a given similarity matrix seems to be right
# checkup the generation of feature matrices and similarity matrices then

class Base_eval(Base):
	def __init__(self,
			name,
			name_eval,
			name_finetuning = None,
			load_epoch=-1,
			data_key='ucf',
			source = 'l'
		):
		super(Base_eval, self).__init__(name=name, data_key=data_key, source=source)
		self.name_eval = name_eval
		self.name_finetuning = name_finetuning
		self.load_epoch = load_epoch
		if self.name_finetuning is not None:
			self.experiment_dir = os.path.join(self.results_dir, self.name_finetuning)
		else:
			self.experiment_dir = os.path.join(self.results_dir, 'experiment')
		self.results_dir = os.path.join(self.results_dir, self.name_eval)
		if not utils.mkdir(self.results_dir):
			print('Existing eval for given name')
		self.list_infos += [('name_finetuning', name_finetuning)]
		self.list_infos += [('name_eval', self.name_eval), ('load_epoch', load_epoch)]

	def _load_net(self):
		path_info = os.path.join(self.experiment_dir, 'net_info.pkl')
		net = get_network(path_info)
		if self.load_epoch != -2:
			path = os.path.join(self.experiment_dir, 'net_%i.pkl' %(self.load_epoch))
			new_sd = torch.load(path)
			utils.load_sd(net, new_sd)
		self._set_network(net)

	# Choose the right subnet for eval procedure
	def _set_network(self, net):
		raise NotImplementedError('_set_network() in Base_eval not implemented')





class Nearest_neighbour(Base_eval):
	dataloader = None
	def __init__(self,
			name,
			name_eval,
			load_epoch = -1,
			data_key = 'ucf',
			source = 'l',
			layer = 'conv4'
		):
		super(Base_eval_feature_mat, self).__init__(name=name, name_eval=name_eval, load_epoch=load_epoch,
		 data_key=data_key, source=source)
		self.layer = layer
		self.list_infos += [('layer', layer)]
		self.batch_size = 20

	def _gen_feature_matrix(self):
		if not hasattr(self, 'feature_matrix'):
			print('generating feature_matrixs')
			self._load_net()
			self.net.cuda()
			self.net.train(mode=False)
			self._reconfigure_dataloader_feature(False)
			iterator = iter(self.dataloader)
			features = []
			while(True):
				data, labels = self._get_data(iterator)
				if data is None:
					break
				output = self.net.get_feature_output(*data, layer=self.feature_layer)
				output = output.view(output.size(0), -1)
				features.append(output.data)
			self.feature_matrix = torch.cat(features, 0)
			self.net.cpu()

	def _gen_similarity_matrix(self):
		if not hasattr(self, 'similarity_matrix'):	
			self._gen_feature_matrix()
			print('generating similarity_matrix')
			length = self.feature_matrix.size(0)
			similarity_mat = torch.Tensor(length, length)
			mat_1 = self.feature_matrix
			for i in range(length):
				sim = utils.cos_sim(mat_1[i:i+1], mat_1)
				similarity_mat[i] = sim
			self.similarity_matrix = similarity_mat

	def _set_dataloader(self, dataset):
		self.dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=10, 
			shuffle=False, drop_last=False)	


	def _reconfigure_dataloader_feature_RGB(self):
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, source=self.source, num_frames=1)
			dataset_infos.append(dataset_info)
		dataset = Dataset_RGB(infos=dataset_infos, train=False, transform=transform_rgb,
		  num_test=1)
		self._reconfigure_dataloader(dataset, self.batch_size, shuffle=False)	

	def _prep_image_RGB(self, image_nn):
		normalize = transforms.Normalize(self.mean, self.std, inverted=True)
		return normalize(image_nn)

	def _reconfigure_dataloader_feature_OF(self):
		transform_of = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(224),
			transforms.ToTensor(), 
			transforms.SubMeanDisplacement()])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, source=self.source, num_frames=self.num_frames)
			dataset_infos.append(dataset_info)
		dataset = Dataset_OF(infos=dataset_infos, train=False, transform=transform_of, 
			num_frames=self.num_frames)
		self._reconfigure_dataloader(dataset, self.batch_size, shuffle=False)	

	def _prep_image_OF(self, image_nn):
		return image_nn



	def _gen_nearest_neighbour(self):
		self._gen_similarity_matrices()
		print('generating nearest_neighbour')
		similarity_mat = self.similarity_matrices['simMatrix']
		N_images = 4
		N_examples = 6
		images = []
		for i in range(N_images):
			sim = similarity_mat[i]
			sim, ind = sim.sort(descending=True)
			dataset = self._get_dataset_feature(False)
			images_i = []
			for j in range(N_examples):
				images_i.append(dataset.get_sample(ind[j]))
			images_i = torch.cat(images_i, 2)
			images.append(images_i)
		image_nn = torch.cat(images, 1)
		image_nn = self._prep_image(image_nn)
		to_pil = transforms.ToPILImage()
		image_nn = to_pil(image_nn)
		return image_nn


	def save_nearest_neighbour(self):
		image_nn = self._gen_nearest_neighbour()
		path = os.path.join(self.results_dir, 'nearest_neighbour_%d.png' %self.load_epoch)
		image_nn.save(path)

	def _save_dict_tensors(self, dict, path):
		f = h5py.File(path)
		for key, tensor in dict.items():	
			f.create_dataset(key, data=tensor.cpu().numpy())

	def _get_data(self, iterator):
		images, labels = next(iterator, (None, None))
		if images is None:
			return None, None
		images = torch.cat(images, dim=0)
		labels = torch.cat(labels, dim=0)	
		images, labels = Variable(images).cuda(), Variable(labels).cuda()
		return [images], labels




























class Base_eval_feature_mat(Base_eval):
	dataloader = None
	def __init__(self,
			name,
			name_eval,
			load_epoch=-1,
			data_key='ucf',
			source = 'l',
			feature_layer = 'conv4'
		):
		super(Base_eval_feature_mat, self).__init__(name=name, name_eval=name_eval, load_epoch=load_epoch,
		 data_key=data_key, source=source)
		self.feature_layer = feature_layer
		self.list_infos += [('feature_layer', self.feature_layer)]
		self.batch_size = 20

	def _set_dataloader(self, dataset):
		self.dataloader = data.DataLoader(dataset, batch_size=self.batch_size, num_workers=10, 
			shuffle=False, drop_last=False)	

	# def _net_forward(self, input):
	# 	if self.feature_layer == 'fc6':
	# 		return self.net.get_fc6_features(input)
	# 	if self.feature_layer == 'conv5':
	# 		return self.net.get_conv5_features(input)
	# 	if self.feature_layer == 'convpp5':
	# 		return self.net.get_convpp5_features(input)
	# 	if self.feature_layer == 'conv4':
	# 		return self.net.get_conv4_features(input)

	def _gen_feature_matrices(self):
		if not hasattr(self, 'feature_matrices'):
			print('generating feature_matrices')
			self._load_net()
			self.net.cuda()
			self.net.train(mode=False)
			self.feature_matrices = {}
			for flip, name in [(False, 'featMatrix'), (True, 'featMatrix_flip')]:
				self._reconfigure_dataloader_feature(flip)
				iterator = iter(self.dataloader)
				features = []
				while(True):
					data, labels = self._get_data(iterator)
					if data is None:
						break
					output = self.net.get_feature_output(*data, layer=self.feature_layer)
					output = output.view(output.size(0), -1)
					features.append(output.data)
				self.feature_matrices[name] = torch.cat(features, 0)
			self.net.cpu()

	def _gen_similarity_matrices(self):
		if not hasattr(self, 'similarity_matrices'):	
			self._gen_feature_matrices()
			print('generating similarity_matrices')
			length = self.feature_matrices['featMatrix'].size(0)
			self.similarity_matrices = {}
			for flip, name in [(False, 'simMatrix'), (True, 'simMatrix_flip')]:
				similarity_mat = torch.Tensor(length, length)
				mat_1 = self.feature_matrices['featMatrix']
				if not flip:
					mat_2 = self.feature_matrices['featMatrix']
				else:
					mat_2 = self.feature_matrices['featMatrix_flip']
				for i in range(length):
					sim = utils.cos_sim(mat_1[i:i+1], mat_2)
					similarity_mat[i] = sim
				self.similarity_matrices[name] = similarity_mat

	def _gen_nearest_neighbour(self):
		self._gen_similarity_matrices()
		print('generating nearest_neighbour')
		similarity_mat = self.similarity_matrices['simMatrix']
		N_images = 4
		N_examples = 6
		images = []
		for i in range(N_images):
			sim = similarity_mat[i]
			sim, ind = sim.sort(descending=True)
			dataset = self._get_dataset_feature(False)
			images_i = []
			for j in range(N_examples):
				images_i.append(dataset.get_sample(ind[j]))
			images_i = torch.cat(images_i, 2)
			images.append(images_i)
		image_nn = torch.cat(images, 1)
		image_nn = self._prep_image(image_nn)
		to_pil = transforms.ToPILImage()
		image_nn = to_pil(image_nn)
		return image_nn

	def save_feature_matrices(self):
		self._gen_feature_matrices()
		path = os.path.join(self.results_dir, 'feature_matrices_%d.mat' %self.load_epoch)
		self._save_dict_tensors(self.feature_matrices, path)

	def save_similarity_matrices(self):
		self._gen_similarity_matrices()
		path = os.path.join(self.results_dir, 'similarity_matrices_%d.mat' %self.load_epoch)
		self._savemat(self.similarity_matrices, path)

	def save_nearest_neighbour(self):
		image_nn = self._gen_nearest_neighbour()
		path = os.path.join(self.results_dir, 'nearest_neighbour_%d.png' %self.load_epoch)
		image_nn.save(path)
	
	def load_similarity_matrices(self):
		path = os.path.join(self.results_dir, 'similarity_matrices_%d.mat' %self.load_epoch)
		self.similarity_matrices = self._load_dict_tensors(path)

	def _save_dict_tensors(self, dict, path):
		f = h5py.File(path)
		for key, tensor in dict.items():	
			f.create_dataset(key, data=tensor.cpu().numpy())

	def _savemat(self, dict, path):
		dict_save = {}
		for key, tensor in dict.items():	
			dict_save[key] = tensor.cpu().numpy()
		savemat(path, dict_save)

	def _load_dict_tensors(self, path):
		f = h5py.File(path)
		dict = {}
		for key in list(f.keys()):	
			arr = np.array(f[key])
			tensor = torch.Tensor(arr).cuda()
			dict[key] = tensor
		return dict

	def _get_data(self, iterator):
		images, labels = next(iterator, (None, None))
		if images is None:
			return None, None
		images = torch.cat(images, dim=0)
		labels = torch.cat(labels, dim=0)	
		images, labels = Variable(images).cuda(), Variable(labels).cuda()
		return [images], labels

	def _reconfigure_dataloader_feature(self, flip):
		dataset = self._get_dataset_feature(flip)
		self._reconfigure_dataloader(dataset, self.batch_size, shuffle=False)	

	# dataset should transforms samples in a deterministic manner
	def _get_dataset_feature(self, flip):
		raise NotImplementedError('_get_dataset_feature() in Base_eval_feature_mat not impl.')

	def _prep_image(self, image_nn):
		raise NotImplementedError('_prep_image() in Base_eval_feature_mat not impl.')		

class Eval_feature_mat_Image(Base_eval_feature_mat):
	def _set_network(self, net):
		if hasattr(net, 'app_net'):
			self.net = net.app_net
		else:
			self.net = net

	def _get_dataset_feature(self, flip):
		if self.data_key == 'leeds' and self.source != 'l':
			scale_size = 224
		else:
			scale_size = 256
		transform_rgb = transforms.Compose([
			transforms.Scale(scale_size), 
			transforms.CenterCrop(224),
			transforms.HorizontalFlip(flip),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=None, source=self.source)
			dataset_infos.append(dataset_info)
		dataset = Dataset_Image(infos=dataset_infos, train=False, transform=transform_rgb,
		  )
		return dataset

	def _prep_image(self, image_nn):
		normalize = transforms.Normalize(self.mean, self.std, inverted=True)
		return normalize(image_nn)
		
class Eval_feature_mat_RGB(Base_eval_feature_mat):
	def _set_network(self, net):
		if hasattr(net, 'app_net'):
			self.net = net.app_net
		else:
			self.net = net

	def _get_dataset_feature(self, flip):
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(224),
			transforms.HorizontalFlip(flip),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, source=self.source, num_frames=1)
			dataset_infos.append(dataset_info)
		dataset = Dataset_RGB(infos=dataset_infos, train=False, transform=transform_rgb,
		  num_test=1)
		return dataset

	def _prep_image(self, image_nn):
		normalize = transforms.Normalize(self.mean, self.std, inverted=True)
		return normalize(image_nn)

class Eval_feature_mat_OF(Base_eval_feature_mat):
	def _set_network(self, net):
		if hasattr(net, 'mot_net'):
			self.net = net.mot_net
		else:
			self.net = net
		self.num_frames = int(self.net.input_dim / 2)

	def _get_dataset_feature(self, flip):
		transform_of = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(224),
			transforms.HorizontalFlip(flip),
			transforms.ToTensor(), 
			transforms.SubMeanDisplacement()])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, source=self.source, num_frames=self.num_frames)
			dataset_infos.append(dataset_info)
		dataset = Dataset_OF(infos=dataset_infos, train=False, transform=transform_of, 
			num_frames=self.num_frames)
		return dataset

	def _prep_image(self, image_nn):
		return image_nn

class Base_eval_CNN_Visualization(Base_eval):
	dataloader = None
	def __init__(self,
			name,
			name_eval = 'eval_cnn_viz',
			name_finetuning = None,
			load_epoch = -1,
			data_key = 'ucf',
			source = 'l',
			num_images = 200
		):
		super(Base_eval_CNN_Visualization, self).__init__(name=name, name_finetuning=name_finetuning,
		 name_eval=name_eval, load_epoch=load_epoch, data_key=data_key, source=source)
		self.num_images = num_images
		self.name_data = 'data_%s_%s_%d.pkl' %(self.data_key, self.source, self.num_images)

	def _save_images(self):
		dataset = self._get_dataset_viz()
		self._reconfigure_dataloader_viz()
		iterator = iter(self.dataloader)
		data = []
		for i in range(self.num_images):
			data.append(self._get_data(iterator))

		path_data = os.path.join(self.results_dir, self.name_data)
		pickle.dump(data, open(path_data, 'wb'))

	def _save_network(self):
		self._load_net()
		path = os.path.join(self.results_dir, 'net_params.pkl')
		torch.save(self.net.state_dict(), path)
		path_info = os.path.join(self.results_dir, 'net_info.pkl')
		utils.pickle_dump(self.net.get_net_info(), path_info)

	def get_network_data(self):
		self._load_net()
		path_data = os.path.join(self.results_dir, self.name_data)
		if os.path.exists(path_data):
			data = utils.pickle_load(path_data)
		else:
			dataset = self._get_dataset_viz()
			self._reconfigure_dataloader_viz()
			iterator = iter(self.dataloader)
			data = []
			for i in range(self.num_images):
				data.append(self._get_data(iterator))
		return self.net, data

	def prepare_eval(self):
		self._save_network()
		self._save_images()

	def _set_dataloader(self, dataset):
		self.dataloader = data.DataLoader(dataset, batch_size=1, num_workers=1, 
			shuffle=None, sampler=self.num_images, drop_last=False)	

	def _reconfigure_dataloader_viz(self):
		dataset = self._get_dataset_viz()
		self._reconfigure_dataloader(dataset, 1, shuffle=None)	

	def _get_data(self, iterator):
		raise NotImplementedError('_get_data() in Base_eval_CNN_Visualization not implemented')

	def _get_dataset_viz(self):
		raise NotImplementedError('Base_eval_CNN_Visualization should implement dataset_type (type)')

class Eval_CNN_Visualization_Image(Base_eval_CNN_Visualization):
	def _get_dataset_viz(self):
		if self.data_key == 'leeds' and self.source != 'l':
			scale_size = 224
		else:
			scale_size = 256
		transform_rgb = transforms.Compose([
			transforms.Scale(scale_size), 
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, source=self.source)
			dataset_infos.append(dataset_info)
		dataset = Dataset_Image(infos=dataset_infos, train=False, transform=transform_rgb,
		  )
		return dataset

	def _set_network(self, net):
		if hasattr(net, 'app_net'):
			self.net = net.app_net
		else:
			self.net = net

	def _get_data(self, iterator):
		images, labels = next(iterator, (None, None))
		if images is None:
			return None, None
		images = torch.cat(images, dim=0)
		labels = torch.cat(labels, dim=0)	
		return images, labels

class Eval_CNN_Visualization_RGB(Base_eval_CNN_Visualization):
	def _get_dataset_viz(self):
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, source=self.source)
			dataset_infos.append(dataset_info)
		dataset = Dataset_RGB(infos=dataset_infos, train=False, transform=transform_rgb,
		  num_test=3)
		return dataset

	def _set_network(self, net):
		if hasattr(net, 'app_net'):
			self.net = net.app_net
		else:
			self.net = net

	def _get_data(self, iterator):
		images, labels = next(iterator, (None, None))
		if images is None:
			return None, None
		images = torch.cat(images, dim=0)
		labels = torch.cat(labels, dim=0)	
		images = images[1:2]
		labels = labels[1:2]
		return images, labels

class Eval_CNN_Visualization_OF(Base_eval_CNN_Visualization):
	def _get_dataset_viz(self):
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		transform = transforms.Compose([
			transforms.Scale(256), 
			transforms.CenterCrop(224),
			transforms.ToTensor(), 
			transforms.SubMeanDisplacement()])
		dataset_infos = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=False, source=self.source, num_frames=self.num_frames)
			dataset_infos.append(dataset_info)
		dataset = Dataset_OF(infos=dataset_infos, train=False, transform=transform,
			transform_rgb=transform_rgb, num_frames=self.num_frames, num_test=3)
		return dataset

	def _set_network(self, net):
		if hasattr(net, 'mot_net'):
			self.net = net.mot_net
		else:
			self.net = net
		self.num_frames = int(self.net.input_dim / 2)

	def _get_data(self, iterator):
		flows, labels, images = next(iterator, (None, None, None))
		if images is None:
			return None, None, None
		flows = torch.cat(flows, dim=0)
		labels = torch.cat(labels, dim=0)
		images = torch.cat(images, dim=0)
		flows = flows[1:2]
		labels = labels[1:2]
		images = images[1:2]
		return flows, labels, images

# class Eval_CNN_Visualization_Image(Base_eval):
# 	dataloader = None
# 	def __init__(self,
# 			name,
# 			name_eval,
# 			load_epoch=-1,
# 			data_key='ucf',
# 			source = 'l',
# 		):
# 		super(Eval_CNN_Visualization_Image, self).__init__(name=name, name_eval=name_eval, 
# 			load_epoch=load_epoch, data_key=data_key, source=source)
# 		self.num_images = 10

# 	def _save_images(self):
# 		dataset = self._get_dataset_viz()
# 		indices = np.linspace(0, len(dataset), self.num_images).astype(np.int32)
# 		self._reconfigure_dataloader_viz()
# 		iterator = iter(self.dataloader)
# 		for i in range(10):
# 			image, label = self._get_data(iterator)
# 			path_image_label = os.path.join(self.results_dir, 'image_label_%d.pkl' %i)
# 			pickle.dump([image, label], open(path_image_label, 'wb'))

# 	def _save_network(self):
# 		self._load_net()
# 		path = os.path.join(self.results_dir, 'net_params.pkl' )
# 		torch.save(self.net.state_dict(), path)
# 		path_info = os.path.join(self.results_dir, 'net_info.pkl')
# 		utils.pickle_dump(self.net.get_net_info(), path_info)

# 	def prepare_eval(self):
# 		self._save_network()
# 		self._save_images()

# 	def _set_dataloader(self, dataset):
# 		self.dataloader = data.DataLoader(dataset, batch_size=1, num_workers=1, 
# 			shuffle=True, drop_last=False)	

# 	def _reconfigure_dataloader_viz(self):
# 		dataset = self._get_dataset_viz()
# 		self._reconfigure_dataloader(dataset, 1)	

# 	def _get_dataset_viz(self):
# 		if self.data_key == 'leeds' and self.source != 'l':
# 			scale_size = 224
# 		else:
# 			scale_size = 256
# 		transform_rgb = transforms.Compose([
# 			transforms.Scale(scale_size), 
# 			transforms.CenterCrop(224),
# 			transforms.ToTensor()])
# 		dataset_infos = []
# 		for dataset_info_type in self.dataset_info_types:
# 			dataset_info = dataset_info_type(train=False, source=self.source)
# 			dataset_infos.append(dataset_info)
# 		dataset = Dataset_Image(infos=dataset_infos, train=False, transform=transform_rgb,
# 		  )
# 		return dataset

# 	def _set_network(self, net):
# 		if isinstance(net, (Siamese, Siamese_fm)):
# 			self.net = net.app_net
# 		else:
# 			self.net = net

# 	def _get_data(self, iterator):
# 		images, labels = next(iterator, (None, None))
# 		if images is None:
# 			return None, None
# 		images = torch.cat(images, dim=0)
# 		labels = torch.cat(labels, dim=0)	
# 		return images, labels

# class Eval_CNN_Visualization_RGB(Base_eval):
# 	dataloader = None
# 	def __init__(self,
# 			name,
# 			name_eval,
# 			load_epoch=-1,
# 			data_key='ucf',
# 			source = 'l',
# 		):
# 		super(Eval_CNN_Visualization_RGB, self).__init__(name=name, name_eval=name_eval, 
# 			load_epoch=load_epoch, data_key=data_key, source=source)
# 		self.num_images = 10

# 	def _save_images(self):
# 		dataset = self._get_dataset_viz()
# 		indices = np.linspace(0, len(dataset), self.num_images).astype(np.int32)
# 		self._reconfigure_dataloader_viz()
# 		iterator = iter(self.dataloader)
# 		for i in range(10):
# 			image, label = self._get_data(iterator)
# 			path_image_label = os.path.join(self.results_dir, 'image_label_%d.pkl' %i)
# 			pickle.dump([image, label], open(path_image_label, 'wb'))

# 	def _save_network(self):
# 		self._load_net()
# 		path = os.path.join(self.results_dir, 'net_params.pkl' )
# 		torch.save(self.net.state_dict(), path)
# 		path_info = os.path.join(self.results_dir, 'net_info.pkl')
# 		utils.pickle_dump(self.net.get_net_info(), path_info)

# 	def prepare_eval(self):
# 		self._save_network()
# 		self._save_images()

# 	def _set_dataloader(self, dataset):
# 		self.dataloader = data.DataLoader(dataset, batch_size=1, num_workers=1, 
# 			shuffle=True, drop_last=False)	

# 	def _reconfigure_dataloader_viz(self):
# 		dataset = self._get_dataset_viz()
# 		self._reconfigure_dataloader(dataset, 1)	

# 	def _get_dataset_viz(self):
# 		transform_rgb = transforms.Compose([
# 			transforms.Scale(256), 
# 			transforms.CenterCrop(224),
# 			transforms.ToTensor()])
# 		dataset_infos = []
# 		for dataset_info_type in self.dataset_info_types:
# 			dataset_info = dataset_info_type(train=False, source=self.source)
# 			dataset_infos.append(dataset_info)
# 		dataset = Dataset_RGB(infos=dataset_infos, train=False, transform=transform_rgb,
# 		  num_test=1)
# 		return dataset

# 	def _set_network(self, net):
# 		if isinstance(net, (Siamese, Siamese_fm)):
# 			self.net = net.app_net
# 		else:
# 			self.net = net

# 	def _get_data(self, iterator):
# 		images, labels = next(iterator, (None, None))
# 		if images is None:
# 			return None, None
# 		images = torch.cat(images, dim=0)
# 		labels = torch.cat(labels, dim=0)	
# 		return images, labels

# class Eval_CNN_Visualization_OF(Base_eval):
# 	dataloader = None
# 	def __init__(self,
# 			name,
# 			name_eval,
# 			load_epoch=-1,
# 			data_key='ucf',
# 			source = 'l',
# 		):
# 		super(Eval_CNN_Visualization_OF, self).__init__(name=name, name_eval=name_eval, 
# 			load_epoch=load_epoch, data_key=data_key, source=source)
# 		self.num_images = 10

# 	def _save_images(self):
# 		dataset = self._get_dataset_viz()
# 		indices = np.linspace(0, len(dataset), self.num_images).astype(np.int32)
# 		self._reconfigure_dataloader_viz()
# 		iterator = iter(self.dataloader)
# 		for i in range(10):
# 			flow, label, image = self._get_data(iterator)
# 			path_flow_label_image = os.path.join(self.results_dir, 'flow_label_image_%d.pkl' %i)
# 			pickle.dump([flow, label, image], open(path_flow_label_image, 'wb'))

# 	def _save_network(self):
# 		self._load_net()
# 		self.num_frames = int(self.net.input_dim/2)
# 		path = os.path.join(self.results_dir, 'net_params.pkl' )
# 		torch.save(self.net.state_dict(), path)
# 		path_info = os.path.join(self.results_dir, 'net_info.pkl')
# 		utils.pickle_dump(self.net.get_net_info(), path_info)

# 	def prepare_eval(self):
# 		self._save_network()
# 		self._save_images()

# 	def _set_dataloader(self, dataset):
# 		self.dataloader = data.DataLoader(dataset, batch_size=1, num_workers=1, 
# 			shuffle=True, drop_last=False)	

# 	def _reconfigure_dataloader_viz(self):
# 		dataset = self._get_dataset_viz()
# 		self._reconfigure_dataloader(dataset, 1)	

# 	def _get_dataset_viz(self):
# 		transform_rgb = transforms.Compose([
# 			transforms.Scale(256), 
# 			transforms.CenterCrop(224),
# 			transforms.ToTensor()])
# 		transform = transforms.Compose([
# 			transforms.Scale(256), 
# 			transforms.CenterCrop(224),
# 			transforms.ToTensor(), 
# 			transforms.SubMeanDisplacement()])
# 		dataset_infos = []
# 		for dataset_info_type in self.dataset_info_types:
# 			dataset_info = dataset_info_type(train=False, source=self.source, num_frames=self.num_frames)
# 			dataset_infos.append(dataset_info)
# 		dataset = Dataset_OF(infos=dataset_infos, train=False, transform=transform,
# 			transform_rgb=transform_rgb, num_frames=self.num_frames, num_test=1)
# 		return dataset

# 	def _set_network(self, net):
# 		if isinstance(net, (Siamese, Siamese_fm)):
# 			self.net = net.mot_net
# 		else:
# 			self.net = net

# 	def _get_data(self, iterator):
# 		flows, labels, images = next(iterator, (None, None, None))
# 		if images is None:
# 			return None, None, None
# 		flows = torch.cat(flows, dim=0)
# 		labels = torch.cat(labels, dim=0)
# 		images = torch.cat(images, dim=0)
# 		return flows, labels, images