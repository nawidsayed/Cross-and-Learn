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
from experiment import Base

import compvis.data as data
from compvis import transforms_det as transforms 
from compvis.datasets import Dataset_OF, Dataset_RGB
from compvis.models import get_network

__all__ = ['Cluster_OF', 'Cluster_RGB']

class Base_Cluster(Base):
	dataloader = None
	num_crops = None
	feature_dim = None
	def __init__(self,
			name,
			name_cluster,
			name_labels,
			name_experiment = None,
			data_key = 'ucf',
			source = 'l',
			load_epoch_pt = -1,
			num_test = -20,
			layer = 'fc6',
			condense = False,
			metric = 'euclidean',
			num_clusters = 200,
			kmeans_iterations = 50,
			use_cc = False,
			use_splitchannels = False
		):
		super(Base_Cluster, self).__init__(name=name, data_key=data_key, source=source)

		self.results_dir = os.path.join(self.results_dir, name_cluster)
		if not utils.mkdir(self.results_dir):
			print('Existing cluster for given name')
		if name_experiment is None:
			name_experiment = 'experiment'
		self.name_experiment = name_experiment
		self.name_cluster = name_cluster
		self.name_labels = name_labels
		self.load_epoch_pt = load_epoch_pt
		self.num_test = num_test
		self.layer = layer
		self.condense = condense
		self.metric = metric
		if num_test < 0:
			self.size_per_sample = int(-300 / num_test)
		else:
			self.size_per_sample = num_test
		if self.condense:
			self.size_per_sample = 1
		self.num_clusters = num_clusters
		self.kmeans_iterations = kmeans_iterations
		self.use_cc = use_cc
		self.use_splitchannels = use_splitchannels
		self.list_infos += [('load_epoch_pt', load_epoch_pt), ('name_cluster', name_cluster), 
		('name_labels', name_labels),
		('name_experiment', name_experiment), ('num_test', num_test), ('layer', layer),
		('condense', condense), ('metric', metric), ('num_clusters', num_clusters), 
		('kmeans_iterations', kmeans_iterations), ('use_cc', use_cc), 
		('use_splitchannels', use_splitchannels)]
		self._load_pretraining()
		# self.pool = nn.MaxPool2d(kernel_size=1, stride=1)
		self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
		self.num_frames = int(self.net.input_dim / 2)
		self.list_infos += [('num_frames', self.num_frames)]
		self.num_crops = 10
		if self.use_cc:
			self.num_crops = 1

	def run(self):
		self._generate_features()
		self._save_features_indices()
		self._save_labels_old()
		self._generate_clusters()

	def acc(self, train=True):
		print(train)
		self._print_infos()	
		if train:
			path_labels_old = os.path.join(self.results_dir, 'labels_old.pkl')
			path_labels = os.path.join(self.results_dir, '%s.pkl' %self.name_labels)
		else:
			path_labels_old = os.path.join(self.results_dir, 'labels_old_test.pkl')
			path_labels = os.path.join(self.results_dir, '%s_test.pkl' %self.name_labels)
		labels_old = pickle.load(open(path_labels_old, 'rb'))
		labels = pickle.load(open(path_labels, 'rb'))[0]
		if not len(labels) == len(labels_old):
			raise Exception('label missmatch: %d, %d' %(len(labels), len(labels_old)))
		matrix_sim = np.zeros((self.num_clusters, self.num_clusters))
		for i in range(len(labels)):
			label_old = labels_old[i]
			label =	labels[i]
			for l in label:
				matrix_sim[label_old, l] += 1
		matrix_cost = np.max(matrix_sim) - matrix_sim
		row, col = linear_sum_assignment(matrix_cost)
		sum_correct = matrix_sim[row, col].sum()
		sum_total = matrix_sim.sum()
		ratio = sum_correct / sum_total
		if train:
			path_acc = os.path.join(self.results_dir, '%s_acc_train.txt' %self.name_labels)
		else:
			path_acc = os.path.join(self.results_dir, '%s_acc_test.txt' %self.name_labels)
		f = open(path_acc, 'a')
		# f.write(str(sum_correct))
		# f.write('\n')
		# f.write(str(sum_total))
		# f.write('\n')
		f.write(str(ratio))
		f.write('\n')
		f.close()	
		print(sum_correct, sum_total, ratio)

	def distance_distribution(self):
		self._print_infos()	
		num_drawers = 10000
		max_dist = 2
		path_features = os.path.join(self.results_dir, 'features.pkl')
		self.features = torch.load(path_features)
		self.feature_dim = self.features.size(1)
		self.features = self.features.cuda()
		self.histogram = np.zeros(num_drawers).astype(int)
		counter = 0
		for feature in self.features:
			feat_dist = self._get_distance(feature, self.features)
			# feat_dist = torch.sqrt(torch.sum((self.features - feature) ** 2, dim=1))
			feat_dist = feat_dist * num_drawers / max_dist
			feat_dist = feat_dist.long()
			feat_dist = feat_dist.cpu().numpy()
			for dist in feat_dist:
				self.histogram[dist] += 1
			if counter % 1000 == 0:
				print(counter)
			counter += 1
		path_histogram = os.path.join(self.results_dir, 'histogram.pkl')
		pickle.dump(self.histogram, open(path_histogram, 'wb'))



	def _generate_features(self):
		path_features = os.path.join(self.results_dir, 'features.pkl')
		path_features_test = os.path.join(self.results_dir, 'features_test.pkl')
		path_indices = os.path.join(self.results_dir, 'indices.pkl')
		path_indices_test = os.path.join(self.results_dir, 'indices_test.pkl')
		path_labels_old = os.path.join(self.results_dir, 'labels_old.pkl')
		path_labels_old_test = os.path.join(self.results_dir, 'labels_old_test.pkl')
		if os.path.exists(path_features) and os.path.exists(path_indices):
			print('features and indices already exist')
			self.features = torch.load(path_features)
			self.features = self.features.cuda()
			self.features_test = torch.load(path_features_test)
			self.features_test = self.features_test.cuda()
			self.indices = pickle.load(open(path_indices, 'rb'))
			self.indices_test = pickle.load(open(path_indices_test, 'rb'))
			self.feature_dim = self.features.size(1)
			if os.path.exists(path_labels_old):
				self.labels_old = pickle.load(open(path_labels_old, 'rb'))
				self.labels_old_test = pickle.load(open(path_labels_old_test, 'rb'))
			return None
		self.net.cuda()
		self.pool.cuda()
		self._write_infos()	
		self._print_infos()	
		t0_tot = time()
		for train in [True, False]:
			self._reconfigure_dataloader_cluster(train=train)
			self.net.train(mode=False)
			iterator = iter(self.dataloader)
			length_iter = len(iterator)
			indices = [0]
			length = 0
			labels_old = []
			while(True):
				data, labels = self._get_data(iterator)
				if data is None:
					break
				labels_old.append(int(labels[0]))
				if not self.condense:
					for data_feat in data:
						data_feat = Variable(data_feat).cuda()
						output = self.net.get_feature_output(data_feat, layer=self.layer)
						output = self._prep_output(output)
						feature = self._get_centroid(output.data) 
						if self.feature_dim is None:
							self.feature_dim = feature.size(0)
							features = torch.Tensor(length_iter*self.size_per_sample, self.feature_dim).cuda()
						features[length] = feature
						length += 1
					indices.append(length)
				else:
					output_data = []
					for data_feat in data:
						data_feat = Variable(data_feat).cuda()
						output = self.net.get_feature_output(data_feat, layer=self.layer)
						output = self._prep_output(output)
						output_data.append(output.data)
					output_data = torch.cat(output_data, dim=0)
					feature = self._get_centroid(output_data) 
					if self.feature_dim is None:
						self.feature_dim = feature.size(0)
						features = torch.Tensor(length_iter*self.size_per_sample, self.feature_dim).cuda()
					features[length:length+1] = feature
					length += 1
					indices.append(length)
			norm = torch.sqrt(torch.sum(features ** 2, dim=1))
			norm += 0.000001
			norm = norm.view(-1, 1)
			features /= norm
			if train:
				self.indices = indices
				self.labels_old = labels_old
				self.features = features[:length]
			else:
				self.indices_test = indices
				self.labels_old_test = labels_old
				self.features_test = features[:length]
		self.pool.cpu()
		self.net.cpu()
		t1_tot = time()
		print('total runtime generate %d, %d features: %f' %(self.indices[-1], self.indices_test[-1],
		 t1_tot-t0_tot))

	def _prep_output(self, output):
		if self.layer == 'fc6':
			return output
		elif self.layer == 'pool5':
			return output.view(1,-1)
		else:
			output = self.pool(output)
			# Currently we pool to 4x4 for VGG and Caffe
			return output.view(1,-1)

	def _generate_clusters(self):
		starting_indices = np.arange(self.indices[-1])
		np.random.shuffle(starting_indices)
		starting_indices = starting_indices[:self.num_clusters]
		cluster_centers = torch.Tensor(self.num_clusters, self.feature_dim).cuda()
		distances = torch.Tensor(self.indices[-1], self.num_clusters).cuda()
		distances_test = torch.Tensor(self.indices_test[-1], self.num_clusters).cuda()
		for i, index in enumerate(starting_indices):
			cluster_centers[i] = self.features[index]
		for iteration in range(self.kmeans_iterations):
			t0 = time()
			mean_dist = 0
			for i, cluster_center in enumerate(cluster_centers):
				distances[:,i] = self._get_distance(cluster_center, self.features)
				distances_test[:,i] = self._get_distance(cluster_center, self.features_test)
			dist, self.indices_min = torch.min(distances, dim=1)
			dist_test, self.indices_min_test = torch.min(distances_test, dim=1)
			dist_max, self.indices_new_centroids = torch.max(distances, dim=0)
			mean_dist = torch.mean(dist)
			mean_dist_test = torch.mean(dist_test)
			t1 = time()
			print('mean distance iteration %d, time %f: %f, %f' %(iteration,t1-t0, mean_dist, mean_dist_test))
			result = [('iteration', iteration), ('runtime', t1-t0), ('mean_distance', mean_dist),
				('mean_distance_test', mean_dist_test)]
			self._write_progress('cluster_'+self.name_labels, result)
			cluster_centers = self._get_cluster_centers()
			self._generate_labels()
			self._generate_labels_test()
		self._save_labels()
		self._save_labels_test()

	def _get_distance(self, cluster_center, features):
		if self.metric == 'euclidean':
			cluster_center = cluster_center.view(1, -1)
			return torch.sqrt(torch.sum((features - cluster_center) ** 2, dim=1))
		if self.metric == 'cosine':
			cluster_center = cluster_center.view(1, -1)
			return 1 - utils.cos_sim(features, cluster_center)
	
	def _get_centroid(self, output):
		feature = torch.sum(output, dim=0)
		norm = np.sqrt(torch.sum(feature**2))
		norm += 0.000001
		feature /= norm
		return feature

	def _generate_labels(self):
		self.labels = []
		for i in range(len(self.indices)-1):
			lower = self.indices[i]
			upper = self.indices[i+1]
			self.labels.append(self.indices_min[lower:upper].cpu().numpy())

	def _generate_labels_test(self):
		self.labels_test = []
		for i in range(len(self.indices_test)-1):
			lower = self.indices_test[i]
			upper = self.indices_test[i+1]
			self.labels_test.append(self.indices_min_test[lower:upper].cpu().numpy())

	def _save_features_indices(self):
		features = self.features.cpu()
		features_test = self.features_test.cpu()
		path_features = os.path.join(self.results_dir, 'features.pkl')
		torch.save(features, path_features)
		path_features_test = os.path.join(self.results_dir, 'features_test.pkl')
		torch.save(features_test, path_features_test)
		path_indices = os.path.join(self.results_dir, 'indices.pkl')
		pickle.dump(self.indices, open(path_indices, 'wb'))
		path_indices_test = os.path.join(self.results_dir, 'indices_test.pkl')
		pickle.dump(self.indices_test, open(path_indices_test, 'wb'))

	def _save_labels_old(self):
		path_labels_old = os.path.join(self.results_dir, 'labels_old.pkl')
		pickle.dump(self.labels_old, open(path_labels_old, 'wb'))
		path_labels_old_test = os.path.join(self.results_dir, 'labels_old_test.pkl')
		pickle.dump(self.labels_old_test, open(path_labels_old_test, 'wb'))

	def _save_labels(self, iteration=-1):
		labels_splitted = []
		counter = 0
		self._reconfigure_dataloader_cluster(train=True)
		for i in range(len(self.dataset_info_lengths)):
			lower = counter
			counter += self.dataset_info_lengths[i]
			upper = counter
			labels_splitted.append(self.labels[lower:upper])
		path_labels = os.path.join(self.results_dir, '%s.pkl' %self.name_labels)
		pickle.dump(labels_splitted, open(path_labels, 'wb'))

	def _save_labels_test(self, iteration=-1):
		labels_splitted = []
		counter = 0
		self._reconfigure_dataloader_cluster(train=False)
		for i in range(len(self.dataset_info_lengths)):
			lower = counter
			counter += self.dataset_info_lengths[i]
			upper = counter
			labels_splitted.append(self.labels_test[lower:upper])
		path_labels_test = os.path.join(self.results_dir, '%s_test.pkl' %self.name_labels)
		pickle.dump(labels_splitted, open(path_labels_test, 'wb'))

	def _get_cluster_centers(self):
		cluster_centers = torch.Tensor(self.num_clusters, self.feature_dim).zero_().cuda()
		num_features = torch.Tensor(self.num_clusters).zero_()
		for i, index_min in enumerate(self.indices_min):
			cluster_centers[index_min] += self.features[i]
			num_features[index_min] += 1
		for i in range(self.num_clusters):
			if float(num_features[i]) == 0:
				cluster_centers[i] = self.features[self.indices_new_centroids[i]]
			norm = torch.sqrt(torch.sum(cluster_centers**2, dim=1))
			norm += 0.000001
			norm = norm.view(-1, 1)
		cluster_centers /= norm
		return cluster_centers

	def _get_data(self, iterator):
		images, labels = next(iterator, (None, None))
		if images is None:
			return None, None
		images = torch.cat(images, dim=0)
		num_features = int(images.size(0) / self.num_crops)
		data = []
		for i in range(num_features):
			data.append(images[i*self.num_crops:(i+1)*self.num_crops])
		return data, labels

	def _set_dataloader(self, dataset):
		self.dataloader = data.DataLoader(dataset, batch_size=1, num_workers=20, 
			shuffle=False, drop_last=True)

class Cluster_OF(Base_Cluster):
	def _reconfigure_dataloader_cluster(self, train=True):
		crop = transforms.TenCrop(self.net.input_spatial_size)
		if self.use_cc:
			crop = transforms.CenterCrop(self.net.input_spatial_size)
		transform_of = transforms.Compose([
			transforms.Scale(256), 
			crop,
			transforms.ToTensor(),
			transforms.SubMeanDisplacement()])
		dataset_infos = []
		self.dataset_info_lengths = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=train, source=self.source, num_frames=self.num_frames)
			dataset_infos.append(dataset_info)
			self.dataset_info_lengths.append(len(dataset_info))
		dataset = Dataset_OF(infos=dataset_infos, train=False, 
		  transform=transform_of, num_frames=self.num_frames, num_test=self.num_test)
		self._reconfigure_dataloader(dataset, 1, shuffle=False)

	def _load_pretraining(self):
		results_dir_pt = os.path.join(self.results_path, self.name, self.name_experiment)
		path_info = os.path.join(results_dir_pt, 'net_info.pkl')
		net_pt = get_network(path_info)
		if self.load_epoch_pt != -2:
			path_params = os.path.join(results_dir_pt, 'net_%i.pkl' %(self.load_epoch_pt))
			new_sd = torch.load(path_params)
			utils.load_sd(net_pt, new_sd)
		if hasattr(net_pt, 'mot_net'):
			self.net = net_pt.mot_net
		else:
			self.net = net_pt

class Cluster_RGB(Base_Cluster):
	def _reconfigure_dataloader_cluster(self, train=True):
		crop = transforms.TenCrop(self.net.input_spatial_size)
		color = transforms.CenterCrop(self.net.input_spatial_size)
		if self.use_cc:
			crop = transforms.CenterCrop(self.net.input_spatial_size)
		if self.use_splitchannels:
			color = transforms.SplitChannels(use_rand=False)
		transform_rgb = transforms.Compose([
			transforms.Scale(256), 
			crop,
			color,
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)])
		dataset_infos = []
		self.dataset_info_lengths = []
		for dataset_info_type in self.dataset_info_types:
			dataset_info = dataset_info_type(train=train, source=self.source, num_frames=self.num_frames)
			dataset_infos.append(dataset_info)
			self.dataset_info_lengths.append(len(dataset_info))
		dataset = Dataset_RGB(infos=dataset_infos, train=False, 
		  transform=transform_rgb, num_test=self.num_test)
		self._reconfigure_dataloader(dataset, 1, shuffle=False)

	def _load_pretraining(self):
		results_dir_pt = os.path.join(self.results_path, self.name, self.name_experiment)
		path_info = os.path.join(results_dir_pt, 'net_info.pkl')
		net_pt = get_network(path_info)
		if self.load_epoch_pt != -2:
			path_params = os.path.join(results_dir_pt, 'net_%i.pkl' %(self.load_epoch_pt))
			new_sd = torch.load(path_params)
			utils.load_sd(net_pt, new_sd)
		if hasattr(net_pt, 'app_net'):
			self.net = net_pt.app_net	
		else:
			self.net = net_pt