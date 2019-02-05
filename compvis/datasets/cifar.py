from torchvision.datasets import CIFAR10 as CIFAR10_old
from torchvision.datasets import CIFAR100 as CIFAR100_old
from PIL import Image
import numpy as np
import os
import _pickle as pickle

# Although this might be ugly, but the code is supposed to be identicel for both classes
# with exception of the norm_file

class CIFAR10_aug(CIFAR10_old):
	root = '/net/hci-storage02/groupfolders/compvis/nsayed/data/'
	norm_file = '/export/home/nsayed/data/CIFAR/CIFAR10.pkl'
	num_classes = 10
	def __init__(self, train=True, 
		transform=None, download=False):
		super().__init__(self.root, train=train, transform=transform, 
			target_transform=None, download=download)

		if self.transform.mode == 'silent':
			raise Exception('Use transforms_det in nonsilent mode')

	def __getitem__(self, index):
		if self.train:
			img = self.train_data[index]
			target = self.train_labels[index]
		else:
			img = self.test_data[index]
			target = self.test_labels[index]			

		img = Image.fromarray(img)
		if self.transform is not None:
			img, rand = self.transform(img)
			return img, target, rand
		return img, target

	@classmethod
	def get_statistics(cls):
		return pickle.load(open(cls.norm_file, 'rb'))

class CIFAR100_aug(CIFAR100_old):
	root = '/net/hci-storage02/groupfolders/compvis/nsayed/data/'
	norm_file = '/export/home/nsayed/data/CIFAR/CIFAR10.pkl'
	num_classes = 100
	def __init__(self, train=True, 
		transform=None, download=False):
		super().__init__(self.root, train=train, transform=transform, 
			target_transform=None, download=download)
		
		if self.transform.mode == 'silent':
			raise Exception('Use transforms_det in nonsilent mode')

	def __getitem__(self, index):
		if self.train:
			img = self.train_data[index]
			target = self.train_labels[index]
		else:
			img = self.test_data[index]
			target = self.test_labels[index]			

		img = Image.fromarray(img)
		if self.transform is not None:
			img, rand = self.transform(img)
			return img, target, rand
		return img, target

	@classmethod
	def get_statistics(cls):
		return pickle.load(open(cls.norm_file, 'rb'))


class CIFAR10(CIFAR10_old):

	info_folder = '/export/home/nsayed/data/CIFAR/'
	norm_file = info_folder + 'CIFAR10.pkl'
	root = '/net/hci-storage01/groupfolders/compvis/nsayed/data/'
	
	def __init__(self, from_A=None, ratio_A=0.5, labels=np.arange(10),
		train=True, transform=None, target_transform=None, download=False):
		if train:
			assert from_A is not None, 'Specify the "from_A" keyword for the cifar training dataset'
		self.labels = set(labels)
		self.from_A = from_A

		super().__init__(self.root, train=train, transform=transform, 
			target_transform=target_transform, download=download)

		if self.train:
			self.images = self.train_data
			self.targets = self.train_labels

		else:
			self.images = self.test_data
			self.targets = self.test_labels

		self.indices = []

		for index in range(len(self.targets)):
			target = self.targets[index]
			if target in self.labels:
				self.indices.append(index)

		total_length = len(self.indices)

		if self.from_A:
			self.len = int(ratio_A*total_length)
			self.indices = np.array(self.indices[:self.len])
		else:
			self.len = total_length - int(ratio_A*total_length)
			self.indices = np.array(self.indices[-self.len:])

		print('Initialized dataset with %i images' %self.len)

	def __getitem__(self, index):
		index = self.indices[index]
		img, target = self.images[index], self.targets[index]
		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)
		# Uncomment if label usage in B should be explicitly frobidden 
		# if self.train:
		# 	if not self.from_A:
		# 		target = -1 

		return img, target

	def __len__(self):
		return self.len
	
	@classmethod
	def get_statistics(cls):
		C = 3
		# The statistics are calculated for images normalized to [0,1] on the entire training set


		# C: number of channels 
		# MAX: number of images taken into account
		# size: rescaled size of the images

		if os.path.exists(cls.norm_file):
			with open(cls.norm_file, 'rb') as f:
				stats = pickle.load(f) 
			return stats


		# If the statistics do not exist, create them
		else:
			trainset = CIFAR10(from_A=True)
			stats = np.zeros((C,2))
			data = trainset.train_data.astype(float)
			for i in range(C):	
				stats[i,0] = np.mean(data[:,:,:,i])
				stats[i,1] = np.mean(data[:,:,:,i]**2)
				stats[i,1] = np.sqrt(stats[i,1]-stats[i,0]**2)
			# array which stores the mean and stddev of the channels of all images up to that point		
			stats /= 255

			with open(cls.norm_file, 'wb') as f:
				pickle.dump(stats, f, -1)
			return stats

class CIFAR100(CIFAR100_old):

	info_folder = '/export/home/nsayed/data/CIFAR/'
	norm_file = info_folder + 'CIFAR100.pkl'
	root = '/net/hci-storage01/groupfolders/compvis/nsayed/data/'
	
	def __init__(self, from_A=None, ratio_A=0.5, labels=np.arange(100),
		train=True, transform=None, target_transform=None, download=False):
		if train:
			assert from_A is not None, 'specify the "from_A" keyword for the cifar training dataset'
		self.labels = set(labels)
		self.from_A = from_A

		super().__init__(self.root, train=train, transform=transform, 
			target_transform=target_transform, download=download)

		if self.train:
			self.images = self.train_data
			self.targets = self.train_labels

		else:
			self.images = self.test_data
			self.targets = self.test_labels

		self.indices = []

		for index in range(len(self.targets)):
			target = self.targets[index]
			if target in self.labels:
				self.indices.append(index)

		total_length = len(self.indices)

		if self.from_A:
			self.len = int(ratio_A*total_length)
			self.indices = np.array(self.indices[:self.len])
		else:
			self.len = total_length - int(ratio_A*total_length)
			self.indices = np.array(self.indices[-self.len:])

		print('Initialized dataset with %i images' %self.len)

	def __getitem__(self, index):
		index = self.indices[index]
		img, target = self.images[index], self.targets[index]
		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)
		# Uncomment if label usage in B should be explicitly frobidden 
		# if self.train:
		# 	if not self.from_A:
		# 		target = -1 

		return img, target

	def __len__(self):
		return self.len
	
	@classmethod
	def get_statistics(cls):
		C = 3
		# The statistics are calculated for images normalized to [0,1] on the entire training set


		# C: number of channels 
		# MAX: number of images taken into account
		# size: rescaled size of the images

		if os.path.exists(cls.norm_file):
			with open(cls.norm_file, 'rb') as f:
				stats = pickle.load(f) 
			return stats


		# If the statistics do not exist, create them
		else:
			trainset = CIFAR100(from_A=True)
			stats = np.zeros((C,2))
			data = trainset.train_data.astype(float)
			for i in range(C):	
				stats[i,0] = np.mean(data[:,:,:,i])
				stats[i,1] = np.mean(data[:,:,:,i]**2)
				stats[i,1] = np.sqrt(stats[i,1]-stats[i,0]**2)
			# array which stores the mean and stddev of the channels of all images up to that point		
			stats /= 255

			with open(cls.norm_file, 'wb') as f:
				pickle.dump(stats, f, -1)
			return stats

