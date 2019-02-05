import torch.utils.data as data_old
import os
import numpy as np
import _pickle as pickle
from PIL import Image
import torch
import os

from compvis import transforms_det as transforms

from time import time

from io import BytesIO
import compvis.data as data

__all__ = ['UCF101_2', 'HMDB51_2', 'ACT_2', 'Videos_2']

# This is suited for optical_flow_2, task is considerably more difficult

def prep_image(path):
	img = Image.open(path)
	img.load()
	w, h = img.size
  
	if 3*w < 4*h:
		wp = 320
		hp = int(320 * h / w)
	else:
		hp = 240
		wp = int(240 * w / h)
      
	img = img.resize((wp, hp), resample=Image.BILINEAR)

	w, h = wp, hp
	tw, th = 320, 240
	x1 = int(round((w - tw) / 2.))
	y1 = int(round((h - th) / 2.))
	img = img.crop((x1, y1, x1 + tw, y1 + th))
	return img

class Base_OF_fast_2(data.Dataset):
	# functions:
	# getitem(info, dict_num)
	# preprocess(transform)
	# memory check
	# 
	# attributes:
	# info, memory_usage

	data_cache = None
	
	# path_data should be already set 
	def __init__(self, transform=None):
		self.transform = transform
		self.dict_num = pickle.load(open(os.path.join(self.path_data, 'dict_num.pkl'), 'rb'))
		self.dict_norm = pickle.load(open(os.path.join(self.path_data, 'dict_norm.pkl'), 'rb'))

	def preprocess(self, raw):
		buffers_norms, paths_norms = raw
		random = np.random.RandomState()
		randomstate = random.get_state() 

		images_norms = []
		for path, norm in paths_norms:
			images_norms.append((prep_image(path), norm))
		for buf, norm in buffers_norms:
			buf.seek(0)
			img = Image.open(buf)
			img.load()
			images_norms.append((img, norm))

		images = []
		for img, norm in images_norms:
			if self.transform is not None:
				random.set_state(randomstate)
				img = self.transform(img, random)
				img *= norm
			images.append(img)

		image = images.pop(0)

		if self.transform is None:
			flow = images
		else:
			flow = torch.cat(images, 0)

		return image, flow	

	# getitem returns buffers and paths
	def __getitem__(self, index):
		item_name = self.info[index]
		item_folder = os.path.join(self.path_data, item_name) 
		num_total = self.dict_num[item_name]

		# This is for pretraining the triplet siamese network
		random = np.random.RandomState()
		num_frame_first = random.randint(1, num_total-11)

		num_frame = num_frame_first + random.randint(0, 13)
		name_frame = item_name + '_' + str(num_frame) + '.jpg'
		path_frame = os.path.join(item_folder, name_frame)
		paths_norms = [(path_frame, 1)]

		buffers_norms = []
		for i in range(12):
			for v in ['_x_', '_y_']:
				name = item_name + v + str(num_frame_first + i) + '.jpg'
				buf = self._get_buf(name)
				norm = self.dict_norm[item_name + '_' + str(num_frame_first + i)]
				buffers_norms.append((buf, norm))

		return buffers_norms, paths_norms

	def _get_buf(self, name):
		if self.data_cache is None:
			self._memory_check()
			self.data_cache = pickle.load(open(os.path.join(self.path_data, 'dict_data.pkl'), 'rb'))
		return self.data_cache[name]

	def _memory_check(self):
		f = open('/proc/meminfo','rb')
		line = f.readlines()[2].decode('UTF-8')
		available_memory = int(line[16:-4])
		if self.memory_usage > available_memory:
			raise MemoryError('Base_OF dataset might run out of memory')

	@property
	def path_data(self):
		raise NotImplementedError('Base_OF dataset should implement path_data')

	@property
	def info(self):
		raise NotImplementedError('Base_OF dataset should implement info containing index: item_name')

	@property
	def memory_usage(self):
		raise NotImplementedError('Base_OF dataset should implement memory_usage')

# This is rather slow
class Base_OF_slow_2(data.Dataset):
	# functions:
	# getitem(info, dict_num)
	# preprocess(transform)
	# memory check
	# 
	# attributes:
	# info, memory_usage

	data_cache = None
	
	# path_data should be already set 
	def __init__(self, transform=None):
		self.transform = transform
		self.dict_num = pickle.load(open(os.path.join(self.path_data, 'dict_num.pkl'), 'rb'))
		self.dict_norm = pickle.load(open(os.path.join(self.path_data, 'dict_norm.pkl'), 'rb'))

	def preprocess(self, raw):
		paths_norms = raw
		random = np.random.RandomState()
		randomstate = random.get_state() 

		images_norms = []
		for path, norm in paths_norms:
			images_norms.append((prep_image(path), norm))

		images = []
		for img, norm in images_norms:
			if self.transform is not None:
				random.set_state(randomstate)
				img = self.transform(img, random)
				img *= norm
			images.append(img)

		image_first = images.pop(0)
		image_last = images.pop(0)

		if self.transform is None:
			flow = images
		else:
			flow = torch.cat(images, 0)
			# Normalize the flow, so that 0 equals no motion

		return image_first, image_last, flow	

	# getitem returns buffers and paths
	def __getitem__(self, index):
		item_name = self.info[index]
		item_folder_img = os.path.join(self.path_data, item_name) 
		item_folder_flow = os.path.join(self.path_data, item_name + '_flow')
		num_total = self.dict_num[item_name]

		# This is for pretraining the triplet siamese network
		random = np.random.RandomState()
		num_frame_first = random.randint(1, num_total-11)
		num_frame_last = num_frame_first + 12
		name_frame_first = item_name + '_' + str(num_frame_first) + '.png'
		name_frame_last = item_name + '_' + str(num_frame_last) + '.png'
		path_frame_first = os.path.join(item_folder_img, name_frame_first)
		path_frame_last = os.path.join(item_folder_img, name_frame_last)
		paths_norms = [(path_frame_first, 1), (path_frame_last, 1)]
		for i in range(12):
			for v in ['_x_', '_y_']:
				name = item_name + v + str(num_frame_first + i) + '.png'
				path = os.path.join(item_folder_flow, name)
				norm = self.dict_norm[item_name + '_' + str(num_frame_first + i)]
				paths_norms.append((path, norm))
		return paths_norms

	@property
	def path_data(self):
		raise NotImplementedError('Base_OF dataset should implement path_data')

	@property
	def info(self):
		raise NotImplementedError('Base_OF dataset should implement info containing index: item_name')

class UCF101_2(Base_OF_fast_2): 
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/UCF101/images'
	path_infos = '/export/home/nsayed/data/UCF101'
	memory_usage = 20000000
	info = None

	def __init__(self, split=1, train=True, transform=None):
		self.split = split
		self.train = train
		super(UCF101_2, self).__init__(transform=transform)
		self._set_info()

		# TODO remove
		# path_dict = os.path.join(self.path_infos, "dict.pkl")

	def __len__(self):
		return self.len

	def _set_info(self):
		if self.train:
			info_name = 'trainlist0' + str(self.split) + '.txt'
		else:
			info_name = 'testlist0' + str(self.split) + '.txt'

		raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)
		dict_names = pickle.load(open(os.path.join(self.path_data, 'dict_names_new.pkl'), 'rb'))

		self.info = []
		for i in range(raw_info.shape[0]):
			if self.train:
				name = (raw_info[i][0][:-4]).decode('UTF-8')
			else:
				name = (raw_info[i][:-4]).decode('UTF-8')
			drop_first = int((len(name) - 11) / 2) + 1
			name = name[drop_first:]
			self.info.append(dict_names[name + '.avi'])
		self.info = np.array(self.info)
		self.len = len(self.info)



class HMDB51_2(Base_OF_fast_2):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/HMDB51/images'
	path_infos = '/net/hci-storage01/groupfolders/compvis/nsayed/data/HMDB51/testTrainMulti_7030_splits'
	memory_usage = 8000000
	info = None

	def __init__(self, split=1, train=True, transform=None):
		self.split = split
		self.train = train
		super(HMDB51_2, self).__init__(transform=transform)
		self._set_info()

	def __len__(self):
		return self.len

	def _set_info(self):
		if self.train:
			required_mark = 1
		else:
			required_mark = 2
		dict_names = pickle.load(open(os.path.join(self.path_data, 'dict_names_new.pkl'), 'rb'))
		self.info = []
		for root, dirs, files in os.walk(self.path_infos):
			for file in files:
				if file.endswith('split%i.txt' %self.split):					
					raw_info = np.genfromtxt(os.path.join(self.path_infos, file), dtype=None, comments=None)
					for i in range(raw_info.shape[0]):	
						path_name = (raw_info[i][0]).decode('UTF-8')
						item_name = dict_names[path_name]
						item_num = self.dict_num[item_name]
						mark = raw_info[i][1]
						if mark == required_mark and item_num > 12:
							self.info.append(item_name)

		self.info = np.array(self.info)
		self.len = len(self.info)


class ACT_2(Base_OF_fast_2):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/ACT/images'
	path_infos = '/net/hci-storage01/groupfolders/compvis/nsayed/data/ACT/labels/task1'
	memory_usage = 15000000
	info = None

	def __init__(self, train=True, transform=None):
		self.train = train
		super(ACT_2, self).__init__(transform=transform)
		self._set_info()

	def __len__(self):
		return self.len

	def _set_info(self):
		if self.train:
			info_name = 'trainlist.txt'
		else:
			info_name = 'testlist.txt'

		dict_names = pickle.load(open(os.path.join(self.path_data, 'dict_names_new.pkl'), 'rb'))
		self.info = []
		raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None, comments=None)
		for i in range(raw_info.shape[0]):	
			path_name = (raw_info[i][0][:-1]).decode('UTF-8')
			item_name = dict_names[path_name + '.avi']
			item_num = self.dict_num[item_name]
			if item_num > 12:
				self.info.append(item_name)

		self.info = np.array(self.info)
		self.len = len(self.info)

class Videos_2(data.Dataset):
	data_cache = None
	memory_usage = 30000000

	def __init__(self, train=True, transform=None):
		self.train = train
		self.transform = transform
		self.datasets = [UCF101_2(split=1, train=self.train, transform=self.transform),
										ACT_2(train=self.train, transform=self.transform),
										HMDB51_2(split=1, train=self.train, transform=self.transform)]
		self.len = 0
		for dataset in self.datasets:
			self.len += len(dataset)

		self.preprocess = self.datasets[0].preprocess

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		for dataset in self.datasets:
			length = len(dataset)
			if length <= index:
				index -= length
			else:
				return dataset[index]

	def preprocess(self, raw):
		return self.preprocess(raw)

	def _memory_check(self):
		f = open('/proc/meminfo','rb')
		line = f.readlines()[2].decode('UTF-8')
		available_memory = int(line[16:-4])
		if self.memory_usage > available_memory:
			raise MemoryError('Videos_2 dataset might run out of memory')

	@property
	def data_cache(self):
		return [dataset.data_cache for dataset in self.datasets]

	@data_cache.setter
	def data_cache(self, cache):
		for i in range(3):
			self.datasets[i].data_cache = cache[i]

