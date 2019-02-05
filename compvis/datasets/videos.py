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

# __all__ = ['UCF101', 'HMDB51', 'ACT', 'Videos', 'UCF101_ar', 'HMDB51_ar']
__all__ = ['UCF101_ar', 'HMDB51_ar']
# TODO unify UCF101 and UCF101_ar into one class to have consistent logic
# TODO implement smaller versions of the data_cahe dictionary for quick testing purposes

# UCF101 50Gb













class Base_OF_fm(data.Dataset):
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
		self.dict_num = pickle.load(open(os.path.join(self.path_data, 'dict_num_s.pkl'), 'rb'))
		self.dict_norm = pickle.load(open(os.path.join(self.path_data, 'dict_norm_s.pkl'), 'rb'))

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
		num_frame_first = random.randint(1, 1+num_total-N_items)

		num_frame = num_frame_first + random.randint(0, N_items+1)
		name_frame = item_name + '_' + str(num_frame) + '.jpg'
		path_frame = os.path.join(item_folder, name_frame)
		paths_norms = [(path_frame, 1)]

		buffers_norms = []
		for i in range(N_items):
			for v in ['_x_', '_y_']:
				name = item_name + v + str(num_frame_first + i) + '.jpg'
				buf = self._get_buf(name)
				norm = self.dict_norm[item_name + '_' + str(num_frame_first + i)]
				buffers_norms.append((buf, norm))

		return buffers_norms, paths_norms

	def _get_buf(self, name):
		if self.data_cache is None:
			self._memory_check()
			self.data_cache = pickle.load(open(os.path.join(self.path_data, 'dict_data_s.pkl'), 'rb'))
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

# This is rather fast
class Base_OF_fast(data.Dataset):
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

		image_first = images.pop(0)
		image_last = images.pop(0)

		image_first, image_last, images = self.randtimeflip(image_first, image_last, images)

		if self.transform is None:
			flow = images
		else:
			flow = torch.cat(images, 0)

		return image_first, image_last, flow	

	def randtimeflip(self, image_first, image_last, flow):
		if np.random.rand() < 0.5:		
			flow_rev = []
			for f in flow:
				f = -f
				flow_rev.insert(0, f)
			return image_last, image_first, flow_rev
		return image_first, image_last, flow

	# getitem returns buffers and paths
	def __getitem__(self, index):
		item_name = self.info[index]
		item_folder = os.path.join(self.path_data, item_name) 
		num_total = self.dict_num[item_name]

		# This is for pretraining the triplet siamese network
		random = np.random.RandomState()
		num_frame_first = random.randint(1, num_total-11)
		num_frame_last = num_frame_first + 12
		name_frame_first = item_name + '_' + str(num_frame_first) + '.jpg'
		name_frame_last = item_name + '_' + str(num_frame_last) + '.jpg'
		path_frame_first = os.path.join(item_folder, name_frame_first)
		path_frame_last = os.path.join(item_folder, name_frame_last)
		paths_norms = [(path_frame_first, 1), (path_frame_last, 1)]

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
class Base_OF_slow(data.Dataset):
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

class UCF101(Base_OF_fast): 
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/UCF101/images'
	path_infos = '/export/home/nsayed/data/UCF101'
	memory_usage = 20000000
	info = None

	def __init__(self, split=1, train=True, transform=None):
		self.split = split
		self.train = train
		super(UCF101, self).__init__(transform=transform)
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



class HMDB51(Base_OF_fast):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/HMDB51/images'
	path_infos = '/net/hci-storage01/groupfolders/compvis/nsayed/data/HMDB51/testTrainMulti_7030_splits'
	memory_usage = 8000000
	info = None

	def __init__(self, split=1, train=True, transform=None):
		self.split = split
		self.train = train
		super(HMDB51, self).__init__(transform=transform)
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


class ACT(Base_OF_fast):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/ACT/images'
	path_infos = '/net/hci-storage01/groupfolders/compvis/nsayed/data/ACT/labels/task1'
	memory_usage = 15000000
	info = None

	def __init__(self, train=True, transform=None):
		self.train = train
		super(ACT, self).__init__(transform=transform)
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

class Videos(data.Dataset):
	data_cache = None
	memory_usage = 30000000

	def __init__(self, train=True, transform=None):
		self.train = train
		self.transform = transform
		self.datasets = [UCF101(split=1, train=self.train, transform=self.transform),
										ACT(train=self.train, transform=self.transform),
										HMDB51(split=1, train=self.train, transform=self.transform)]
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
			raise MemoryError('Videos dataset might run out of memory')

	@property
	def data_cache(self):
		return [dataset.data_cache for dataset in self.datasets]

	@data_cache.setter
	def data_cache(self, cache):
		for i in range(3):
			self.datasets[i].data_cache = cache[i]


class Base_ar(data.Dataset):
	data_cache = None		

	def __init__(self, train=True, transform=None, num_frames=5):
		self.train = train
		self.transform = transform
		self.num_frames = num_frames
		self.dict_num = pickle.load(open(os.path.join(self.path_data, 'dict_num_l.pkl'), 'rb'))

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		item_name = self.info[index]
		item_folder = os.path.join(self.path_data, item_name) 
		num_total = self.dict_num[item_name]

		if self.train:
			random = np.random.RandomState()
			num_frames = np.array([random.randint(1, num_total+1)]).astype(np.int32)
		else:
			num_frames = np.linspace(1, num_total, self.num_frames).astype(np.int32)
		path_frames = []
		for num_frame in num_frames:
			name_frame = item_name + '_' + str(num_frame) + '.jpg'	
			path_frame = os.path.join(item_folder, name_frame)
			path_frames.append(path_frame)
		return path_frames, self.dict_label[item_name]			

	# returns a list of transformed images and a equally long list of labels
	def preprocess(self, raw):
		paths, label = raw
		random = np.random.RandomState()
		randomstate = random.get_state() 
		images_ = []
		for path in paths:
			images_.append(prep_image(path))
		images = []
		for img in images_:
			if self.transform is not None:
				random.set_state(randomstate)
				img = self.transform(img, random)
			if self.train:
				images.append(img)
			else:
				images += img
		num = len(images)
		label = num * [label] 
		return images, label

def set_num_frames_test(num):
	self.num_frames = num


class UCF101_ar(Base_ar):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/UCF101/images'
	path_infos = '/export/home/nsayed/data/UCF101'
	data_cache = None	

	def __init__(self, split=1, train=True, transform=None, num_frames=5):
		super(UCF101_ar, self).__init__(train=train, 
			transform=transform, num_frames=num_frames)
		self.split = split

		dict_names = pickle.load(open(os.path.join(self.path_data, 'dict_names_l_new.pkl'), 'rb'))

		self.dict_label = {}
		for split in [1,2,3]:
			info_name = 'trainlist0' + str(split) + '.txt'
			raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)		
			for i in range(raw_info.shape[0]):
				name = (raw_info[i][0][:-4]).decode('UTF-8')
				label = raw_info[i][1] - 1
				drop_first = int((len(name) - 11) / 2) + 1
				name = name[drop_first:]
				name = dict_names[name + '.avi']
				self.dict_label[name] = label


		self._set_info()

	def _set_info(self):
		if self.train:
			info_name = 'trainlist0' + str(self.split) + '.txt'
		else:
			info_name = 'testlist0' + str(self.split) + '.txt'

		raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)
		dict_names = pickle.load(open(os.path.join(self.path_data, 'dict_names_l_new.pkl'), 'rb'))

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


	# # returns a list of paths and the according label
	# def __getitem__(self, index):
	# 	item_name = self.info[index]
	# 	item_folder = os.path.join(self.path_data, item_name) 
	# 	num_total = self.dict_num[item_name]

	# 	if self.train:
	# 		random = np.random.RandomState()
	# 		num_frames = np.array([random.randint(1, num_total+1)]).astype(np.int32)
	# 	else:
	# 		num_frames = np.linspace(1, num_total, 20).astype(np.int32)
	# 	path_frames = []
	# 	for num_frame in num_frames:
	# 		name_frame = item_name + '_' + str(num_frame) + '.jpg'	
	# 		path_frame = os.path.join(item_folder, name_frame)
	# 		path_frames.append(path_frame)
	# 	return path_frames, self.dict_label[item_name]			

	# # returns a list of transformed images and a equally long list of labels
	# def preprocess(self, raw):
	# 	paths, label = raw
	# 	random = np.random.RandomState()
	# 	randomstate = random.get_state() 
	# 	images_ = []
	# 	for path in paths:
	# 		images_.append(prep_image(path))
	# 	images = []
	# 	for img in images_:
	# 		if self.transform is not None:
	# 			random.set_state(randomstate)
	# 			img = self.transform(img, random)
	# 		if self.train:
	# 			images.append(img)
	# 		else:
	# 			images += img
	# 	num = len(images)
	# 	label = num * [label] 
	# 	return images, label

class HMDB51_ar(Base_ar):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/HMDB51/images'
	path_infos = '/net/hci-storage01/groupfolders/compvis/nsayed/data/HMDB51/testTrainMulti_7030_splits'
	data_cache = None	

	def __init__(self, split=1, train=True, transform=None, num_frames=5):
		super(HMDB51_ar, self).__init__(train=train, 
			transform=transform, num_frames=num_frames)
		self.split = split
		self._set_info()

	def _set_info(self):
		self.dict_label = {}
		if self.train:
			required_mark = 1
		else:
			required_mark = 2
		dict_names = pickle.load(open(os.path.join(self.path_data, 'dict_names_new.pkl'), 'rb'))
		self.info = []
		cur_label = 0
		for root, dirs, files in os.walk(self.path_infos):
			files.sort()
			for file in files:
				if file.endswith('split1.txt'):					
					raw_info = np.genfromtxt(os.path.join(self.path_infos, file), dtype=None, comments=None)
					for i in range(raw_info.shape[0]):	
						path_name = (raw_info[i][0]).decode('UTF-8')
						item_name = dict_names[path_name]
						self.dict_label[item_name] = cur_label
						item_num = self.dict_num[item_name]
						mark = raw_info[i][1]
						if mark == required_mark and item_num > 12:
							self.info.append(item_name)
					cur_label += 1

		self.info = np.array(self.info)
		self.len = len(self.info)

	# # returns a list of paths and the according label
	# def __getitem__(self, index):
	# 	item_name = self.info[index]
	# 	item_folder = os.path.join(self.path_data, item_name) 
	# 	num_total = self.dict_num[item_name]

	# 	if self.train:
	# 		random = np.random.RandomState()
	# 		num_frames = np.array([random.randint(1, num_total+1)]).astype(np.int32)
	# 	else:
	# 		num_frames = np.linspace(1, num_total, 8).astype(np.int32)
	# 	path_frames = []
	# 	for num_frame in num_frames:
	# 		name_frame = item_name + '_' + str(num_frame) + '.jpg'	
	# 		path_frame = os.path.join(item_folder, name_frame)
	# 		path_frames.append(path_frame)
	# 	return path_frames, self.dict_label[item_name]			

	# # returns a list of transformed images and a equally long list of labels
	# def preprocess(self, raw):
	# 	paths, label = raw
	# 	random = np.random.RandomState()
	# 	randomstate = random.get_state() 
	# 	images_ = []
	# 	for path in paths:
	# 		images_.append(prep_image(path))
	# 	images = []
	# 	for img in images_:
	# 		if self.transform is not None:
	# 			random.set_state(randomstate)
	# 			img = self.transform(img, random)
	# 		if self.train:
	# 			images.append(img)
	# 		else:
	# 			images += img
	# 	num = len(images)
	# 	label = num * [label] 
	# 	return images, label


# class UCF101_ar(data.Dataset): 
# 	path_data = '/net/hci-storage01/groupfolders/compvis/nsayed/data/UCF101/data/data'
# 	path_infos = '/export/home/nsayed/data/UCF101'

# 	data_cache = None	

# 	def __init__(self, split=1, train=True, transform=None):
# 		self.split = split
# 		self.train = train
# 		self.transform = transform

# 		# Getting the labels for all samples
# 		self.dict_label = {}
# 		for split in [1,2,3]:
# 			info_name = 'trainlist0' + str(split) + '.txt'
# 			raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)		
# 			for i in range(raw_info.shape[0]):
# 				name = (raw_info[i][0][:-4]).decode('UTF-8')
# 				label = raw_info[i][1] - 1
# 				drop_first = int((len(name) - 11) / 2) + 1
# 				name = name[drop_first:]
# 				self.dict_label[name] = label

# 		self._set_info()
# 		self.dict_num = pickle.load(open(os.path.join(self.path_data, 'dict_num.pkl'), 'rb'))

# 	def __len__(self):
# 		return self.len

# 	# returns a list of paths and the according label
# 	def __getitem__(self, index):
# 		item_name = self.info[index]
# 		item_folder = os.path.join(self.path_data, item_name) 
# 		num_total = self.dict_num[item_name]

# 		if self.train:
# 			random = np.random.RandomState()
# 			num_frames = np.array([random.randint(1, num_total+1)]).astype(np.int32)
# 		else:
# 			num_frames = np.linspace(1, num_total, 5).astype(np.int32)
# 		path_frames = []
# 		for num_frame in num_frames:
# 			name_frame = item_name + '_' + str(num_frame) + '.jpg'	
# 			path_frame = os.path.join(item_folder, name_frame)
# 			path_frames.append(path_frame)
# 		return path_frames, self.dict_label[item_name]			

# 	# returns a list of transformed images and a equally long list of labels
# 	def preprocess(self, raw):
# 		paths, label = raw
# 		random = np.random.RandomState()
# 		randomstate = random.get_state() 
# 		images_ = []
# 		for path in paths:
# 			images_.append(prep_image(path))
# 		images = []
# 		for img in images_:
# 			if self.transform is not None:
# 				random.set_state(randomstate)
# 				img_rand = self.transform(img, random)
# 				img, rand = img_rand
# 			if self.train:
# 				images.append(img)
# 			else:
# 				images += img
# 		num = len(images)
# 		label = num * [label] 
# 		if self.train:
# 			return images, label, rand
# 		return images, label

# This class is not ready to use
class UCF101_of(Base_OF_slow): 
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/UCF101/images'
	path_infos = '/export/home/nsayed/data/UCF101'
	memory_usage = 20000000
	info = None

	def __init__(self, split=1, train=True, transform=None):
		self.split = split
		self.train = train
		super(UCF101_of, self).__init__(transform=transform)

		# Getting the labels for all samples
		self.dict_label = {}
		dict_names = pickle.load(open(os.path.join(self.path_data, 'dict_names_new.pkl'), 'rb'))
		for split in [1,2,3]:
			info_name = 'trainlist0' + str(split) + '.txt'
			raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)		
			for i in range(raw_info.shape[0]):
				name = (raw_info[i][0][:-4]).decode('UTF-8')
				label = raw_info[i][1] - 1
				drop_first = int((len(name) - 11) / 2) + 1
				name = name[drop_first:]
				name = dict_names[name + '.avi']
				self.dict_label[name] = label

		self._set_info()

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		item_name = self.info[index]
		item_folder = os.path.join(self.path_data, item_name) 
		num_total = self.dict_num[item_name]

		# This is for pretraining the triplet siamese network
		random = np.random.RandomState()
		num_frame_first = random.randint(1, num_total-11)
		names = []
		for i in range(12):
			for v in ['_x_', '_y_']:
				name = item_name + v + str(num_frame_first + i) + '.jpg'
				names.append(name)

		buffers = []
		for name in names:
			buf = self._get_buf(name)
			buffers.append(buf)
		return buffers


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


# class UCF101(data.Dataset): 
# 	path_items = '/net/hci-storage01/groupfolders/compvis/nsayed/data/UCF101/data'
# 	path_infos = '/export/home/nsayed/data/UCF101'	
# 	data_cache = {}

# 	def __init__(self, split=1, train=True, transform=None):
# 		self.split = split
# 		self.train = train
# 		self.transform = transform

# 		self._set_info(train, split)

# 		path_dict = os.path.join(self.path_infos, "dict.pkl")
# 		self.dict_num = pickle.load(open(path_dict, "rb"))
		
# 		self.data_cache = {}

# 	def __len__(self):
# 		return self.len

# 	def __getitem__(self, index):
# 		item_name = self.info[index]
# 		item_folder = os.path.join(self.path_items, item_name)
# 		num_total = self.dict_num[item_name]

# 		random = np.random.RandomState()

# 		num_frame_first = random.randint(1, num_total-11)
# 		num_frame_last = num_frame_first + 12

# 		name_frame_first = item_name + '_' + str(num_frame_first) + '.jpg'
# 		name_frame_last = item_name + '_' + str(num_frame_last) + '.jpg'

# 		names = [name_frame_first, name_frame_last]

# 		for i in range(12):
# 			for v in ['_x_', '_y_']:
# 				name = item_name + v + str(num_frame_first + i) + '.jpg'
# 				names.append(name)
		
# 		buffers = []
# 		for name in names:
# 			buf = self._get_buf(name, item_folder)
# 			buffers.append(buf)
# 		return buffers

# 	def preprocess(self, raw):
# 		random = np.random.RandomState()
# 		randomstate = random.get_state() 
# 		images_ = []
# 		for buf in raw:
# 			buf.seek(0)
# 			img = Image.open(buf)
# 			img.load()
# 			images_.append(img)

# 		images = []
# 		for img in images_:
# 			if self.transform is not None:
# 				random.set_state(randomstate)
# 				img = self.transform(img, random)
# 			images.append(img)

# 		image_first = images.pop(0)
# 		image_last = images.pop(0)

# 		if self.transform is None:
# 			flow = images
# 		else:
# 			flow = torch.cat(images, 0)
# 			# Normalize the flow, so that 0 equals no motion
# 			flow = flow - 0.5

# 		return image_first, image_last, flow		

# 	def _set_info(self, train, split):
# 		if train:
# 			info_name = 'trainlist0' + str(split) + '.txt'
# 		else:
# 			info_name = 'testlist0' + str(split) + '.txt'

# 		raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)

# 		self.info = []
# 		for i in range(raw_info.shape[0]):
# 			if self.train:
# 				name = (raw_info[i][0][:-4]).decode('UTF-8')
# 			else:
# 				name = (raw_info[i][:-4]).decode('UTF-8')
# 			drop_first = int((len(name) - 11) / 2) + 1
# 			name = name[drop_first:]
# 			# v_PommelHorse_g05 differs in resolution from all other
# 			if name[:-4] != 'v_PommelHorse_g05':
# 				self.info.append(name)
# 		self.info = np.array(self.info)
# 		self.len = len(self.info)

# 	def _get_buf(self, name, item_folder):
# 		if name in self.data_cache:
# 			buf = self.data_cache[name]
# 			return buf

# 		else:
# 			path_item = os.path.join(item_folder, name)
# 			img = Image.open(path_item)
# 			img.load()
# 			buf = BytesIO()
# 			img.save(buf, "JPEG", quality=50)
# 			self.data_cache[name] = buf
# 			return buf




# class UCF101_ar(data.Dataset): 
# 	path_items = '/net/hci-storage01/groupfolders/compvis/nsayed/data/UCF101/data'
# 	path_infos = '/export/home/nsayed/data/UCF101'	

# 	def __init__(self, split=1, train=True, transform=None):
# 		self.split = split
# 		self.train = train
# 		self.transform = transform

# 		# if train:
# 		# 	info_name = 'trainlist0' + str(split) + '.txt'
# 		# else:
# 		# 	info_name = 'testlist0' + str(split) + '.txt'

# 		self.dict_label = {}

# 		for split in [1,2,3]:
# 			info_name = 'trainlist0' + str(split) + '.txt'
# 			raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)		

# 			for i in range(raw_info.shape[0]):
# 				name = (raw_info[i][0][:-4]).decode('UTF-8')
# 				label = raw_info[i][1] - 1
# 				drop_first = int((len(name) - 11) / 2) + 1
# 				name = name[drop_first:]
# 				# v_PommelHorse_g05 differs in resolution from all other videos
# 				if name[:-4] != 'v_PommelHorse_g05':
# 					self.dict_label[name] = label

# 		self._set_info(self.train, self.split)

# 		path_dict = os.path.join(self.path_infos, "dict.pkl")
# 		self.dict_num = pickle.load(open(path_dict, "rb"))
		
# 		self.data_cache = {}

# 	def __len__(self):
# 		return self.len

# 	def __getitem__(self, index):
# 		item_name = self.info[index]
# 		item_folder = os.path.join(self.path_items, item_name)
# 		num_total = self.dict_num[item_name]

# 		if self.train:
# 			random = np.random.RandomState()
# 			num_frame = random.randint(1, num_total+1)
# 			name_frame = item_name + '_' + str(num_frame) + '.jpg'	
# 			buf = self._get_buf(name_frame, item_folder)
# 			return buf, self.dict_label[item_name]
# 		else:
# 			buffers = []
# 			num_frames = np.linspace(1, num_total, 40).astype(np.int32)
# 			for num_frame in num_frames:
# 				name_frame = item_name + '_' + str(num_frame) + '.jpg'	
# 				buf = self._get_buf(name_frame, item_folder)
# 				buffers.append(buf)
# 			return buffers, self.dict_label[item_name]

# 	def preprocess(self, raw):
# 		random = np.random.RandomState()
# 		randomstate = random.get_state() 
# 		if self.train:
# 			buf, label = raw
# 			buf.seek(0)
# 			img = Image.open(buf)
# 			img.load()
# 			if self.transform is not None:
# 				random.set_state(randomstate)
# 				img = self.transform(img, random)
# 			return img, label
# 		else:
# 			buffers, label = raw
# 			tensors = []
# 			for buf in buffers:
# 				buf.seek(0)
# 				img = Image.open(buf)
# 				img.load()					
# 				if self.transform is not None:
# 					random.set_state(randomstate)
# 					img = self.transform(img, random)
# 					tensors = tensors + img
# 			num = len(tensors)
# 			labels = num * [label] 
# 			return tensors, labels

# 	def _set_info(self, train, split):
# 		if train:
# 			info_name = 'trainlist0' + str(split) + '.txt'
# 		else:
# 			info_name = 'testlist0' + str(split) + '.txt'

# 		raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)

# 		self.info = []
# 		for i in range(raw_info.shape[0]):
# 			if self.train:
# 				name = (raw_info[i][0][:-4]).decode('UTF-8')
# 			else:
# 				name = (raw_info[i][:-4]).decode('UTF-8')
# 			drop_first = int((len(name) - 11) / 2) + 1
# 			name = name[drop_first:]
# 			# v_PommelHorse_g05 differs in resolution from all other
# 			if name[:-4] != 'v_PommelHorse_g05':
# 				self.info.append(name)
# 		self.info = np.array(self.info)
# 		self.len = len(self.info)

# 	def _get_buf(self, name, item_folder):
# 		if name in self.data_cache:
# 			buf = self.data_cache[name]
# 			return buf

# 		else:
# 			path_item = os.path.join(item_folder, name)
# 			img = Image.open(path_item)
# 			img.load()
# 			buf = BytesIO()
# 			img.save(buf, "JPEG", quality=50)
# 			self.data_cache[name] = buf
# 			return buf

# # class Prep(data.Preprocess):

# # 	def __init__(self, transform=None):
# # 		self.transform = transform

# # 	def __call__(self, raw):
# # 		random = np.random.RandomState()
# # 		randomstate = random.get_state() 
# # 		images_ = []
# # 		for buf in raw:
# # 			buf.seek(0)
# # 			img = Image.open(buf)
# # 			img.load()
# # 			images_.append(img)

# # 		images = []
# # 		for img in images_:
# # 			if self.transform is not None:
# # 				random.set_state(randomstate)
# # 				img = self.transform(img, random)
# # 			images.append(img)

# # 		image_first = images.pop(0)
# # 		image_last = images.pop(0)

# # 		if self.transform is None:
# # 			flow = images
# # 		else:
# # 			flow = torch.cat(images, 0)
# # 			# Normalize the flow, so that 0 equals no motion
# # 			flow = flow - 0.5

# # 		return image_first, image_last, flow



















# class UCF101_old(data_old.Dataset):
# 	path_items = '/net/hci-storage01/groupfolders/compvis/nsayed/data/UCF101/data'
# 	path_infos = '/export/home/nsayed/data/UCF101'

# 	def __init__(self, split=1, train=True, transform=None):
# 		self.split = split
# 		self.train = train
# 		self.transform = transform

# 		path_dict = os.path.join(self.path_infos, "dict.pkl")
# 		self.dict_num = pickle.load(open(path_dict, "rb"))

# 		if train:
# 			info_name = 'trainlist0' + str(split) + '.txt'
# 		else:
# 			info_name = 'trainlist0' + str(split) + '.txt'

# 		raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)

# 		self.info = []
# 		for i in range(raw_info.shape[0]):
# 			name = (raw_info[i][0][:-4]).decode('UTF-8')
# 			drop_first = int((len(name) - 11) / 2) + 1
# 			name = name[drop_first:]
# 			self.info.append(name)
# 		self.info = np.array(self.info)
# 		self.len = len(self.info)
		
# 		self.buffer = BytesIO()
# 		self.ls = []
# 		self.dict_lul = {}

# 	def __len__(self):
# 		return self.len

# 	def __getitem__(self, index):

# 		t0 = time()
# 		item_name = self.info[index]
# 		item_folder = os.path.join(self.path_items, item_name)
# 		num_total = self.dict_num[item_name]

# 		random = np.random.RandomState()
# 		randomstate = random.get_state() 
		
# 		num_frame_first = random.randint(1, num_total-11)
# 		num_frame_last = num_frame_first + 12

# 		paths = []

# 		path_frame_first = os.path.join(item_folder, item_name + '_' + str(num_frame_first) + '.jpg')
# 		path_frame_last = os.path.join(item_folder, item_name + '_' + str(num_frame_last) + '.jpg')

# 		paths.append(path_frame_first)
# 		paths.append(path_frame_last)

# 		for i in range(12):
# 			for v in ['_x_', '_y_']:
# 			# path = os.path.join(item_folder, item_name + '_' + str(num_frame_first + i) + '.jpg')
# 				path = os.path.join(item_folder, item_name + v + str(num_frame_first + i) + '.jpg')
# 				paths.append(path)

# 		t_load = 0
# 		t_trans = 0

# 		images = []
# 		for path in paths:
# 			img = Image.open(path)
# 			img.load()
# 			if self.transform is not None:
# 				random.set_state(randomstate)
# 				img = self.transform(img, random)
# 			images.append(img)

# 		image_first = images.pop(0)
# 		image_last = images.pop(0)

# 		if self.transform is None:
# 			flow = images
# 		else:
# 			flow = torch.cat(images, 0)
			
# 		return image_first, image_last, flow


# 	# def __getitem__(self, index):
# 	# 	t0 = time()
# 	# 	item_name = self.info[index]
# 	# 	item_path = os.path.join(self.path_items, item_name)
# 	# 	f = h5py.File(item_path + '.hdf5','r')

# 	# 	t1 = time()

# 	# 	num_total = self.dict_num[item_name]

# 	# 	random = np.random.RandomState()

# 	# 	ind_start = random.randint(1, num_total-12)
# 	# 	ind_end = ind_start + 12

# 	# 	# item shape: 13 x 240 x 320 x 5
# 	# 	item = f[item_name][ind_start:ind_end + 1]

# 	# 	t2 = time()

# 	# 	images_arr = item[...,:3]
# 	# 	flow_arr = item[...,3:]

# 	# 	images = []

# 	# 	images.append(Image.fromarray(images_arr[0]))
# 	# 	images.append(Image.fromarray(images_arr[-1]))

# 	# 	for i in range(12):
# 	# 		for j in range(2):
# 	# 			images.append(Image.fromarray(flow_arr[i,:,:,j]))
		

# 	# 	randomstate = random.get_state() 
# 	# 	if self.transform is not None:
# 	# 		images_trans = []
# 	# 		for image in images:
# 	# 			random.set_state(randomstate)
# 	# 			image = self.transform(image, random)
# 	# 			images_trans.append(image)
# 	# 		images = images_trans

		

# 	# 	image_first = images.pop(0)
# 	# 	image_last = images.pop(0)

# 	# 	if self.transform is None:
# 	# 		flow = images
# 	# 	else:
# 	# 		flow = torch.cat(images, 0)

			

# 	# 	print(t1-t0, t2-t1)

# 	# 	return image_first, image_last, flow

# #### This one is works for farneback stuff with UCF folder not being numerated
# class UCF101(Base_OF): 
# 	path_data = '/net/hci-storage01/groupfolders/compvis/nsayed/data/UCF101/data/data'
# 	path_infos = '/export/home/nsayed/data/UCF101'
# 	memory_usage = 20000000
# 	info = None

# 	def __init__(self, split=1, train=True, transform=None):
# 		self.split = split
# 		self.train = train
# 		super(UCF101, self).__init__(transform=transform)

# 		self._set_info()

# 		path_dict = os.path.join(self.path_infos, "dict.pkl")

# 	def __len__(self):
# 		return self.len

# 	def _set_info(self):
# 		if self.train:
# 			info_name = 'trainlist0' + str(self.split) + '.txt'
# 		else:
# 			info_name = 'testlist0' + str(self.split) + '.txt'

# 		raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)

# 		self.info = []
# 		for i in range(raw_info.shape[0]):
# 			if self.train:
# 				name = (raw_info[i][0][:-4]).decode('UTF-8')
# 			else:
# 				name = (raw_info[i][:-4]).decode('UTF-8')
# 			drop_first = int((len(name) - 11) / 2) + 1
# 			name = name[drop_first:]
# 			self.info.append(name)
# 		self.info = np.array(self.info)
# 		self.len = len(self.info)

