import torch.utils.data as data_old
import os
import numpy as np
import _pickle as pickle
from PIL import Image
import torch
import os
from copy import deepcopy

from compvis import transforms_det as transforms

from time import time

from io import BytesIO
import compvis.data as data

__all__ = ['UCF101', 'HMDB51', 'ACT', 'Videos']

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

# TODO fix preprocess, implement ar_f, do not use get_buf for images

class Base_OF(data.Dataset):
	# functions:
	# getitem(info, dict_num)
	# preprocess(transform)
	# memory check
	# 
	# attributes:
	# info, memory_usage

	data_cache = {}
	
	# path_data should be already set 
	def __init__(self, train=True, transform=None, mode='def', num_frames=12, num_test=5, source='l', diff=4):	
		if transform is None:
			raise Exception('Transform is none in Base_OF')
		self.train = train
		self.transform = transform
		self.mode = mode
		if self.mode == 'fm_fc':
			self.mode = 'fm'
		self.num_frames = num_frames
		self.num_test = num_test
		self.source = source
		if self.mode == 'fm':
			self.diff = diff 
		self.dict_num = pickle.load(open(os.path.join(self.path_data, 'dict_num_'+self.source+'.pkl'), 'rb'))
		self.dict_norm = pickle.load(open(os.path.join(self.path_data, 'dict_norm_'+self.source+'.pkl'), 'rb'))
		self.data_cache = deepcopy(self.data_cache)

	def preprocess(self, raw):
		buffers_norms, paths_norms, label = raw
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
			random.set_state(randomstate)
			img = self.transform(img, random)
			
			# if not self.train and self.mode == 'ar':
			# 	images += img
			# else:
			# 	images.append(img)
			if not self.train and (self.mode == 'ar' or self.mode == 'ar_f'):
				for image in img:
					images.append(image * norm)
			else:
				img *= norm
				images.append(img)

		if self.mode == 'ar':
			return self._prep_ar(images, label)
		if self.mode == 'ar_f':
			return self._prep_ar_f(images, label)
		if self.mode == 'def':
			return self._prep_def(images, label)
		if self.mode == 'fm':
			return self._prep_fm(images, label)

	def _prep_ar(self, images, label):
		num = len(images)
		label = num * [label] 
		return images, label

	def _prep_ar_f(self, images, label):
		if self.train:
			flow = torch.cat(images, 0)
			return [flow], [label]
		else:
			flows = []
			for i in range(self.num_test):
				for j in range(10):
					flow = torch.cat(images[j::10][(2*self.num_frames)*i:(2*self.num_frames)*(i+1)], 0)
					flows.append(flow)
			num = len(flows)
			label = num * [label]
			return flows, label

	def _prep_def(self, images, label):	
		image_first = images.pop(0)
		image_last = images.pop(0)
		image_first, image_last, images = self._randtimeflip(image_first, image_last, images)	
		flow = torch.cat(images, 0)	
		return image_first, image_last, flow

	def _prep_fm(self, images, label):	
		image= images.pop(0)
		flow = torch.cat(images, 0)	
		return image, flow
		
	def _randtimeflip(self, image_first, image_last, flow):
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
		random = np.random.RandomState()
		ind_frame_first = random.randint(1, 1+num_total-self.num_frames)

		paths_norms = self._get_path_norms(item_name, item_folder, num_total, ind_frame_first, random)
		buffers_norms = self._get_buffers_norms(item_name, item_folder, num_total, ind_frame_first, random)
		return buffers_norms, paths_norms, self.dict_label[item_name]

	def _get_path_norms(self, item_name, item_folder, num_total, ind_frame_first, random):
		paths_norms = []
		if self.mode == 'def': 
			name_frame_first = item_name + '_' + str(ind_frame_first) + '.jpg'
			path_frame_first = os.path.join(item_folder, name_frame_first)		
			ind_frame_last = ind_frame_first + self.num_frames
			name_frame_last = item_name + '_' + str(ind_frame_last) + '.jpg'
			path_frame_last = os.path.join(item_folder, name_frame_last)
			paths_norms = [(path_frame_first, 1), (path_frame_last, 1)]
		elif self.mode == 'fm': 
			diff = int((self.diff - self.num_frames) / 2)
			lower = np.maximum(ind_frame_first-diff, 1)
			upper = np.minimum(ind_frame_first+self.num_frames+diff, num_total+1)
			ind_frame = random.randint(lower, upper)
			name_frame = item_name + '_' + str(ind_frame) + '.jpg'
			path_frame = os.path.join(item_folder, name_frame)
			paths_norms = [(path_frame, 1)]
		elif self.mode == 'ar':
			if self.train:
				ind_frames = np.array([random.randint(1, num_total+1)]).astype(np.int32)
			else:
				ind_frames = np.linspace(1, num_total, self.num_test).astype(np.int32)
			for ind_frame in ind_frames:
				name_frame = item_name + '_' + str(ind_frame) + '.jpg'	
				path_frame = os.path.join(item_folder, name_frame)
				paths_norms.append((path_frame, 1))
		return paths_norms

	def _get_buffers_norms(self, item_name, item_folder, num_total, ind_frame_first, random):
		buffers_norms = []
		if self.mode == 'def' or self.mode == 'fm':# or self.mode == 'ar_f':
			for i in range(self.num_frames):
				for v in ['_x_', '_y_']:
					name_frame = item_name + v + str(ind_frame_first + i) + '.jpg'
					buf = self._get_buf_flow(name_frame)
					norm = self.dict_norm[item_name + '_' + str(ind_frame_first + i)]
					buffers_norms.append((buf, norm))
		elif self.mode == 'ar_f':
			if self.train:
				ind_frames_first = np.array([random.randint(1, num_total+1-self.num_frames)]).astype(np.int32)
			else:
				ind_frames_first = np.linspace(1, num_total-self.num_frames, self.num_test).astype(np.int32)
			for ind_frame_first in ind_frames_first:
				for i in range(self.num_frames):
					for v in ['_x_', '_y_']:
						name_frame = item_name + v + str(ind_frame_first + i) + '.jpg'
						buf = self._get_buf_flow(name_frame)
						norm = self.dict_norm[item_name + '_' + str(ind_frame_first + i)]
						buffers_norms.append((buf, norm))

		return buffers_norms



	def _get_buf_flow(self, name):
		if not 'flow' in self.data_cache:
			self._memory_check()
			self.data_cache['flow'] = pickle.load(open(os.path.join(self.path_data, 'dict_data_'+self.source+'.pkl'), 'rb'))
		return self.data_cache['flow'][name]

	# def _get_buf_image(self, name):
	# 	if not 'image' in self.data_cache:
	# 		self._memory_check()
	# 		self.data_cache['image'] = pickle.load(open(os.path.join(self.path_data, 'dict_img_'+self.source+'.pkl'), 'rb'))
	# 	return self.data_cache['image'][name]

	def _memory_check(self):
		f = open('/proc/meminfo','rb')
		line = f.readlines()[2].decode('UTF-8')
		available_memory = int(line[16:-4])
		if self.memory_usage > available_memory:
			print('Base_OF dataset might run out of memory')

	@property
	def path_data(self):
		raise NotImplementedError('Base_OF dataset should implement path_data')

	@property
	def info(self):
		raise NotImplementedError('Base_OF dataset should implement info containing index: item_name')

	@property
	def dict_label(self):
		raise NotImplementedError('Base_OF dataset should implement dict_label')

	@property
	def memory_usage(self):
		raise NotImplementedError('Base_OF dataset should implement memory_usage')

class UCF101(Base_OF): 
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/UCF101/images'
	path_infos = '/export/home/nsayed/data/UCF101'
	memory_usage = 20000000
	info = None
	dict_label = None
	def __init__(self, split=1, train=True, transform=None, mode='def', num_frames=12, num_test=5, source='l', diff=4):
		self.split = split
		super(UCF101, self).__init__(train=train, transform=transform, mode=mode, num_frames=num_frames, num_test=num_test, source=source, diff=diff)
		self._set_info()

	def __len__(self):
		return self.len

	def _set_info(self):
		self.dict_label = {}
		dict_names = pickle.load(open(os.path.join(self.path_data, 'dict_names_'+self.source+'_new.pkl'), 'rb'))
		for split in [1,2,3]:
			info_name = 'trainlist0' + str(split) + '.txt'
			raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)		
			for i in range(raw_info.shape[0]):
				name = (raw_info[i][0][:-4]).decode('UTF-8')
				label = raw_info[i][1] - 1
				drop_first = int((len(name) - 11) / 2) + 1
				name = name[drop_first:]
				item_name = dict_names[name + '.avi']
				self.dict_label[item_name] = label

		self.info = []
		if self.train:
			info_name = 'trainlist0' + str(self.split) + '.txt'
		else:
			info_name = 'testlist0' + str(self.split) + '.txt'
		raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)
		for i in range(raw_info.shape[0]):
			if self.train:
				name = (raw_info[i][0][:-4]).decode('UTF-8')
			else:
				name = (raw_info[i][:-4]).decode('UTF-8')
			drop_first = int((len(name) - 11) / 2) + 1
			name = name[drop_first:]
			item_name = dict_names[name + '.avi']
			item_num = self.dict_num[item_name]
			if item_num > self.num_frames:
				self.info.append(item_name)
		self.info = np.array(self.info)
		self.len = len(self.info)

class HMDB51(Base_OF):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/HMDB51/images'
	path_infos = '/net/hci-storage01/groupfolders/compvis/nsayed/data/HMDB51/testTrainMulti_7030_splits'
	memory_usage = 8000000
	info = None
	dict_label = None
	def __init__(self, split=1, train=True, transform=None, mode='def', num_frames=12, num_test=5, source='l', diff=4):
		self.split = split
		super(HMDB51, self).__init__(train=train, transform=transform, mode=mode, num_frames=num_frames, num_test=num_test, source=source, diff=diff)
		self._set_info()

	def __len__(self):
		return self.len

	def _set_info(self):
		self.dict_label = {}
		if self.train:
			required_mark = 1
		else:
			required_mark = 2
		dict_names = pickle.load(open(os.path.join(self.path_data, 'dict_names_'+self.source+'_new.pkl'), 'rb'))
		self.info = []
		cur_label = 0
		for root, dirs, files in os.walk(self.path_infos):
			files.sort()
			for file in files:
				if file.endswith('split%i.txt' %self.split):					
					raw_info = np.genfromtxt(os.path.join(self.path_infos, file), dtype=None, comments=None)
					for i in range(raw_info.shape[0]):	
						path_name = (raw_info[i][0]).decode('UTF-8')
						item_name = dict_names[path_name]
						self.dict_label[item_name] = cur_label
						item_num = self.dict_num[item_name]
						mark = raw_info[i][1]
						if mark == required_mark and item_num > self.num_frames:
							self.info.append(item_name)
					cur_label += 1

		self.info = np.array(self.info)
		self.len = len(self.info)

class ACT(Base_OF):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/ACT/images'
	path_infos = '/net/hci-storage01/groupfolders/compvis/nsayed/data/ACT/labels/task1'
	memory_usage = 15000000
	info = None
	dict_label = None
	def __init__(self, split=1, train=True, transform=None, mode='def', num_frames=12, num_test=5, source='l', diff=4):
		super(ACT, self).__init__(train=train, transform=transform, mode=mode, num_frames=num_frames, num_test=num_test, source=source, diff=diff)
		self._set_info()

	def __len__(self):
		return self.len

	def _set_info(self):
		self.dict_label = {}
		if self.train:
			info_name = 'trainlist.txt'
		else:
			info_name = 'testlist.txt'

		dict_names = pickle.load(open(os.path.join(self.path_data, 'dict_names_'+self.source+'_new.pkl'), 'rb'))
		self.info = []
		raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None, comments=None)
		for i in range(raw_info.shape[0]):	
			path_name = (raw_info[i][0][:-1]).decode('UTF-8')
			label = raw_info[i][1] - 1
			item_name = dict_names[path_name + '.avi']
			self.dict_label[item_name] = label
			item_num = self.dict_num[item_name]
			if item_num > self.num_frames:
				self.info.append(item_name)

		self.info = np.array(self.info)
		self.len = len(self.info)

# class YTA(Base_OF):
# 	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/YTA/images'
# 	path_infos = '/net/hci-storage01/groupfolders/compvis/nsayed/data/YTA/labels/task1'
# 	memory_usage = 15000000
# 	info = None

# 	def __init__(self, train=True, transform=None, mode='def', num_frames=12, source='s', diff=4):
# 		super(ACT, self).__init__(train=train, transform=transform, mode=mode, num_frames=num_frames, source=source, diff=diff)
# 		self._set_info()

# 	def __len__(self):
# 		return self.len

# 	def _set_info(self):
# 		self.dict_label = {}
# 		if self.train:
# 			info_name = 'trainlist.txt'
# 		else:
# 			info_name = 'testlist.txt'

# 		dict_names = pickle.load(open(os.path.join(self.path_data, 'dict_names_'+self.source+'_new.pkl'), 'rb'))
# 		self.info = []
# 		raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None, comments=None)
# 		for i in range(raw_info.shape[0]):	
# 			path_name = (raw_info[i][0][:-1]).decode('UTF-8')
# 			label = raw_info[i][1] - 1
# 			item_name = dict_names[path_name + '.avi']
# 			self.dict_label[item_name] = label
# 			item_num = self.dict_num[item_name]
# 			if item_num > self.num_frames:
# 				self.info.append(item_name)

# 		self.info = np.array(self.info)
# 		self.len = len(self.info)

class Videos(data.Dataset):
	data_cache = None
	memory_usage = 30000000

	def __init__(self, train=True, transform=None, mode='def', num_frames=12, source='l', diff=4):
		if mode == 'ar':
			raise Exception('mode ar not implemented for Videos dataset')
		self.datasets = [UCF101(train=train, transform=transform, mode=mode, num_frames=num_frames, source=source, diff=diff),
										ACT(train=train, transform=transform, mode=mode, num_frames=num_frames, source=source, diff=diff),
										HMDB51(train=train, transform=transform, mode=mode, num_frames=num_frames, source=source, diff=diff)]
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



# elif self.mode == 'ar':
# 	for ind_frame in ind_frames:
# 		name_frame = item_name + '_' + str(ind_frame) + '.jpg'	
# 		buf = self._get_buf_image(name_frame)
# 		buffers_norms.append((buf, 1))

# elif self.mode == 'ar':
# 	paths_norms = [] 
# 	if self.train:
# 		ind_frames = np.array([random.randint(1, num_total+1)]).astype(np.int32)
# 	else:
# 		ind_frames = np.linspace(1, num_total, self.num_test).astype(np.int32)

# TODO currently implemented as buffer loader