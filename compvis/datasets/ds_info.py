import os
import numpy as np
import _pickle as pickle

__all__ = ['UCF101_i', 'HMDB51_i', 'ACT_i', 'Leeds_i', 'OlympicSports_i']

# train = None returns combined train and test set
# source allows for different modes in dataset

class Base_Info(object):
	def __init__(self, train=True, source='l'):
		self.train = train
		self.source = source
		self.num_info = 0

	def __len__(self):
		raise NotImplementedError('Base_Info should implement __len__')

	def get_label(self, index):
		raise NotImplementedError('Base_Info should implement get_label')

	def get_data_cache(self):
		raise NotImplementedError('Base_Info should implement get_data_cache')

class Base_Info_Video(Base_Info):
	def get_rgb_path(self, index, ind_frame):
		raise NotImplementedError('Base_Info_Video should implement get_rgb_path')

	def get_of_path(self, index, ind_frame, direction):
		raise NotImplementedError('Base_Info_Video should implement get_of_path')

	def get_of_key(self, index, ind_frame, direction):
		raise NotImplementedError('Base_Info_Video should implement get_of_key')

	def get_norm(self, index, ind_frame):
		raise NotImplementedError('Base_Info_Video should implement get_norm')

	def get_num_rgb(self, index):
		raise NotImplementedError('Base_Info_Video should implement get_num_rgb')

	def get_mag(self, index):
		raise NotImplementedError('Base_Info_Video should implement get_mag')

	def get_label(self, index, ind_frame):
		raise NotImplementedError('Base_Info_Video should implement get_label using ind_frame')

class Base_Info_Image(Base_Info):
	def get_image_path(self, index):
		raise NotImplementedError('Base_Info_Image should implement get_image_path')

class UCF101_i(Base_Info_Video):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/UCF101/images'
	path_infos = '/export/home/nsayed/data/UCF101'
	def __init__(self, train=True, split=1, source='l', num_frames=12, min_msd=0):
		super(UCF101_i, self).__init__(train=train, source=source)
		self.split = split
		self.use_groups = 100
		if split < 0:
			self.split = 1
			self.use_groups = -split
		self.num_frames = num_frames
		self.min_msd = min_msd
		self.dict_num = pickle.load(open(os.path.join(self.path_data, 
			'dict_num_'+self.source+'.pkl'), 'rb'))
		self.dict_mag = pickle.load(open(os.path.join(self.path_data, 
			'dict_mag_'+ 'l' +'.pkl'), 'rb'))
		self.dict_norm = pickle.load(open(os.path.join(self.path_data, 
			'dict_norm_'+self.source+'.pkl'), 'rb'))
		dict_names = pickle.load(open(os.path.join(self.path_data,
			'dict_names_'+self.source+'_new.pkl'), 'rb'))
		self.dict_list_3rd_max_msd = pickle.load(open(os.path.join(self.path_data, 
			'dict_list_3rd_max_msd_'+ 'l' +'.pkl'), 'rb'))
		self.dict_label = {}
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
		self.list_items = []
		info_names = []
		if self.train or self.train is None:
			info_names.append('trainlist0%d.txt' %self.split)
		if not self.train or self.train is None:
			info_names.append('testlist0%d.txt' %self.split)
		for info_name in info_names:
			self._initialize(info_name, dict_names)
		self.len = len(self.list_items)

	def _initialize(self, info_name, dict_names):
		raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None)
		for i in range(raw_info.shape[0]):
			if info_name[:5] == 'train':
				name = (raw_info[i][0][:-4]).decode('UTF-8')
			else:
				name = (raw_info[i][:-4]).decode('UTF-8')
			drop_first = int((len(name) - 11) / 2) + 1
			name = name[drop_first:]
			item_name = dict_names[name + '.avi']
			item_num = self.dict_num[item_name]
			item_group = int(name[-6:-4])
			if item_num > self.num_frames:
				item_max_msd = np.max(self.dict_list_3rd_max_msd[item_name])
				if item_max_msd > self.min_msd and item_group < 8+self.use_groups:
					self.list_items.append(item_name)

	def __len__(self):
		return self.len

	def get_label(self, index, ind_frame=None):
		item_name = self.list_items[index]
		return int(self.dict_label[item_name])

	def get_data_cache(self):
		return pickle.load(open(os.path.join(self.path_data, 'dict_data_'+self.source+'.pkl'), 'rb'))

	def get_rgb_path(self, index, ind_frame):
		item_name = self.list_items[index]
		item_folder = os.path.join(self.path_data, item_name)
		name_frame = '%s_%d.jpg' %(item_name, ind_frame+1)
		return os.path.join(item_folder, name_frame)	

	def get_of_key(self, index, ind_frame, direction):
		item_name = self.list_items[index]
		return '%s_%s_%d.jpg' %(item_name, direction, ind_frame+1)

	def get_norm(self, index, ind_frame):
		item_name = self.list_items[index]
		name_norm = '%s_%d' %(item_name, ind_frame+1)
		return self.dict_norm[name_norm]

	def get_num_rgb(self, index):
		item_name = self.list_items[index]
		return self.dict_num[item_name]

	def get_mag(self, index):
		item_name = self.list_items[index]
		num = self.dict_num[item_name]
		mag = []
		for i in range(1, num):
			key = '%s_%d' %(item_name, i)
			mag.append(self.dict_mag[key])
		return mag

class HMDB51_i(Base_Info_Video):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/HMDB51/images'
	path_infos = '/net/hci-storage01/groupfolders/compvis/nsayed/data/HMDB51/testTrainMulti_7030_splits'
	def __init__(self, train=True, split=1, source='l', num_frames=12, min_msd=0):
		super(HMDB51_i, self).__init__(train=train, source=source)
		self.split = split
		self.num_frames = num_frames
		self.min_msd = min_msd
		self.dict_num = pickle.load(open(os.path.join(self.path_data, 
			'dict_num_'+self.source+'.pkl'), 'rb'))
		self.dict_mag = pickle.load(open(os.path.join(self.path_data, 
			'dict_mag_'+ 'l' +'.pkl'), 'rb'))
		self.dict_norm = pickle.load(open(os.path.join(self.path_data, 
			'dict_norm_'+self.source+'.pkl'), 'rb'))
		dict_names = pickle.load(open(os.path.join(self.path_data, 
			'dict_names_'+self.source+'_new.pkl'), 'rb'))
		self.dict_list_3rd_max_msd = pickle.load(open(os.path.join(self.path_data, 
			'dict_list_3rd_max_msd_'+ 'l' +'.pkl'), 'rb'))
		self.dict_label = {}
		required_marks = []
		if self.train or self.train is None:
			required_marks.append(1)
		if not self.train or self.train is None:
			required_marks.append(2)
		self.list_items = []

		for required_mark in required_marks:
			self._initialize(required_mark, dict_names)
		self.len = len(self.list_items)

	def _initialize(self, required_mark, dict_names):
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
							item_max_msd = np.max(self.dict_list_3rd_max_msd[item_name])
							if item_max_msd > self.min_msd:
								self.list_items.append(item_name)
					cur_label += 1

	def __len__(self):
		return self.len

	def get_label(self, index, ind_frame=None):
		item_name = self.list_items[index]
		return int(self.dict_label[item_name])

	def get_data_cache(self):
		return pickle.load(open(os.path.join(self.path_data, 'dict_data_'+self.source+'.pkl'), 'rb'))

	def get_rgb_path(self, index, ind_frame):
		item_name = self.list_items[index]
		item_folder = os.path.join(self.path_data, item_name)
		name_frame = '%s_%d.jpg' %(item_name, ind_frame+1)
		return os.path.join(item_folder, name_frame)	

	def get_of_key(self, index, ind_frame, direction):
		item_name = self.list_items[index]
		return '%s_%s_%d.jpg' %(item_name, direction, ind_frame+1)

	def get_norm(self, index, ind_frame):
		item_name = self.list_items[index]
		name_norm = '%s_%d' %(item_name, ind_frame+1)
		return self.dict_norm[name_norm]

	def get_num_rgb(self, index):
		item_name = self.list_items[index]
		return self.dict_num[item_name]

	def get_mag(self, index):
		item_name = self.list_items[index]
		num = self.dict_num[item_name]
		mag = []
		for i in range(1, num):
			key = '%s_%d' %(item_name, i)
			mag.append(self.dict_mag[key])
		return mag

class ACT_i(Base_Info_Video):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/ACT/images'
	path_infos = '/net/hci-storage01/groupfolders/compvis/nsayed/data/ACT/labels/task1'	
	def __init__(self, train=True, split=1, source='l', num_frames=12, min_msd=0):
		super(ACT_i, self).__init__(train=train, source=source)
		self.num_frames = num_frames
		self.min_msd = min_msd
		self.dict_num = pickle.load(open(os.path.join(self.path_data, 
			'dict_num_'+self.source+'.pkl'), 'rb'))
		self.dict_mag = pickle.load(open(os.path.join(self.path_data, 
			'dict_mag_'+ 'l' +'.pkl'), 'rb'))
		self.dict_norm = pickle.load(open(os.path.join(self.path_data, 
			'dict_norm_'+self.source+'.pkl'), 'rb'))
		dict_names = pickle.load(open(os.path.join(self.path_data, 
			'dict_names_'+self.source+'_new.pkl'), 'rb'))
		self.dict_list_3rd_max_msd = pickle.load(open(os.path.join(self.path_data, 
			'dict_list_3rd_max_msd_'+ 'l' +'.pkl'), 'rb'))
		info_names = []
		if self.train or self.train is None:
			info_names.append('trainlist.txt')
		if not self.train or self.train is None:
			info_names.append('testlist.txt')
		self.list_items = []
		self.dict_label = {}
		for info_name in info_names:
			self._initialize(info_name, dict_names)
		self.len = len(self.list_items)		

	def _initialize(self, info_name, dict_names):
		raw_info = np.genfromtxt(os.path.join(self.path_infos, info_name), dtype=None, comments=None)
		for i in range(raw_info.shape[0]):	
			path_name = (raw_info[i][0][:-1]).decode('UTF-8')
			label = raw_info[i][1] - 1
			item_name = dict_names[path_name + '.avi']
			self.dict_label[item_name] = label
			item_num = self.dict_num[item_name]
			if item_num > self.num_frames: 
				item_max_msd = np.max(self.dict_list_3rd_max_msd[item_name])
				if item_max_msd > self.min_msd:
					self.list_items.append(item_name)

	def __len__(self):
		return self.len

	def get_label(self, index, ind_frame=None):
		item_name = self.list_items[index]
		return int(self.dict_label[item_name])

	def get_data_cache(self):
		return pickle.load(open(os.path.join(self.path_data, 'dict_data_'+self.source+'.pkl'), 'rb'))

	def get_rgb_path(self, index, ind_frame):
		item_name = self.list_items[index]
		item_folder = os.path.join(self.path_data, item_name)
		name_frame = '%s_%d.jpg' %(item_name, ind_frame+1)
		return os.path.join(item_folder, name_frame)	

	def get_of_key(self, index, ind_frame, direction):
		item_name = self.list_items[index]
		return '%s_%s_%d.jpg' %(item_name, direction, ind_frame+1)

	def get_norm(self, index, ind_frame):
		item_name = self.list_items[index]
		name_norm = '%s_%d' %(item_name, ind_frame+1)
		return self.dict_norm[name_norm]

	def get_num_rgb(self, index):
		item_name = self.list_items[index]
		return self.dict_num[item_name]

	def get_mag(self, index):
		item_name = self.list_items[index]
		num = self.dict_num[item_name]
		mag = []
		for i in range(1, num):
			key = '%s_%d' %(item_name, i)
			mag.append(self.dict_mag[key])
		return mag

class OlympicSports_i(Base_Info_Video):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/OlympicSports/images'
	path_info = '/net/hci-storage02/groupfolders/compvis/nsayed/data/OlympicSports'
	def __init__(self, train=True, split=1, source='l', num_frames=12, min_msd=0):
		super(OlympicSports_i, self).__init__(train=train, source=source)
		self.num_frames = num_frames
		self.min_msd = min_msd
		self.dict_num = pickle.load(open(os.path.join(self.path_data, 
			'dict_num_'+self.source+'.pkl'), 'rb'))
		self.dict_mag = pickle.load(open(os.path.join(self.path_data, 
			'dict_mag_'+ 'l' +'.pkl'), 'rb'))
		self.dict_norm = pickle.load(open(os.path.join(self.path_data, 
			'dict_norm_'+self.source+'.pkl'), 'rb'))
		dict_names = pickle.load(open(os.path.join(self.path_data, 
			'dict_names_'+self.source+'.pkl'), 'rb'))
		self.dict_list_3rd_max_msd = pickle.load(open(os.path.join(self.path_data, 
			'dict_list_3rd_max_msd_'+ 'l' +'.pkl'), 'rb'))
		self.dict_label = {}
		required_marks = []
		if self.train or self.train is None:
			required_marks.append('train')
		if not self.train or self.train is None:
			required_marks.append('test')
		self.list_items = []
		for required_mark in required_marks:
			self._initialize(required_mark, dict_names)
		self.len = len(self.list_items)

	def _initialize(self, required_mark, dict_names):
		cur_label = 0
		path_mark = os.path.join(self.path_info, required_mark)
		for root, dirs, files in os.walk(path_mark):
			files.sort()
			for file in files:
				if file.endswith('.txt'):					
					raw_info = np.genfromtxt(os.path.join(path_mark, file), dtype=None, comments=None)
					path_name_base = os.path.join(self.path_info, 'clips', file[:-4])
					for i in range(raw_info.shape[0]):	
						path_name = (raw_info[i]).decode('UTF-8')
						path_name = os.path.join(path_name_base, path_name)
						if path_name in dict_names:
							item_name = dict_names[path_name]
							self.dict_label[item_name] = cur_label
							item_num = self.dict_num[item_name]
							if item_num > self.num_frames:
								item_max_msd = np.max(self.dict_list_3rd_max_msd[item_name])
								if item_max_msd > self.min_msd:
									self.list_items.append(item_name)
					cur_label += 1

	def __len__(self):
		return self.len

	def get_label(self, index, ind_frame=None):
		item_name = self.list_items[index]
		return int(self.dict_label[item_name])

	def get_data_cache(self):
		return pickle.load(open(os.path.join(self.path_data, 'dict_data_'+self.source+'.pkl'), 'rb'))

	def get_rgb_path(self, index, ind_frame):
		item_name = self.list_items[index]
		item_folder = os.path.join(self.path_data, item_name)
		name_frame = '%s_%d.jpg' %(item_name, ind_frame+1)
		return os.path.join(item_folder, name_frame)	

	def get_of_path(self, index, ind_frame, direction):
		item_name = self.list_items[index]
		item_folder = os.path.join(self.path_data, item_name + '_flow')
		name_frame = '%s_%s_%d.png' %(item_name, direction, ind_frame+1)
		return os.path.join(item_folder, name_frame)	

	def get_of_key(self, index, ind_frame, direction):
		item_name = self.list_items[index]
		return '%s_%s_%d.jpg' %(item_name, direction, ind_frame+1)

	def get_norm(self, index, ind_frame):
		item_name = self.list_items[index]
		name_norm = '%s_%d' %(item_name, ind_frame+1)
		return self.dict_norm[name_norm]

	def get_num_rgb(self, index):
		item_name = self.list_items[index]
		return self.dict_num[item_name]

	def get_mag(self, index):
		item_name = self.list_items[index]
		num = self.dict_num[item_name]
		mag = []
		for i in range(1, num):
			key = '%s_%d' %(item_name, i)
			mag.append(self.dict_mag[key])
		return mag

class Leeds_i(Base_Info_Image):
	path_data = '/net/hci-storage02/groupfolders/compvis/nsayed/data/LEEDS/lsp_dataset_original'
	def __init__(self, train=True, source='l'):
		super(Leeds_i, self).__init__(train=train, source=source)
		if source == 'l':
			self.path_images = os.path.join(self.path_data, 'images')
		else:
			self.path_images = os.path.join(self.path_data, 'crops_227x227')
		self.list_items = []
		if self.train or self.train is None:
			self.list_items += list(np.arange(1, 3939))
		if not self.train or self.train is None:
			self.list_items += list(np.arange(3939, 5939))
		self.len = len(self.list_items)		

	def __len__(self):
		return self.len

	def get_label(self, index):
		return -1

	def get_data_cache(self):
		return None

	def get_image_path(self, index):
		item = self.list_items[index]
		name_image = 'im%05d.png' %item
		return os.path.join(self.path_images, name_image)

if __name__ == '__main__':
	import _pickle as pickle
	# labels = pickle.load(open('/export/home/nsayed/results/of_2/fm_l_caffe_hm2/cluster_norm_euclidean/labels_-1.pkl', 'rb'))
	dset = UCF101_i(split=1)
	print(len(dset))

	# dset.set_labels(labels[0], -30)
	# print(dset.get_label(10, 60))
	# for i in range(96):
	# 	index = i*100
	# 	item_name = dset.list_items[index]
	# 	labs = dset.dict_label[item_name]
	# 	num = dset.get_num_rgb(index)
	# 	print(labs, num)
	# num_test = -20
	# num_rgb = 300
	# num_frames = 10
	# ind_frame = 232
	# get_label_index(num_test, num_rgb, num_frames, ind_frame)


	# import ipdb; ipdb.set_trace()