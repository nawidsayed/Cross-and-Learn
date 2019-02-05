from __future__ import print_function
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import operator
import _pickle as pickle
from time import time

class Bayer(data.Dataset):
	path_original = "/net/hci-storage01/groupfolders/compvis/bbrattol/database_restricted/Bayer_HD/"
	path_rescaled = "/net/hci-storage01/groupfolders/compvis/nsayed/data/Bayer/"
	image_folder_original = path_original + 'Weed images/'
	mask_folder_original = path_original + 'Binary images/'
	image_folder_rescaled = path_rescaled + 'images/'
	mask_folder_rescaled = path_rescaled + 'masks/'
	info_folder = '/export/home/nsayed/data/Bayer/'
	train_file = info_folder+'trainset.csv'
	test_file = info_folder+'testset.csv'
	statistics_file = info_folder+'normBayer.pkl'

	def __init__(self, train=True, transform=None):
		assert transform is not None, 'Initialize Bayer dataset with a transform'
		self.train = train
		self.transform = transform
		self.len = None
		self.label_nums = None
		self.info = None
		self.label_info = None
		self.num_classes = None

		if self.train:
			info_file = Bayer.train_file
		else:
			info_file = Bayer.test_file
			
		raw_info = np.genfromtxt(info_file, skip_header=1, dtype=None, delimiter=';',usecols=(0,11))

		self.info = []
		num_images = 0
		for i in range(raw_info.shape[0]):
			file_num = (raw_info[i,1][:-4]).decode('UTF-8')
			img_path = file_num + '.jpg'
			tif_path = file_num + '.tif'
			if os.path.exists(Bayer.image_folder_original + img_path) and os.path.exists(Bayer.mask_folder_original + tif_path):
				eppo = (raw_info[i,0]).decode('UTF-8')
				self.info.append([file_num,eppo])
				num_images += 1

		self.len = num_images
		self.info = np.array(self.info)		
		self.label_nums = self._get_label_nums(self.info)

		label_info = sorted(self.label_nums.keys(), key=operator.itemgetter(0))

		label_counter = 0
		self.label_info = dict()
		for eppo in label_info:
			self.label_info[eppo] = label_counter
			label_counter += 1

		self.num_classes = label_counter

		print('Initialized Dataset, found %i images and %i classes' %(self.len, self.num_classes))


		'''
		info = []

		for file_num, eppo in self.info:
			if eppo in label_nums:
				info.append([file_num, eppo])

		info=np.array(info)

		
		
		random=np.random
		random.seed(42)
		random.shuffle(info)
		if self.train:
			self.info = info[:self.len]
		else:
			self.info = info[-self.len:]
		self.label_nums = self._get_label_nums(self.info)

		label_info=[]
		for eppo in label_nums:
			label_info.append(eppo)

		label_info.sort()
		label_counter = 0
		self.label_info = dict()
		for eppo in label_info:
			self.label_info[eppo] = label_counter
			label_counter += 1
		'''
		


	def _get_label_nums(self, info):
		label_nums = dict()
		for eppo in info[:,1]:
			if eppo in label_nums:
				label_nums[eppo] += 1
			else:
				label_nums[eppo] = 1
		return label_nums

	def get_label_vector(self):
		gt = np.zeros(self.len).astype(int)
		for i in range(self.len):
			eppo = self.info[i,1]
			label  = self.label_info[eppo]
			gt[i] = label
		return gt

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		# Keep in mind that the .png masks have 3 channels while the .tif masks only have 1 channel
		file_num = self.info[index,0]
		eppo = self.info[index,1]
		# In case of unlabeled data
		if eppo == '-1':
			label = -1
		else:
			label = self.label_info[eppo]
		img_path_rescaled = Bayer.image_folder_rescaled + file_num + '.png'
		mask_path_rescaled = Bayer.mask_folder_rescaled + file_num + '.png'
	
		try:
			img = Image.open(img_path_rescaled)
			mask = Image.open(mask_path_rescaled)	
		except FileNotFoundError:
			img_path_original = Bayer.image_folder_original + file_num + '.jpg'
			mask_path_original = Bayer.mask_folder_original + file_num + '.tif'
			img = Image.open(img_path_original)
			mask = Image.open(mask_path_original)
			pre_transforms = self.transform.transforms[:2]
			for trans in pre_transforms:
				img, mask = trans(img, mask)
			img.save(img_path_rescaled)
			mask.save(mask_path_rescaled)
		img, mask = self.transform(img, mask, drop_trans=2)
		return img, label 


	@classmethod
	def get_statistics(self, C=3, MAX=1000, size=(300,300)):
		# Loads the statistics of the Bayer dataset into a static member of the class
		# The statistics are calculated for images normalized to [0,1] 

		# C: number of channels 
		# MAX: number of images taken into account
		# size: rescaled size of the images

		picklefile = Bayer.statistics_file

		if os.path.exists(picklefile):
			with open(picklefile, 'rb') as f:
				stats = pickle.load(f) 
			return stats


		# If the statistics do not exist, create them
		else:
			counter = 1

			# array which stores the mean and stddev of the channels of all images up to that point
			stats = np.zeros((C,2))

			for filename in os.listdir(Bayer.image_folder_original):
				if filename.endswith(".jpg"): 
					img_path = os.path.join(Bayer.image_folder_original, filename)
					img = Image.open(img_path)
					img.thumbnail(size, resample=Image.BILINEAR)
					img = np.array(img).astype(float)

					# img_stat contains same properties as stats but imagewise
					img_stat = np.zeros((C,2))
					for i in range(C):
						img_stat[i,0] = np.mean(img[:,:,i])

					for i in range(C):
						img_stat[i,1] = np.mean(img[:,:,i]**2)
						img_stat[i,1] = np.sqrt(img_stat[i,1]-img_stat[i,0]**2)

					stats += img_stat


					counter +=1
					if counter <= MAX:
						continue
					else: 
						break
				else:
					continue
			    	

			stats /= MAX
			stats /= 255

			with open(picklefile, 'wb') as f:
				pickle.dump(stats, f, -1)
			return stats

'''

with open('norm.pkl', 'rb') as f:
	stats = pickle.load(f)

stats = stats/255

print(stats)

import transforms

transform = transforms.Compose([transforms.Scale(330), 
	transforms.CenterCrop(330), transforms.RandomCrop(299),
	transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
	transforms.Normalize(stats[:,0], stats[:,1]), transforms.ApplyMask()])

dataset = Bayer(train=False, transform=transform)

tpl = transforms.ToPILImage()

dataloader=torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4)

for img, mask in dataloader:
	print(torch.min(img))
	print(torch.max(img))
	print(img)
	img, mask = tpl(img[0], mask[0]) 
	img.save('image2.png')
	mask.save('mask2.png')
	break


'''



