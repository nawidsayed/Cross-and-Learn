from __future__ import print_function
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import operator
import _pickle as pickle
from time import time

# For now, we dont store the rescaled images

class Pix2Pix(data.Dataset):
	path_original = '/export/home/nsayed/data/pix2pix/'

	def __init__(self, item=None, from_A=None, train=True, transform=None):
		assert item is not None, 'Specify the "item" keyword for the Pix2Pix dataset'
		assert from_A is not None, 'Specify the "from_A" keyword for the Pix2Pix dataset'
		self.item = item
		self.from_A = from_A
		self.train = train
		self.transform = transform
		self.path_original = os.path.join(Pix2Pix.path_original, self.item)
		if self.train:
			self.path_original = os.path.join(self.path_original, 'train')
		else:
			# TODO implement test or val case
			raise NotImplementedError

		self.info = []
		num_images = 0

		for file in os.listdir(self.path_original):
			self.info.append(file)
			num_images += 1

		self.len = num_images
		self.info = np.array(self.info)	

		print('Initialized Dataset, found %i images for item %s' %(self.len, self.item))

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		file = self.info[index]
		image_path = os.path.join(self.path_original, file)
		image = Image.open(image_path)
		w, h = image.size
		wh = int(w/2)
		if self.from_A:
			image = image.crop((0,0,wh,h))
		else:
			image = image.crop((wh,0,w,h))
		if self.transform is not None:
			image = self.transform(image)		

		return image

	# def __getitem__(self, index):
	# 	file = self.info[index]
	# 	image_path = os.path.join(self.path_original, file)
	# 	if self.from_A:
	# 		domain = 'A'
	# 	else:
	# 		domain = 'B'

	# 	image = read_image(image_path, domain=domain)
	# 	image = torch.Tensor(image)

	# 	return image


	def read_image(filename, domain=None, image_size=64):

		image = []
		fn =filename
		image = cv2.imread(fn)

		if domain == 'A':
			kernel = np.ones((3,3), np.uint8)
			image = image[:, :256, :]
			image = 255. - image
			image = cv2.dilate( image, kernel, iterations=1 )
			image = 255. - image
		elif domain == 'B':
			image = image[:, 256:, :]

		image = cv2.resize(image, (image_size,image_size))
		image = image.astype(np.float32) / 255.
		image = image.transpose(2,0,1)

		return image


	@classmethod
	def get_statistics(cls, item=None, from_A=None):
		assert item is not None, 'Specify the "item" keyword for the Pix2Pix dataset'
		assert from_A is not None, 'Specify the "from_A" keyword for the Pix2Pix get_statistics'
		# Loads the statistics of the Pix2Pix dataset into a static member of the class
		# The statistics are calculated for images normalized to [0,1] 

		# C: number of channels 
		# MAX: Maximum number of images taken into account
		C = 3
		MAX = 1000

		info_folder = os.path.join(cls.path_original, item)
		if from_A:
			statistics_file = os.path.join(info_folder, 'stats_A.pkl')
		else:
			statistics_file = os.path.join(info_folder, 'stats_B.pkl')

		if os.path.exists(statistics_file):
			with open(statistics_file, 'rb') as f:
				stats = pickle.load(f) 
			return stats


		# If the statistics do not exist, create them
		else:
			dataset = Pix2Pix(from_A=from_A, item=item)
			num_images = np.minimum(MAX, len(dataset))
			stats = np.zeros((C,2))
			for i in range(num_images):
				image = dataset[i]
				image = np.array(image).astype(float)
				# img_stat contains same properties as stats but imagewise
				img_stat = np.zeros((C,2))
				for i in range(C):
					img_stat[i,0] = np.mean(image[:,:,i])

				for i in range(C):
					img_stat[i,1] = np.mean(image[:,:,i]**2)
					img_stat[i,1] = np.sqrt(img_stat[i,1]-img_stat[i,0]**2)

				stats += img_stat

			stats /= num_images
			stats /= 255

			with open(statistics_file, 'wb') as f:
				pickle.dump(stats, f, -1)
			return stats
			    


