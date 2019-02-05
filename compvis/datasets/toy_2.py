import torch.utils.data as data
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms
import torch

class Toy_2(data.Dataset):

	def __init__(self, length=1000, size=(7,7,1)):
		self.len = length
		self.size = size
		self.random = np.random.RandomState()
		self.totensor = transforms.ToTensor()
		self.half = int(size[0] / 2)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		image = np.zeros(self.size)
		# image = self.random.rand(*self.size)
		# image = np.uint8(image*255)
		counter = 0

		ind = self.random.randint(1, self.size[1]-1)
		image[:,ind,:] = 255

		ind = self.random.randint(1, self.size[0]-1)
		image[ind,:,:] = 255

		if ind == self.half:
			label = 1
		else:
			label = 0

		image = self.totensor(image)
		return image, label


