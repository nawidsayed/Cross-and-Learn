import torch.utils.data as data
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms
import torch

class Toy(data.Dataset):

	def __init__(self, from_A=None, train=True, length=1000, size=(64,64,3), mode='shapes', ratio=0.2):
		assert from_A is not None, 'specify the "from_A" keyword for the toy dataset'
		self.from_A = from_A
		self.train = train
		self.len = length
		self.size = size
		self.mode = mode
		if self.mode == 'colors':
			self.ratio = ratio
		self.size_big = tuple([int(1.5*size[0]), int(1.5*size[1]), size[2]])
		self.totensor = transforms.ToTensor()
		self.centercrop = transforms.CenterCrop(self.size[0])
		if from_A:
			self.random = np.random.RandomState()
		else:
			self.random = np.random.RandomState()
		if not self.train:
			self.len = 256 * 256

	def __len__(self):
		return self.len

	def __getitem__(self,index):
		if self.train:
			if self.mode == 'colors':
				return self._get_colors_image()
			if self.mode == 'shapes':
				return self._get_shapes_image()
		else:
			if self.mode == 'colors':
				return self._get_colors_image_test(index)

	def _get_colors_image(self):
		# GMM Power colors
		rgb = self._get_GMM_power_rgb()
		# Uniform colors:
		# rgb = self._get_uniform_rgb()
		image = np.zeros(self.size).astype(np.uint8)
		qw = int(self.size[0] / 4)
		qh = int(self.size[1] / 4)
		for chanel in range(3):
			# Color fills the entire image 
			image[:,:,chanel] = rgb[chanel]
			# Color fills only a box in the center
			# image[qw:3*qw,qh:3*qh,chanel] = rgb[chanel]
		image = Image.fromarray(image)
		filter = ImageFilter.GaussianBlur(radius=3)
		image = image.filter(filter)
		image = self.totensor(image)
		return image, -1

	def _get_colors_image_test(self, index):
		x1 = index % 256
		x2 = int(index / 256)

		vn = np.array([0,0,1]) 
		v1 = np.array([0,1,-1]) 
		v2 = np.array([1,0,-1]) 

		rgb = 255*vn + x1*v1 + x2*v2
		rgb = rgb.astype(np.uint8)

		image = np.zeros(self.size).astype(np.uint8)
		for chanel in range(3):
			image[:,:,chanel] = rgb[chanel]

		image = Image.fromarray(image)
		image = self.totensor(image)

		# Only accept images fulfilling x1 + x2 <= 255
		return image, torch.IntTensor([x1, x2])


	def _get_uniform_rgb(self):
		rgb = self.random.randint(0, 255, size=3)
		return rgb

	def _get_normal_rgb(self, mean, stdev):
		rgb = np.zeros(3)
		for color in range(3):
			rgb[color] = self.random.normal(loc=mean[color], scale=stdev[color])
		rgb = np.minimum(rgb, 255)
		rgb = np.maximum(rgb, 0)
		rgb = rgb.astype(np.uint8)
		return rgb

	def _get_GMM_rgb(self):
		mean1 = [200,100,0]
		mean2 = [0,200,100]
		stdev = [20,20,20]
		ratio = self.ratio
		r = self.random.rand()
		if r < ratio:
			if self.from_A:
				return self._get_normal_rgb(mean1, stdev)
			else:
				return self._get_normal_rgb(mean2, stdev)
		else:
			if not self.from_A:
				return self._get_normal_rgb(mean1, stdev)
			else:
				return self._get_normal_rgb(mean2, stdev)

	def _get_GMM_power_rgb(self):
		num = 8
		power = 2
		std = 5
		mean_dist = 32

		mean_v = np.array([[32,32,32],[32,32,224],[32,224,32],[32,224,224],
			[224,32,32],[224,32,224],[224,224,32],[224,224,224]])
		# mean_v = np.array([mean_dist,mean_dist,mean_dist])
		stdev = [std,std,std]

		num += 1
		phase_changes = np.arange(num) ** power / (num-1)**power
		r = self.random.rand()
		ind = 0
		while r >= phase_changes[ind]:
			ind += 1

		# mean = ind * mean_v
		ind -= 1
		mean = mean_v[ind]
		return self._get_normal_rgb(mean, stdev)

	def _get_shapes_image(self):
		angle = self.random.randint(0, 360)
		image = self.random.rand(*self.size_big)
		image = self._float_to_uint(image)
		# image = np.zeros(self.size).astype(np.uint8)
		image = Image.fromarray(image)
		if self.from_A:
			self._draw_pacman(image)
		else:
			self._draw_rect(image)
		image = image.rotate(angle)	
		image = self.centercrop(image)
		filter = ImageFilter.GaussianBlur(radius=3)
		image = image.filter(filter)
		image = self.totensor(image)
		image = self._normalize(image)
		return image, angle


	def _float_to_uint(self, image):
		mini = np.min(image)
		maxi = np.max(image)
		image = (image - mini) / (maxi-mini)
		image = np.uint8(image*255)
		return image

	def _draw_pacman(self, image):
		w, h = image.size
		x = int(w / 2)
		y = int(h / 2)
		r = int(w / 4)
		draw = ImageDraw.Draw(image)
		draw.pieslice((x-r, y-r, x+r, y+r), -60, 240, fill=(255,255,255,255))
	    
	def _draw_triangle(self, image):
		w, h = image.size
		x = int(w / 2)
		y = int(h / 2)
		r = int(w / 4)
		draw = ImageDraw.Draw(image)
		draw.pieslice((x-r, y-r, x+r, y+r), 60, 120, fill=(255,255,255,255))

	def _draw_rect(self, image):
		w, h = image.size
		x0 = 0
		x1 = w
		y0 = int(w / 1.7)
		y1 = int(w)
		draw = ImageDraw.Draw(image)
		draw.rectangle([x0,y0,x1,y1], fill=(255,255,255,255))

	def _normalize(self, image):
		mean = image.mean(1)
		mean = mean.mean(2)
		mean = torch.squeeze(mean)
		image_squared = image * image
		sdev = image_squared.mean(1)
		sdev = sdev.mean(2)
		sdev = torch.squeeze(sdev)
		for t, m, s in zip(image, mean, sdev):
			t.sub_(m).div_(s)					
		return image




