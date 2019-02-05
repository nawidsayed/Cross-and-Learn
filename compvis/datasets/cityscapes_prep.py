from PIL import Image
import os
import os.path
import numpy as np

def mkdir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

path_original = "/net/hci-storage01/groupfolders/compvis/nsayed/data/cityscapes/original"
path_rescaled = "/net/hci-storage01/groupfolders/compvis/nsayed/data/cityscapes/rescaled"

splits = ['train', 'val', 'test']
infos = ['images_fine', 'labels_fine']

number_of_file = 0

size = (256,128)

for info in infos:
	for split in splits:
		path_o = os.path.join(path_original, info, split)
		path_r = os.path.join(path_rescaled, info, split)
		mkdir(path_r)

		for root, dirs, files in os.walk(path_o):
			for file in files:			
				if info == 'images_fine' or file.endswith('labelIds.png'):
					number_of_file += 1
					img = Image.open(os.path.join(root, file))
					print(number_of_file)
					if info == 'images_fine':
						img = img.resize(size, resample=Image.BILINEAR)
					else:
						img = img.resize(size, resample=Image.NEAREST)
					path_save = os.path.join(path_r, file)
					img.save(path_save)



