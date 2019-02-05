import numpy as np
import os
import _pickle as pickle
import torch	

def mkdir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)
		return True
	return False

def pickle_load(path):
	return pickle.load(open(path, 'rb'))

def pickle_dump(data, path):
	return pickle.dump(data, open(path, 'wb'))

def print_iterable(iterable, delimiter=', ', max_digits=None, print_keys=True):
	string = ''
	if isinstance(iterable, dict):
		iterable = iterable.items()
	for key, val in iterable:
		if print_keys:
			string += (str(key) + ': ')
		if max_digits is not None and isinstance(val, float):
			val = '%.*f' %(max_digits, val)
		string += (str(val) + delimiter) 
	return string[:-len(delimiter)]

def adjust_learning_rate(optimizer, learning_rate):
	if learning_rate < 0.000001:
		learning_rate = 0.000001
	for param_group in optimizer.param_groups:
		param_group['lr'] = learning_rate

# Used for saving filters
def normalize(arr):
    mini = np.min(arr)
    arr -= mini
    maxi = np.max(arr)
    arr /= maxi
    return arr, mini, maxi

def cos_sim(f_1, f_2):
	len_1 = torch.sqrt(torch.sum(f_1 ** 2, dim=1))
	len_2 = torch.sqrt(torch.sum(f_2 ** 2, dim=1))
	return torch.sum(f_1 * f_2, dim=1) / (len_1 * len_2 + 0.00001)

def euc_sim(f_1, f_2):
	sim = cos_sim(f_1, f_2)
	if not isinstance(sim, float):
		return 1 - torch.sqrt(2-2*sim)
	else:
		return 1 - np.sqrt(2-2*sim)

def lin_sim(f_1, f_2):
	sim = cos_sim(f_1, f_2)
	if not isinstance(sim, float):
		return 1 - torch.acos(sim) / np.pi
	else:
		return 1 - np.arccos(sim) / np.pi

def euclidean_sim(f_1, f_2):
	dist = torch.sum(torch.sqrt((f_1-f_2) ** 2), dim=1)
	return -dist

def get_color_img(img):
	if isinstance(img, np.ndarray):
		pass
	elif isinstance(img, (torch.Tensor, torch.cuda.FloatTensor)):
		img = img.cpu().numpy()
	return (img * 255).astype(np.uint8) 

def get_color_flow(flow_in, chanel_first=False):
	if chanel_first == True:
		flow_in = flow_in.transpose(1,2,0)
	vs = np.array([[[1,1,1]]])
	v1 = np.array([[[1,0,-1]]])/np.sqrt(2)
	v2 = np.array([[[1,-2,1]]])/np.sqrt(6)
	flow_norm = np.sqrt(np.max(flow_in[...,0]**2 + flow_in[...,1]**2)) + 0.000001
	flow = flow_in / flow_norm
	x = flow[:,:,0]
	y = flow[:,:,1]
	darkness = np.sqrt(x**2 + y**2)+ 0.000001
	x /= darkness
	y /= darkness
	x = np.expand_dims(x, axis=2)
	y = np.expand_dims(y, axis=2)
	color = 0.5*(vs + v1*x + v2*y)
	darkness = np.expand_dims(darkness, axis=2)
	res = 255*(vs - darkness*color)
	flow_color = res.astype(np.uint8)
	if chanel_first == True:
		flow_in = flow_in.transpose(2,0,1)
		flow = flow.transpose(2,0,1)
		flow_color = flow_color.transpose(2,0,1)
	return flow_color

# load sd currently omits additional keyss
def load_sd(net, new_sd):
	sd = net.state_dict()
	for key in sd:
		if key in new_sd:
			sd[key] = new_sd[key]
		else:
			print('missing: ', key)
	net.load_state_dict(sd)

def fix_state_dict(path, replace_dict=None, remove_list=None):
	sd = torch.load(path)
	if replace_dict is not None:
		sd_new = {}
		for key, val in sd.items():
			key_new = key
			for word, word_rep in replace_dict.items():
				l = len(word)
				ind = key.find(word)
				if ind != -1:
					key_new = key[:ind] + word_rep + key[ind+l:]
					print('replace: ', key, key_new)
					break					
			sd_new[key_new] = val
		sd = sd_new
	if remove_list is not None:
		sd_new = {}
		for key, val in sd.items():
			found = False
			for word in remove_list:
				ind = key.find(word)
				if ind != -1:
					print('remove: ', key)
					found = True
					break
			if not found:		 
				sd_new[key] = val
		sd = sd_new
	torch.save(sd, path)	

# Write functions here for debugging:
def count_zeros(t):
	print(t.nonzero().size()[0], t.size()[0]*t.size(1))

# TODO finish or remove
# def set_locals(self):
# 	dict_info = locals()
# 	del self.dict_info['self']
# 	if self.dict_info['mode'] != 'fm_fc':
# 		del self.dict_info['output_dim']
# 	for key, val in self.dict_info.items():
# 		setattr(self, key, val)
# 	del self.dict_info['save']