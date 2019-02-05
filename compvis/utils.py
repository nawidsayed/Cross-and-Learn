import os
import _pickle as pickle

def mkdir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def pickle_load(file):
	with open(file, 'rb') as f:
		content = pickle.load(f) 
	return content

def pickle_save(content, file):
	with open(file, 'wb') as f:
		pickle.dump(content, f, -1)

