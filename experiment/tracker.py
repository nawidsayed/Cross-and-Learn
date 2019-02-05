import numpy as np

import torch
import torch.nn as nn
from time import time

class Tracker(object):
	def update(self, output, labels):
		raise NotImplementedError('update() in Tracker not implemented')

	def result(self):
		raise NotImplementedError('result() in Tracker not implemented')

class Tracker_classification(Tracker):
	softmax = nn.Softmax(dim=1)
	def __init__(self, mode='single_frame', only_loss=False, with_nonzeros=False):
		self.tir = False
		self.loss = 0
		self.acc = 0
		self.err = 0
		self.counter = 0
		self.mode = mode
		self.only_loss = only_loss
		self.with_nonzeros = with_nonzeros
		self.norm = 0

	def track_individual_results(self):
		self.tir = True
		self.list_ir = []
		self.list_pred = []

	def update(self, output, labels, loss=None):
		if self.with_nonzeros:
			norms = output[-1]
			output = output[:-1]
			sum_norm = 0
			for norm in norms:
				sum_norm += norm
			sum_norm = float(sum_norm)
			self.norm += sum_norm
		if isinstance(output, (list, tuple)):
			output = torch.cat(output, dim=0)
			labels = torch.cat(labels, dim=0)
		labels = labels.data.cpu()
		output = self.softmax(output).data.cpu()
		if not self.mode == 'single_frame':
			output = torch.mean(output, dim=0, keepdim=True)		
			labels = labels[:1]	
		length = labels.size()[0]
		self.counter += length
		if not self.only_loss:
			pred, predind = torch.max(output, 1)
			diff = predind - labels
			for i in range(length):
				self.acc += output[i][labels[i]]
				if diff[i] != 0:
					self.err += 1	
				if self.tir:
					if diff[i] == 0:
						self.list_ir.append(1)
					else:
						self.list_ir.append(0)
					self.list_pred.append(predind[i])
		if loss is not None:
			loss = float(loss.data.cpu()) * length
			self.loss += loss

	def result(self):
		acc = self.acc / self.counter
		err = self.err / self.counter
		loss = self.loss / self.counter
		norm = self.norm / self.counter
		self.loss = 0
		self.acc = 0
		self.err = 0
		self.norm = 0
		self.counter = 0
		if self.with_nonzeros:
			return [('loss', loss), ('norm', norm), ('acc', acc), ('err', err)]
		return [('loss', loss), ('acc', acc), ('err', err)]

	def list_individual_results(self):
		list_ir = self.list_ir
		self.list_ir = []
		list_pred = self.list_pred
		self.list_pred = []
		return list_ir, list_pred


class Tracker_similarity(Tracker):
	def __init__(self, names=['mst', 'msf']):
		self.names = names
		self.size = len(self.names)
		self.similarities = np.zeros(self.size)
		self.loss = 0
		self.counter = 0
		self.norm = 0

	def update(self, output, labels, loss=None):
		norms = output[-1]
		output = output[:-1]
		sum_norm = 0
		for norm in norms:
		# Uncomment for lengths
		# 	sum_norm += norm.data.sum()
		# Uncomment for nonzeros
			sum_norm += norm
		sum_norm = float(sum_norm / len(norms))
		self.norm += sum_norm
		for i in range(self.size):
			if len(output) != 2:
				similarity = (output[2*i] + output[2*i+1]) / 2
			else:
				similarity = output[i]
			if i == 0:
				length = similarity.size()[0]
				self.counter += length
			self.similarities[i] += torch.sum(similarity, dim=0).data.cpu().numpy()
		if loss is not None:
			loss = float(loss.data.cpu()) * length
			self.loss += loss

	def result(self):
		similarities = self.similarities / self.counter
		loss = self.loss / self.counter
		norm = self.norm / self.counter
		self.similarities = np.zeros(self.size)
		self.loss = 0
		self.norm = 0
		self.counter = 0
		result = [('loss', loss), ('norm', norm)]
		for i in range(self.size):
			result.append((self.names[i], similarities[i]))
		return result

class Tracker_similarity_rec(Tracker):
	def __init__(self, names=['mst', 'msf'], names_rec=['rgb', 'of']):
		self.names = names
		self.names_rec = names_rec
		self.size = len(self.names)
		self.size_rec = len(self.names_rec)
		self.similarities = np.zeros(self.size)
		self.reconstructions = np.zeros(self.size)
		self.loss = 0
		self.counter = 0

	def update(self, output, labels, loss=None):
		output, losses_rec = output 
		for i in range(self.size):
			similarity = output[i]
			if i == 0:
				length = similarity.size()[0]
				self.counter += lengthq
			self.similarities[i] += torch.sum(similarity, dim=0).data.cpu().numpy()
		for i in range(self.size_rec):
			loss = losses_rec[i]
			self.reconstructions[i] += torch.sum(loss*length, dim=0).data.cpu().numpy()
		if loss is not None:
			loss = float(loss.data.cpu()) * length
			self.loss += loss

	def result(self):
		similarities = self.similarities / self.counter
		reconstructions = self.reconstructions / self.counter
		loss = self.loss / self.counter
		self.similarities = np.zeros(self.size)
		self.reconstructions = np.zeros(self.size)
		self.loss = 0
		self.counter = 0
		result = [('loss', loss)]
		for i in range(self.size):
			result.append((self.names[i], similarities[i]))
		for i in range(self.size_rec):
			result.append((self.names_rec[i], reconstructions[i]))
		return result

if __name__ == '__main__':
	from torch.autograd import Variable
	t0 = time()
	labels = torch.LongTensor(128).random_(0,101).cuda()
	output = torch.Tensor(128,101).normal_().cuda()
	t1 = time()
	labels = labels.cpu()
	output = output.cpu()
	t2 = time()
	labels = Variable(labels)
	output = Variable(output)
	print(t1-t0, t2-t1)
	tracker = Tracker_classification()
	tracker.update(output, labels)
	
