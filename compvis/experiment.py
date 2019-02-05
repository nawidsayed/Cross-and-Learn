'''
class experiment(object):
	def __init__(self):
		print('Experiment declared')

	def ini_traintrans(self, augment_trans=None, default_trans=None):
		self.augment_trans = augment_trans
		self.default_trans = default_trans

	def ini_traindata(self, dataset, **kwargs):















import numpy as np
import compvis.transforms as transforms
from compvis.datasets import Bayer
import _pickle as pickle
from PIL import Image
import torch.utils.data as data
import torch
from torchvision.models import *

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from time import time

#TODO Think about a learnable representation of colors instead of the default normalization
#TODO Dont forget the idea behind validation and test data. Tune on val, eval
#TODO Think about per leaf classification 

def adjust_lr(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr




def training(learning_rate=0.0005, batch_size=20, epochs=40, augmentation=True, mask=True, pretrained=True, adj_lr=True, save=True):
	stats = Bayer.get_statistics()


	# Specifies the number of images considered for estimating trianloss
	N_loss_img = 1000

	# Initializing trainloader

	if augmentation:
		list_of_transformations = [transforms.Scale(330), 
			transforms.CenterCrop(330), transforms.RandomCrop(299), transforms.RandomRotate(),
			transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
			transforms.Normalize(stats[:,0], stats[:,1])]
	else:
		list_of_transformations = [transforms.Scale(330), 
			transforms.CenterCrop(330), transforms.CenterCrop(299),
			transforms.ToTensor(), 
			transforms.Normalize(stats[:,0], stats[:,1])]

	if mask:
		list_of_transformations.append(transforms.ApplyMask())

	transform = transforms.Compose(list_of_transformations)

	trainset = Bayer(transform=transform)

	num_classes = trainset.num_classes

	trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True,num_workers=20, drop_last=False)
  
	# Initializing testloader
	transform_test = transforms.Compose([transforms.Scale(330), 
		transforms.CenterCrop(330), transforms.ToTensor(), 
		transforms.Normalize(stats[:,0], stats[:,1]), transforms.ApplyMask()])

	testset = Bayer(transform=transform_test, train=False)

	testloader = data.DataLoader(testset, batch_size=batch_size, num_workers=20, drop_last=False)

	hist_testpred = np.zeros((epochs, len(testset))).astype(int)

	# Initializing NN
	net = inception_v3(pretrained=pretrained, num_classes=num_classes).cuda()


	criterion = nn.CrossEntropyLoss()
	softmax = nn.Softmax()

	lr = learning_rate

	optimizer = optim.Adam(net.parameters(), lr=lr)

	print('Training started with learning rate: %f, batch_size: %i, epochs: %i, augmentation: %i, mask: %i, pretrained: %i, adjustable lr: %i' %(learning_rate,batch_size,epochs,augmentation,mask,pretrained, adj_lr))

	t_train = 0

	counter = 0

	hist_loss = []
	hist_acc = [0.]
	hist_err = [1.]
	loss_train = 0
	for epoch in range(epochs):
		net.train(mode = True)
		t0_train = time()
		for inputs, labels in trainloader:
			### Pytorch is not utilizing multiple gpus properly for batchsplitting, a
			### single GPU is quicker in this setup right now. My thoughts: Gradient calculation requires
			### memory transfer inbetween the gpus, which might be expensive
			inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

			optimizer.zero_grad()		

			outputs = net(inputs)


			if counter % 10000 == 0:
				pred, predind = torch.max(outputs[0], 1)
				pred_aux, predind_aux = torch.max(outputs[1], 1)
				print(counter)
				print(predind-labels)
				#print('aux')
				#print(predind_aux-labels)


			losses = []
			for output in outputs:
				losses.append(criterion(output, labels)) 
			loss = sum(losses)
			loss.backward()
			optimizer.step()

			loss_train += loss.data[0]

			counter += batch_size
			
			if counter%N_loss_img == 0:
				loss_train = loss_train * batch_size / N_loss_img
				print(loss_train)
				hist_loss.append(loss_train)
				loss_train = 0
			#TODO remove
			#if counter > 200:
			#	break
		t1_train = time()

		t_train = t_train + t1_train - t0_train

		net.train(mode=False)

		counter_test = 0
		acc = 0
		err = 0
		for inputs, labels in testloader:		
			inputs = Variable(inputs).cuda()
			outputs = net(inputs)
			outputs = softmax(outputs).data

			length = labels.size()[0]

			#print(length)

			counter_test += length

			#print(labels)
			#print(outputs)
				
			pred, predind = torch.max(outputs, 1)
			predind = predind.cpu()

			predind_np = predind.numpy()

			lower_ind = counter_test - length
			upper_ind = counter_test

			hist_testpred[epoch, lower_ind:upper_ind] = predind_np.flatten()

			diff = predind - labels
			#print(diff)
			for i in range(length):
				acc += outputs[i][labels[i]]
				#print(diff[i].numpy())
				if diff[i].numpy() != 0:
					err += 1

			#if counter_test%200 == 0:
			#	break

		acc /= counter_test
		err /= counter_test		
		hist_acc.append(acc)
		hist_err.append(err)
		print('accuracy after %i epochs: %f' %(epoch+1, acc))
		print('error after %i epochs: %f' %(epoch+1, err))

		err1 = 1-hist_acc[-2]
		err2 = 1-hist_acc[-1]
		if adj_lr:
			if err1 * 0.97 < err2:
				lr *= 0.8
				print('adjusting learning rate to %f' %lr)
				adjust_lr(optimizer, lr)



	print('train time %i' %t_train)



	if save:
		name = '%f_%i_%i_%i_%i_%i' %(learning_rate,batch_size,epochs,augmentation,mask,pretrained)
		if not adj_lr:
			name = name + 'not_adj'
		train_file = name + '_train.pkl'
		param_file = name + '_param.pth'
		testpred_file = name + '_testpred.pkl'

		with open(train_file, 'wb') as f:
			pickle.dump(hist_loss, f, -1)
			pickle.dump(hist_acc, f, -1)
			pickle.dump(hist_err, f, -1)

		with open(testpred_file, 'wb') as f:
			pickle.dump(hist_testpred, f, -1)

		state = net.state_dict()
		torch.save(state, param_file)


for learning_rate in [0.0003]:
	training(learning_rate=learning_rate, batch_size=40, epochs=0, save=False)
	#training(learning_rate=learning_rate, batch_size=40, epochs=60, augmentation=False)
	pass

for learning_rate in [0.0003]:
	#training(learning_rate=learning_rate, batch_size=40, epochs=60, mask=False)
	#training(learning_rate=learning_rate, batch_size=40, epochs=60, augmentation=False, mask=False)
	pass

for learning_rate in [0.0001]:
	#training(learning_rate=learning_rate, batch_size=40, epochs=60, adj_lr=False)
	#training(learning_rate=learning_rate, batch_size=40, epochs=60, augmentation=False, mask=False, adj_lr=False)
	pass

'''