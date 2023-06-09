import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import argparse

import torch.utils.data as Data
from einops import rearrange
from lite import *
from torch.autograd import Variable
parser = argparse.ArgumentParser()
parser.add_argument('-m','--model')
parser.add_argument('-n','--need', default='False')

args = parser.parse_args()
model_dict=args.model
need=args.need
BATCH=1
BLOCK=4

train_set=dict()
label_set=dict()
print("get polish set")
model=transformer_model(BATCH=1,d_model=256).eval()
model.load_state_dict(torch.load(model_dict))
print("succeed loading model")


def load_voc(path):
	voc_num=dict()
	voc_label=dict()
	with open(path,'r') as f:
		for line in f:
			items = line.rstrip().split(' ')
			label = items[0]
			number = int(items[1])
			voc_num[number]=label
			voc_label[label]=number
	return voc_num,voc_label

def label(tgt):
	tgt=tgt[0]
	seq=''
	for i in range(6):
		one=tgt[i]
		seq+=voc_num[int(one[i])]
	return seq

def greedy_decode(model,matrix):	
	ys=torch.ones(1,1).fill_(0).type_as(matrix.data).long()
	for i in range(4):
		out=model(matrix,ys)
		out=model.projection(out[:, -1])
#		print(out)
		_,next_base=torch.max(out,dim=1)
		next_base=next_base.data[0]
#		print(next_base)
		next_base=torch.ones(1,1).fill_(next_base).long()
		ys = torch.cat([ys, next_base], dim=1)
	ys = ys[0, 1:]
	return ys


def polish(train_loader,model):
	corret=0
	wrong=0
	model.eval()
	with torch.no_grad():
		for step, (X,target) in enumerate(train_loader):
			#print(X)
			#print('target={}'.format(target[0,1:5]))
			ys=greedy_decode(model,X)
			#print('result={}'.format(ys))
			if list(target[0,1:5])==list(ys):
				corret+=1
			else:
				wrong+=1
	return corret,wrong
num=20000
voc_num,voc_label=load_voc("voc5.txt")

if need=='False':
	train_set = np.load('train_be.npy',allow_pickle=True).item()
	label_set = np.load('label_be.npy',allow_pickle=True).item()
	
	x=torch.LongTensor(np.array([train_set[i] for i in range(num-500,num)]))
	y=torch.LongTensor(np.array([label_set[i] for i in range(num-500,num)]))
	train_dataset = Data.TensorDataset(x, y)
	train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH,shuffle=False,num_workers=2,)
	corret,wrong=polish(train_loader,model)
	print('corret={},wrong={},rate={}'.format(corret,wrong,corret/(corret+wrong)))

if need == 'True':
	corret=0
	wrong=0
	need_set=np.load('need.npy',allow_pickle=True).item()
	model.eval()
	for Y in need_set.keys():
		X=need_set[Y]
		X=torch.LongTensor(X)
		X=torch.unsqueeze(X, 0)
		ys=greedy_decode(model,X)
		if list(Y)==list(ys):
			corret+=1
		else:
			wrong+=1
	print('corret={},wrong={},rate={}'.format(corret,wrong,corret/(corret+wrong)))