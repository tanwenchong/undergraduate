import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import argparse
import torch.utils.data as Data
from einops import rearrange
from torch.autograd import Variable
from transformers import AutoModelForSeq2SeqLM
#from acc import *


parser = argparse.ArgumentParser()
parser.add_argument('-m','--model')
parser.add_argument('-i','--input_file')
parser.add_argument('-n','--need', default='False')
parser.add_argument('-q','--quast',default='False')
args = parser.parse_args()
model_dict=args.model
need=args.need
quast=args.quast
input_file=args.input_file
BATCH=1
BLOCK=4

train_set=dict()
label_set=dict()
print("get polish set")
model = AutoModelForSeq2SeqLM.from_pretrained("small.pth")
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
	seq=''
	for i in range(4):
		one=tgt[i]
		seq+=voc_num[int(one)]
	return seq


def polish(train_loader,model):
	corret=0
	wrong=0
	model.eval()
	with torch.no_grad():
		for step, (X,target) in enumerate(train_loader):
			#print('target={}'.format(target[0,1:5]))
			out=model.generate(X,max_length=7,min_length=0,eos_token_id=2)[0][2:6]
			#print(out)
			if list(target[0,1:5])==list(out):
				corret+=1
			else:
				wrong+=1
				#print(X)
				#print('target={}'.format(target[0,1:5]))
				#print('prdict={}'.format(out))

	return corret,wrong
num=40000
voc_num,voc_label=load_voc("voc5.txt")

if quast=='True':
	seq=''
	label_seq=''
	train_set = np.load('train_{}.npy'.format(input_file),allow_pickle=True).item()
	label_set = np.load('label_{}.npy'.format(input_file),allow_pickle=True).item()	
	x=torch.LongTensor(np.array([train_set[i] for i in range(num)]))
	y=torch.LongTensor(np.array([label_set[i] for i in range(num)]))
	train_dataset = Data.TensorDataset(x, y)
	train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH,shuffle=False,num_workers=2,)
	with torch.no_grad():
		for step, (X,target) in enumerate(train_loader):
			predict=model.generate(X,max_length=7,min_length=0,eos_token_id=2)[0][2:6].tolist()
			seq+=label(predict)
			label_seq+=label(target[0,1:5].tolist())
	with open ("quast_seq.fasta","w+") as f:
		f.write(">{}\n".format('contig1'))
		f.write("{}\n".format(seq))	
	with open ("quast_ref.fasta","w+") as f:
		f.write(">{}\n".format('contig1'))
		f.write("{}\n".format(label))	
	
else:

	if need=='False':
		train_set = np.load('train_{}.npy'.format(input_file),allow_pickle=True).item()
		label_set = np.load('label_{}.npy'.format(input_file),allow_pickle=True).item()
	
		x=torch.LongTensor(np.array([train_set[i] for i in range(num-500,num)]))
		y=torch.LongTensor(np.array([label_set[i] for i in range(num-500,num)]))
		train_dataset = Data.TensorDataset(x, y)
		train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH,shuffle=False,num_workers=2,)
		corret,wrong=polish(train_loader,model)
		print('corret={},wrong={},rate={}'.format(corret,wrong,corret/(corret+wrong)))
	else:
		corret=0
		wrong=0
		need_set=np.load('need_{}.npy'.format(input_file),allow_pickle=True).item()
		model.eval()
		for Y in need_set.keys():
			X=need_set[Y]
			X=torch.LongTensor(X)
			X=torch.unsqueeze(X, 0)
			predict=model.generate(X,max_length=7,min_length=0,eos_token_id=2)[0][2:6]
			#print(Y,predict)
			if list(Y)==list(predict):
				corret+=1
			else:
				wrong+=1
				print(X)
				print('target={}'.format(label(Y)))
				print('prdict={}'.format(label(predict.tolist())))
		print('corret={},wrong={},rate={}'.format(corret,wrong,corret/(corret+wrong)))
		


		


