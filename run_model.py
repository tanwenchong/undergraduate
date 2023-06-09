import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import argparse
from lite import *
import torch.utils.data as Data
from einops.layers.torch import Rearrange
from einops import rearrange
from transformers import get_linear_schedule_with_warmup
from torch.autograd import Variable

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

parser = argparse.ArgumentParser()
parser.add_argument('-m','--mode')
args = parser.parse_args()
mode=args.mode
train_set=dict()
label_set=dict()
voc_num,voc_label=load_voc("voc.txt")
num=10000
BLOCK=4
MODEL_NAME='transformer.pkl'

train_set = np.load('train_be.npy',allow_pickle=True).item()
label_set = np.load('label_be.npy',allow_pickle=True).item()

print("get train set")
print("get label data")

def greedy(output):
    for j in range(BATCH):
        one=output[j]
        seq=""
        _,next_base=torch.max(one,dim=1)
        for i in next_base:
            seq+=voc_num[int(i)]
        print(seq)
def label(tgt):
    for j in range(BATCH):
        seq=""
        one=tgt[j]
        for i in range(BLOCK):
            seq+=voc_num[int(one[i])]
        print(seq)

BATCH=50
learning_rate=5e-5
d_model=256
tgt_size=25
model=transformer_model(d_model=d_model,tgt_size=tgt_size)

criterion= nn.CrossEntropyLoss()
epoch_num=100
total_steps=epoch_num*(num-500)/BATCH
#optimizer = torch.optim.AdamW(model.parameters(), lr=0,betas=(0.9, 0.98), eps=1e-9)
#model_opt = NoamOpt(d_model, 1, 400, optimizer)
#criterion = LabelSmoothing(size=tgt_size, padding_idx=1, smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=.0004)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = total_steps*0.1, num_training_steps = total_steps)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

x=torch.LongTensor(np.array([train_set[i] for i in range(num-2000)]))
y=torch.LongTensor(np.array([label_set[i] for i in range(num-2000)]))
X=torch.LongTensor(np.array([train_set[i] for i in range(num-2000,num)]))
Y=torch.LongTensor(np.array([label_set[i] for i in range(num-2000,num)]))


train_dataset = Data.TensorDataset(x, y)
valid_dataset=Data.TensorDataset(X, Y)
train_set=None
label_set=None
train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH,shuffle=False,num_workers=0,)
valid_loader = Data.DataLoader(dataset=valid_dataset,batch_size=BATCH,shuffle=False,num_workers=0,)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def train(model,criterion,optimizer,loader,scheduler):
	total_loss=0
	model.train()
	for step, (X,target) in enumerate(loader):
		X=X.to(device)
		target=target.to(device)
		optimizer.zero_grad()
		output=model(X,target[0:,:-1])
		output=model.projection(output)
		#loss = criterion(output.log_softmax(-1).contiguous().view(-1,output.size(-1)),target[0:,1:].contiguous().view(-1))
		loss = criterion(output.contiguous().view(-1,output.size(-1)),target[0:,1:].contiguous().view(-1))
		model.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
		optimizer.step()
		scheduler.step()
		total_loss+=loss
	return total_loss

def evaluate(model, criterion, loader):
	total_loss =0
	model.eval()
	with torch.no_grad():
		for step, (X,target) in enumerate(loader):
			output=model(X,target[0:,:-1])
			output=model.projection(output)
			#norm = (target != 0).sum()
			loss = criterion(output.contiguous().view(-1,output.size(-1)),target[0:,1:].contiguous().view(-1))
			#loss = criterion(output.log_softmax(-1).contiguous().view(-1,output.size(-1)),target[0:,1:].contiguous().view(-1))
			total_loss+=loss
	return total_loss
save_loss=0

if mode=='first':
	for epoch in range(epoch_num):
		train_loss = train(model,criterion, optimizer,train_loader,scheduler)
		valid_loss = evaluate(model, criterion, valid_loader)
		print('epoch={},train_loss={},valid_loss={},lr={}'.format(epoch,train_loss,valid_loss,scheduler.get_last_lr()[0]))
		if save_loss==0:
			save_loss=valid_loss
		if valid_loss < save_loss:
			torch.save(model.state_dict(),MODEL_NAME)
			save_loss=valid_loss
		
	torch.save(model.state_dict(),MODEL_NAME)

