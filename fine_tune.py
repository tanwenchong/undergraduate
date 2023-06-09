import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import argparse
import torch.utils.data as Data
from einops.layers.torch import Rearrange
from einops import rearrange
from transformers import get_linear_schedule_with_warmup
from torch.autograd import Variable
from transformers import AutoModelForSeq2SeqLM
from acc import *
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
parser.add_argument('-i','--input_file')
args = parser.parse_args()
mode=args.mode
input_file=args.input_file
train_set=dict()
label_set=dict()
voc_num,voc_label=load_voc("voc5.txt")
num=40000
BLOCK=4
MODEL_NAME='5mer.pkl'

train_set = np.load('train_{}.npy'.format(input_file),allow_pickle=True).item()
label_set = np.load('label_{}.npy'.format(input_file),allow_pickle=True).item()

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

BATCH=10
epoch_num=100
total_steps=epoch_num*(num-500)/BATCH
learning_rate=5e-6
model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum")
model, list_en, list_de = create_student_by_copying_alternating_layers(model, 'small.pth', 12, 3)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=.0004)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = total_steps*0.1, num_training_steps = total_steps)


x=torch.LongTensor(np.array([train_set[i] for i in range(num-4000)]))
y=torch.LongTensor(np.array([label_set[i] for i in range(num-4000)]))
X=torch.LongTensor(np.array([train_set[i] for i in range(num-4000,num)]))
Y=torch.LongTensor(np.array([label_set[i] for i in range(num-4000,num)]))


train_dataset = Data.TensorDataset(x, y)
valid_dataset=Data.TensorDataset(X, Y)
train_set=None
label_set=None
train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH,shuffle=False,num_workers=0,)
valid_loader = Data.DataLoader(dataset=valid_dataset,batch_size=BATCH,shuffle=False,num_workers=0,)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def train(model,optimizer,loader):
	total_loss=0
	model.train()
	for step, (X,target) in enumerate(loader):

		outputs=model(X,labels=target,attention_mask=(X != 1))
		loss=outputs.loss
		total_loss+=loss.item()
		loss.backward()
		optimizer.step()
		scheduler.step()
		optimizer.zero_grad()
	return total_loss

def evaluate(model,loader):
	total_loss =0
	model.eval()
	with torch.no_grad():
		for step, (X,target) in enumerate(loader):

			outputs=model(X,labels=target,attention_mask=(X != 1))
			loss=outputs.loss
			total_loss+=loss.item()
	return total_loss
save_loss=0
if mode=='first':
	for epoch in range(epoch_num):
		train_loss = train(model, optimizer,train_loader)
		valid_loss = evaluate(model,valid_loader)
		print('epoch={},train_loss={},valid_loss={}'.format(epoch,train_loss,valid_loss))
		if save_loss==0:
			save_loss=valid_loss
		if valid_loss < save_loss:
			torch.save(model.state_dict(),MODEL_NAME)
			save_loss=valid_loss

		
	torch.save(model.state_dict(),MODEL_NAME)
if mode=='continue':
	model.load_state_dict(torch.load(MODEL_NAME))
	for epoch in range(1000):
		train_loss = train(model, optimizer,train_loader)
		if epoch % 10==0:
			valid_loss = evaluate(model, valid_loader)
			torch.save(model.state_dict(),MODEL_NAME)
			print('epoch={},train_loss={},valid_loss={},lr={}'.format(epoch,train_loss,valid_loss,scheduler.get_last_lr()[0]))
		else:
			print('epoch={},train_loss={}'.format(epoch,train_loss))
