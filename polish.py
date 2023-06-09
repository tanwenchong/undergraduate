import numpy as np
import torch
import argparse
import torch.utils.data as Data
from einops import rearrange
from torch.autograd import Variable
from transformers import AutoModelForSeq2SeqLM

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model')
parser.add_argument('-f','--file')
args = parser.parse_args()
model_dict=args.model
imput_file=args.file
model = AutoModelForSeq2SeqLM.from_pretrained("small.pth")
model.load_state_dict(torch.load(model_dict))
print("load model")

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
		try:
			one=tgt[i]
			seq+=voc_num[int(one)]
		except:
			return seq
	return seq


def polish(polish_set,model):
	model.eval()
	seq=''
	for X in polish_set.values():
		X=torch.LongTensor(X)
		X=torch.unsqueeze(X, 0)
		predict=model.generate(X,max_length=7,min_length=0,eos_token_id=2)[0][2:6]
		seq+=label(predict)
	return seq
				

voc_num,voc_label=load_voc("voc5.txt")
polish_set = np.load(imput_file,allow_pickle=True).item()
seq=polish(polish_set,model)
with open ("polish.fasta","w+") as f:
	f.write(">{}\n".format('contig1'))
	f.write("{}\n".format(seq))
print('finish')
