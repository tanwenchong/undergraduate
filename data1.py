import pysam
import numpy as np
import torch

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


def change_seq(seq):
	seqcode=list()
	for i in seq:
		seqcode.append(vocab[i])
	return seqcode
def get_pileups(bam_file,contigs,num,BLOCK,train_set,c,mark=0):
	voc_num,voc_label=load_voc("voc5.txt")
	bam=pysam.AlignmentFile("{}".format(bam_file))
	print("start get_pileups")
	a=np.ones(shape=(20,BLOCK))
	GO=0
	train_set=dict()
	pileupcolumn=bam.pileup(contig=contigs)
	for base in pileupcolumn:
		if mark>0:
			mark-=1
			print(base.reference_pos)
			continue
		base.set_min_base_quality(60)#质控1
		baselist=list()
		pileupreads=base.get_query_sequences(add_indels=True)
		allreads=base.pileups
		read_num=0
		for read in pileupreads:
			#质控2
			if allreads[read_num].alignment.reference_length/allreads[read_num].alignment.query_length<0.95 or allreads[read_num].alignment.query_length/allreads[read_num].alignment.reference_length<0.95:
				read_num+=1
				continue
			else:
				read_num+=1
			if len(baselist)<len(pileupreads) and len(baselist)<20:
				if read in ['A','C','T','G','a','c','t','g','*']:
					if read == '*':
						baselist.append(voc_label['N'])		
					else:
						baselist.append(voc_label[read.upper()])
		
				else:
					if len(read)>1:
						if read[1]=='-' and read[0] in ['A','C','T','G','a','c','t','g']:
							baselist.append(voc_label[read[0].upper()])
						if read[1]=='+' and read[0] in ['A','C','T','G','a','c','t','g']:
							if read[3:].isalpha()==True:
								if len(read[3:])<4:
									baselist.append(voc_label[read[0].upper()+read[3:].upper()])
								else:
									baselist.append(voc_label[read[0].upper()+read[3:7].upper()])
							else:
								baselist.append(voc_label[read[0].upper()])

			if len(baselist)==len(pileupreads) or len(baselist)==20:
				b=np.array(baselist)
				b=b.reshape(-1,1)
				if b.shape[0]<20:
					a[:b.shape[0]-20,[GO]]=b
				else:
					a[:,[GO]]=b

				GO+=1
				baselist=list()
				break
				  
		if GO==BLOCK:
			GO=0
			p=np.array(torch.ones(20,1).fill_(voc_label['E']).long())
			a=np.append(a,p,axis=1)
			a=a.flatten()
			a=np.insert(a,0,[voc_label['B']])
			train_set[c]=a
			c+=1
			#if base.reference_pos==7:          #canu专用
			#	c-=1
			a=np.ones(shape=(20,BLOCK))
			pas=list()
			if c==num:
				return c,GO,train_set
	if  GO!=0:
		return c,a,train_set
        

