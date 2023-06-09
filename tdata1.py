import pysam
import numpy as np


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
	
voc_num,voc_label=load_voc("voc5.txt")

def change_seq(seq):
	seqcode=list()
	for i in seq:
		seqcode.append(voc_label[i])
	return seqcode
def get_labels(bam_file,contigs,num,BLOCK,label_set,c,mark=0):
	bam=pysam.AlignmentFile("{}".format(bam_file))
	pileupcolumn=bam.pileup(contig=contigs)
	voc=label_set
	baselist=list()
	baselist.append(voc_label['B'])
	fine=list()

	for base in pileupcolumn:
		if mark>0:
			mark-=1
			continue
		is_fine=False
		if c==num:
			return c,voc,fine
		pileupreads=base.get_query_sequences(add_indels=True)
		read=pileupreads[0]
		#if c==0:
		#	print(pileupreads,base.reference_pos)
		if read in ['A','C','T','G','a','c','t','g','*']:
				if read == '*':
					baselist.append(voc_label['N'])
					is_fine=True			
				else:
					baselist.append(voc_label[read.upper()])			
		else:
			is_fine=True
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

		if len(baselist)==BLOCK+1:
			baselist.append(voc_label['E'])
			#print(baselist)
			voc[c]=baselist
			if is_fine==True:
				fine.append(c)
			c+=1
			baselist=list()
			baselist.append(voc_label['B'])
	print(c)
	return c,voc,fine

		
        

