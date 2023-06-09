import numpy as np
import math
import pysam
import argparse
from data1 import *
from tdata1 import *
from run_mapping import *


parser = argparse.ArgumentParser()
parser.add_argument('-b','--bam_file',default=None)
parser.add_argument('-l','--label_file',default=None)
parser.add_argument('-o','--output_file',default=None)
parser.add_argument('-i','--input_file',default=None)
parser.add_argument('-m','--map',default=False)
parser.add_argument('--ref',default=None)
parser.add_argument('--raw',default=None)
parser.add_argument('--ass',default=None)
args = parser.parse_args()
output_file=args.output_file
if args.map==False:
    if args.input_file==None:
        bam_file=args.bam_file
        label_bam=args.label_file
    else:
        bam_file='{}s.bam'.format(args.input_file)
        label_bam='label_{}s.bam'.format(args.input_file)
else:
    bam_file=bam_mapping(args.raw,args.ass,args.output_file)   
    label_bam=label_mapping(args.ref,args.ass,args.output_file)


bam=pysam.AlignmentFile("{}".format(bam_file))
label=pysam.AlignmentFile("{}".format(label_bam))
train_set=dict()
label_set=dict()
voc_num,voc_label=load_voc("voc5.txt")
num=10000000
BLOCK=4
need_set={}

c=0
d=0
print(bam.references)
for contig in bam.references:
    mark=0
    pb=bam.pileup(contig=contig)
    pl=label.pileup(contig=contig)
    print(next(pb).reference_pos,next(pl).reference_pos)
    if next(pb).reference_pos>next(pl).reference_pos:
        mark=next(pb).reference_pos-next(pl).reference_pos
        c,a,train_set=get_pileups(bam_file,contig,num,BLOCK,train_set,c)
        d,label_set,need=get_labels(label_bam,contig,num,BLOCK,label_set,d,mark)
    if next(pb).reference_pos<next(pl).reference_pos:
        mark=next(pl).reference_pos-next(pb).reference_pos
        c,a,train_set=get_pileups(bam_file,contig,num,BLOCK,train_set,c,mark)
        d,label_set,need=get_labels(label_bam,contig,num,BLOCK,label_set,d)   
    if next(pb).reference_pos==next(pl).reference_pos:
        c,a,train_set=get_pileups(bam_file,contig,num,BLOCK,train_set,c)
        d,label_set,need=get_labels(label_bam,contig,num,BLOCK,label_set,d)
    
    if c==num:
        break

print(c)

for i in need:
    try:
        need_set[tuple(label_set[i][1:5])]=train_set[i]
    except:
        continue


np.save('train_{}.npy'.format(output_file),train_set)
np.save('label_{}.npy'.format(output_file),label_set)
np.save('need_{}.npy'.format(output_file),need_set)