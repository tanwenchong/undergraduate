import numpy as np
import math
import pysam
import argparse
from data1 import *




parser = argparse.ArgumentParser()
parser.add_argument('-b','--bam_file')

args = parser.parse_args()
bam_file=args.bam_file

bam=pysam.AlignmentFile("{}".format(bam_file))
train_set=dict()

voc_num,voc_label=load_voc("voc.txt")
num=10000000
BLOCK=4
c=0
for contig in bam.references:
    c,a,train_set=get_pileups(bam_file,contig,num,BLOCK,train_set,c)

print(a)
np.save('full.npy',polish_set)

