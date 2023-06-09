import pysam
import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_file')
args = parser.parse_args()
input_file=args.input_file

ass_fasta=pysam.FastaFile(input_file)
begin=0
num=40000*4
for contig in ass_fasta.references:
	seq=ass_fasta.fetch(reference=contig,start=begin,end=num)
with open ("quast_ass.fasta","w+") as f:
	f.write(">{}\n".format('contig1'))
	f.write("{}\n".format(seq))	

subprocess.run('quast quast_seq.fasta quast_ass.fasta -r quast_ref.fasta',shell=True)