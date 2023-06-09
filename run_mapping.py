import subprocess
def bam_mapping(raw,ass,output):
    subprocess.run('minimap2 -x map-ont -a {} {} > {}.sam'.format(ass,raw,output),shell=True)
    subprocess.run('samtools view -b {}.sam >{}.bam'.format(output,output),shell=True)
    subprocess.run('samtools sort {}.bam >{}s.bam'.format(output,output),shell=True)
    subprocess.run('samtools index {}s.bam'.format(output),shell=True)

    bam_file='{}s.bam'.format(output)
    return bam_file

def label_mapping(ref,ass,output):
    subprocess.run('minimap2 -a {} {} > label_{}.sam'.format(ass,ref,output),shell=True)
    subprocess.run('samtools view -b label_{}.sam >label_{}.bam'.format(output,output),shell=True)
    subprocess.run('samtools sort label_{}.bam > label_{}s.bam'.format(output,output),shell=True)
    subprocess.run('samtools index label_{}s.bam'.format(output),shell=True)

    label_bam='label_{}s.bam'.format(output)
    return label_bam