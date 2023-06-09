
voc=dict()
with open('voc_5mers.txt','r') as f:
    c=5
    for line in f:
        items = line.rstrip().split(' ')
        label = items[0]
        voc[label]=c
        c+=1

with open('voc2.txt','w+') as f:
    for label in voc.keys():
        f.write("{} {}\n".format(label,voc[label]))