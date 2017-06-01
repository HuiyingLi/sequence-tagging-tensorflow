#!/usr/bin/python
import pdb
import sys

data = sys.stdin.readlines()
#data = open("eval.txt").readlines()
correct = 0
totalpred = 0
totalgold = 0
def get_chunks(tags):
    res=set()
    i=0
    while i<len(tags):
        l=tags[i].strip()
        if l.startswith('S'):
            res.add(l+" "+str(i)+" "+str(i+1))
            i+=1
        elif l.startswith('B'):
            b=i
            while i<len(tags) and not tags[i].startswith('E'):
                i+=1
            i+=1
            res.add(tags[i].strip()+" "+str(b)+" "+str(i))
        else:
            i+=1
    return res

predl=[l.split(" ")[4] for l in data]
goldl=[l.split(" ")[3] for l in data]

c1 = get_chunks(predl)
c2 = get_chunks(goldl)

for c in c1:
    if c in c2:
        correct+=1

precision = correct*1.0/len(c1)
recall = correct*1.0/len(c2)
print "This script evaluates P/R/F of BIOES tagged NER"
print "Precision:", precision, "Recall:", recall, "F1:", 2*precision*recall/(precision+recall)
