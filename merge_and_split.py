import os
import numpy as np

models_path = os.path.abspath("/home/deeplearning/Documents/paw_tracking_caffe/models/")
saving_path_pos = os.path.abspath("/home/deeplearning/Documents/paw_tracking_caffe/models/pos.txt") 
saving_path_neg = os.path.abspath("/home/deeplearning/Documents/paw_tracking_caffe/models/neg.txt") 

"""
for directory in os.walk(models_path):
  print os.path.abspath(directory[0])
"""
counter_pos = 0
counter_neg = 0
list_positives = []
list_negatives = []
outfilepos =  open(saving_path_pos, 'w')
outfileneg = open(saving_path_neg, 'w')
for f in os.listdir(models_path):
  if os.path.isdir(os.path.join(models_path, f)) is True:
    fname = os.path.join(models_path, f, "pos.txt")
    if os.path.isfile(fname):
      with open(fname, 'r') as infilepos:
	for line in infilepos:
	  #outfilepos.write(os.path.join(models_path, f, line))
	  outfilepos.write(os.path.join(f, line))
	  list_positives.append(os.path.join(f, line))
    fname = os.path.join(models_path, f, "neg.txt")
    if os.path.isfile(fname):
      with open(os.path.join(models_path, f, "neg.txt"), 'r') as infileneg:
	for line in infileneg:
	  #outfileneg.write(os.path.join(models_path, f, line))
	  outfileneg.write(os.path.join(f, line))
	  list_negatives.append(os.path.join(f, line))
	
outfilepos.close()
outfileneg.close()
  
np.random.shuffle(list_positives)
np.random.shuffle(list_negatives)

cut = int(0.75*len(list_positives))
print cut
with open(os.path.join(models_path, "train.txt"),'w') as models:
  models.writelines(list_positives[0:cut])
  models.writelines(list_negatives[0:cut])
  
with open(os.path.join(models_path, "val.txt"),'w') as models:
  models.writelines(list_positives[cut+1:cut+1+len(list_positives)-cut])
  models.writelines(list_negatives[cut+1:cut+1+len(list_positives)-cut])
