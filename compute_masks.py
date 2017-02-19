import scipy.io, os
from scipy.misc import imsave
import numpy as np
import cPickle
from PIL import Image
import shutil
import matplotlib


def create_masks_for_model(annotated_model, mask_path, video_name, img_width, img_height, save_option):
  
 images_counter = 1
 file_name = annotated_model 
 annot = load_annotations_from_file(file_name)
 
 for x in range(0, len(annot)):
   arr = np.zeros((img_height, img_width))
   if (annot[x] <> 0):
     
     arr[annot[x][1]:annot[x][3], annot[x][0]:annot[x][2]] = 1

     MM = { 'PartMask' : arr}
     
     if save_option == "image":
       # save as image
       imsave(mask_path + video_name +"_{}.png".format(images_counter), arr) 
     elif save_option == "mat": 
       # save as mat file
       scipy.io.savemat(mask_path + video_name +"_{0}.mat".format(images_counter), mdict = {'MM': MM}, do_compression = True)
       
   else:
     # remove old masks not present in the new load_annotations
     if os.path.exists(mask_path + video_name +"_{}.png".format(images_counter)):
       os.remove(mask_path + video_name +"_{}.png".format(images_counter))
       #print "deleted :" + mask_path + video_name +"_{}.png".format(images_counter)
     
     if os.path.exists(mask_path + video_name +"_{}.mat".format(images_counter)):
       os.remove(mask_path + video_name +"_{}.mat".format(images_counter))
       #print "deleted :" + mask_path + video_name +"_{}.mat".format(images_counter)
     
   
   images_counter = images_counter + 1 
   if (x % 1000 == 0 and x>=1000):
     print images_counter
 
  
def load_annotations_from_file(file_name):
  f = file(file_name, 'rb')
  frame_rectangle_pairs = cPickle.load(f)
  f.close()
  return frame_rectangle_pairs


