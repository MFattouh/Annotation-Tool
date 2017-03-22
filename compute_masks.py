import scipy.io, os
from scipy.misc import imsave
import numpy as np
import cPickle
from PIL import Image
import shutil
import matplotlib.pyplot as plt

def create_masks_for_model(annotated_model, mask_path, video_name, img_width, img_height, save_option):
  images_counter = 1
  file_name = annotated_model 
  annotations = load_annotationsations_from_file(file_name)
 
  # loop through each frame
  for frame in range(0, len(annotations)):
    # initialize zero array as big as the frame size
    arr = np.zeros((img_height, img_width))
   
    # for each label in the current frame
    if annotations[frame] != 0:
      for label in range(0,len(annotations[frame])):
	if (annotations[frame][label] <> 0):
	  arr[annotations[frame][label][1]:annotations[frame][label][3], annotations[frame][label][0]:annotations[frame][label][2]] = annotations[frame][label][-1]

	  MM = { 'PartMask' : arr}
	
	  if save_option == "image":
	    # save as image
	    imsave(mask_path + video_name +"_{}.png".format(images_counter), arr) 
	  elif save_option == "mat": 
	    # save as mat file
	    scipy.io.savemat(mask_path + video_name +"_{0}.mat".format(images_counter), mdict = {'MM': MM}, do_compression = True)
	  
    else:
      # remove old masks not present in the new load_annotationsations
      if os.path.exists(mask_path + video_name +"_{}.png".format(images_counter)):
	os.remove(mask_path + video_name +"_{}.png".format(images_counter))
	#print "deleted :" + mask_path + video_name +"_{}.png".format(images_counter)
      
      if os.path.exists(mask_path + video_name +"_{}.mat".format(images_counter)):
	os.remove(mask_path + video_name +"_{}.mat".format(images_counter))
	#print "deleted :" + mask_path + video_name +"_{}.mat".format(images_counter)
	
   
    images_counter = images_counter + 1 
    if (frame % 1000 == 0 and frame>=1000):
      print images_counter
      
def create_mask_for_image(image_array, image_annotation, label_number):
  image_height = image_array.shape[0]
  image_width = image_array.shape[1]
  
  # background is 1
  image_array_only_mask = np.ones_like(image_array)*255
  
  # Do nothing if there is no annotations
  if image_annotation == 0:
    print "No annotations for current image"
    return image_array_only_mask
    
  
  number_of_annotations = len(image_annotation)
  label_number_index = -1
  
  if label_number == 0:
    for annotation in image_annotation:
      image_array_only_mask[annotation[1]:annotation[3],annotation[0]:annotation[2]] = image_array[annotation[1]:annotation[3],annotation[0]:annotation[2]]
  
  else:
    # get the index of the label_number
    for i in range(0, number_of_annotations):
      if label_number == image_annotation[i][-1]:
	label_number_index = i
	break
      
    # check if the their is annotation for the given label_number
    if label_number_index == -1:
      print "No annotations for label number {} for current image".format(label_number)
      
    # make an image array with white background and only the annotated labels showing
    else:  
      image_array_only_mask[image_annotation[label_number_index][1]:image_annotation[label_number_index][3],image_annotation[label_number_index][0]:image_annotation[label_number_index][2]] = image_array[image_annotation[label_number_index][1]:image_annotation[label_number_index][3],image_annotation[label_number_index][0]:image_annotation[label_number_index][2]]     
      
  return image_array_only_mask 
        
  
def load_annotationsations_from_file(file_name):
  f = file(file_name, 'rb')
  frame_rectangle_pairs = cPickle.load(f)
  f.close()
  return frame_rectangle_pairs


