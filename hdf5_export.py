import h5py  # HDF5 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys 
import os
from skimage import io
from skimage import color
from skimage.transform import rotate
from skimage.transform import rescale
import glob
import Image, ImageEnhance
import random


from compute_masks import load_annotationsations_from_file
# script for downsampling and producing hd5f for frames(data) and the masks(labels)

# debugging flags
do_plot = False
verbose = False

extract_shape = False  # threshold for white background 
extract_th    = 253    # Threshold 

# function to calculate the mean over all frames
def calculate_mean_all_frames(color, num_all_frames, frames_folder_path):  
  if color:
    suma = [0,0,0]
  else:
    suma = [0]
  for root, dirs, files in os.walk(frames_folder_path):
    for file in files:
      if color:
	img = cv2.imread(os.path.join(frames_folder_path,file))
	width = img.shape[1]
	height = img.shape[0]
	
	suma[0] += np.sum(img[:,:,0]) / (width*height)
	suma[1] += np.sum(img[:,:,1]) / (width*height)
	suma[2] += np.sum(img[:,:,2]) / (width*height)
      else:
	img = cv2.imread(os.path.join(frames_folder_path,file),0)
	width = img.shape[1]
	height = img.shape[0]
	
	suma[0] += np.sum(img[:,:]) / (width*height)      
  mean = np.array(suma) / num_all_frames  
  return mean


def generate_hd5f(TARGET_X_DIM, TARGET_Y_DIM, num_all_frames, annotation_folder_path, frames_folder_path, output_folder_path, num_scales=0, num_rotations=0, num_colors=0, color=False):
  
  # calculate the mean over all pixels in all frames
  mean_value = calculate_mean_all_frames(color, num_all_frames, frames_folder_path)
  
  if extract_shape:
    if extract_th - mean_value <= 1:
      print "STOP:\n mean value too close to threshold extraction value"
      quit() 
  
  # mask shape = frame shape
  TARGET_MASK_X_DIM = TARGET_X_DIM
  TARGET_MASK_Y_DIM = TARGET_Y_DIM
  
  # get list of annotation models
  annotation_list_full_path = glob.glob(os.path.join(annotation_folder_path, "*.model"))
  
  annotation_list_name =[]
  annotation_list_rectangle_pairs = []
  max_examples = 0
  for i,annotation_path in enumerate(annotation_list_full_path):
    # add annotations to the list
    annotation_list_rectangle_pairs.append(load_annotationsations_from_file(annotation_path))
    # add the annotation name to list
    annotation_list_name.append(os.path.basename(annotation_path))
    max_examples += sum(1 for x in annotation_list_rectangle_pairs[i] if x != 0)
  
  print annotation_list_name
  print "Number of models", len(annotation_list_rectangle_pairs)
  print "Number of examples with annotations ", max_examples
  
  # adding number of augmentations
  max_examples = max_examples + num_rotations + num_colors + num_scales
  
  # initialize structure for the data(frames) and the labels(masks) in hdf5 file 
  if color:
    colorstr = "_color"
    data_set = np.zeros((max_examples,3,TARGET_Y_DIM,TARGET_X_DIM),dtype=np.float32)
    label_set = np.zeros((max_examples,3,TARGET_MASK_Y_DIM,TARGET_MASK_X_DIM),dtype=np.float32)
  else:
    colorstr = ""
    data_set = np.zeros((max_examples,1,TARGET_Y_DIM,TARGET_X_DIM),dtype=np.float32)
    label_set = np.zeros((max_examples,1,TARGET_MASK_Y_DIM,TARGET_MASK_X_DIM),dtype=np.float32)


  # check if hdf5 folder exists
  hdf5_folder = os.path.join(output_folder_path,"hdf5_files")
  if not os.path.exists(hdf5_folder):
    os.makedirs(hdf5_folder)
    
  # hdf5 file name
  fname = hdf5_folder+"/20170219_meansub_downsample_"+str(TARGET_X_DIM)+"x"+str(TARGET_Y_DIM)+"_MASK_"+str(TARGET_MASK_X_DIM)+"x"+str(TARGET_MASK_Y_DIM)+colorstr+".hdf5"  
  
  # open hdf4 file to write
  f=h5py.File(fname,"w")

  fid = 0
  
  # loop for each annotation model
  # downsample + putting data and label in the corresponding structures for hdf5 file
  print "---------------------- Begin Augmentation -------------------------"
  for index, (annotation_video,annotation_video_name) in enumerate(zip(annotation_list_rectangle_pairs,annotation_list_name)): 
    
    # add frames and masks to hdf5 if only both the frame and its annotation exists
    # otherwise u have a missing data (either the frame or its annotation)
    video_name = os.path.splitext(annotation_video_name)[0]
    print "--------------"
    print "-Video: {}\n".format(video_name) 
    for frame_number, annotation_frame in enumerate(annotation_video):
      # ignore data if frame is not annotated or the frame doesn't exist but its annotation exists
      frame_name = video_name+ "_" + str(frame_number+1) + ".png"
      frame_path = os.path.join(frames_folder_path,frame_name)
      
      if annotation_frame != 0 and os.path.exists(frame_path):
	print "- Frame {} with label done".format(frame_number+1)
	# read the frame to arrays
	if color:
	  frame = cv2.imread(frame_path)
	  # mean substraction
	  frame[:,:,0] = frame[:,:,0] - mean_value[0]
	  frame[:,:,1] = frame[:,:,1] - mean_value[1]
	  frame[:,:,2] = frame[:,:,2] - mean_value[2]
	  
	else:
	  frame = cv2.imread(frame_path,0)
	  # mean substraction
	  frame[:,:] = frame[:,:] - mean_value[0]
	
	height = frame.shape[0]
	width = frame.shape[1]
	
	mask = np.zeros((height,width))

	# creating the mask array for the current frame
	# for each label in the current frame
	for label in range(0,len(annotation_frame)):
	  mask[annotation_frame[label][1]:annotation_frame[label][3], annotation_frame[label][0]:annotation_frame[label][2]] = annotation_frame[label][-1]
	#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
	# Color
	if num_colors != 0:
	  
	  if color:
	    frame_contrasted_set = np.zeros((num_colors,3,TARGET_Y_DIM,TARGET_X_DIM),dtype=np.float32)
	  else:
	    frame_contrasted_set = np.zeros((num_colors,1,TARGET_Y_DIM,TARGET_X_DIM),dtype=np.float32)
	  mask_contrasted_set = np.zeros((num_colors,1,TARGET_MASK_Y_DIM,TARGET_MASK_X_DIM),dtype=np.float32)
	  
	  # resize mask here as it doesn't change afterwards
	  mask_contrasted = cv2.resize(mask,(TARGET_MASK_X_DIM,TARGET_MASK_Y_DIM))
	  mask_contrasted = mask_contrasted.astype(dtype=np.float32)
	  
	  # add the mask to the mask set
	  mask_contrasted_set[:,0,:,:] = mask_contrasted
	  
	  for i in range(0,num_colors):
	    
	    # set random contrast parameters
	    contrast_random_number = random.uniform(1,40)
	    tile_random_number = random.randint(10,100)
	    
	    # apply contrast
	    clahe = cv2.createCLAHE(clipLimit=contrast_random_number, tileGridSize=(tile_random_number,tile_random_number))
	    
	    # get the new contrrasted frame
	    if color:
	      frame_contrasted = np.zeros_like(frame)
	      frame_contrasted[:,:,0] = clahe.apply(frame[:,:,0])
	      frame_contrasted[:,:,1] = clahe.apply(frame[:,:,1])
	      frame_contrasted[:,:,2] = clahe.apply(frame[:,:,2])
	    else:
	      frame_contrasted = clahe.apply(frame)
	    
	    # downsampling the frame
	    frame_contrasted = cv2.resize(frame_contrasted,(TARGET_X_DIM,TARGET_Y_DIM))
	    
	    # cast to float32 for Caffe
	    frame_contrasted = frame_contrasted.astype(dtype=np.float32) 
	    
	    # add the frame the set
	    if color:
	      frame_contrasted_set[i,0,:,:] = frame_contrasted[:,:,0]
	      frame_contrasted_set[i,1,:,:] = frame_contrasted[:,:,1]
	      frame_contrasted_set[i,2,:,:] = frame_contrasted[:,:,2]
	    
	    else: 
	      frame_contrasted_set[i,0,:,:] = frame_contrasted
   	    
	# debugging after augmenting color is finished
	if False:
	  print "-------------------------------------------"
	  print "CONTRAST augmentation"
	  print "-Frame set shape", frame_contrasted_set.shape
	  print "-Mask set shape", mask_contrasted_set.shape
	  for i in range(0,num_colors):
	    plt.figure
	    if color:
	      frame_debug = np.zeros((frame_contrasted_set.shape[2],frame_contrasted_set.shape[3],frame_contrasted_set.shape[1]))
	      frame_debug[:,:,0] = frame_contrasted_set[i,0,:,:]
	      frame_debug[:,:,1] = frame_contrasted_set[i,1,:,:]
	      frame_debug[:,:,2] = frame_contrasted_set[i,2,:,:]
	      
	      plt.subplot(121)
	      plt.imshow(frame_debug)
	      
	    else:
	      frame_debug = frame_contrasted_set[i,0,:,:]
	      plt.subplot(121)
	      plt.imshow(frame_debug,cmap='gray')
	      
	    mask_debug = np.copy(mask_contrasted_set[i,0,:,:])
	      
	    plt.subplot(122)
	    plt.imshow(mask_debug,cmap='gray')
	    plt.show
	    plt.pause(1)
	#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
	# Scaling
	if num_scales != 0:
	  
	  if color:
	    frame_scaled_set = np.zeros((num_scales,3,TARGET_Y_DIM,TARGET_X_DIM),dtype=np.float32)
	  else:
	    frame_scaled_set = np.zeros((num_scales,1,TARGET_Y_DIM,TARGET_X_DIM),dtype=np.float32)
	    
	  mask_scaled_set = np.zeros((num_scales,1,TARGET_MASK_Y_DIM,TARGET_MASK_X_DIM),dtype=np.float32)
	 
	  original_height_frame = frame.shape[0]
	  original_width_frame = frame.shape[1]
	  original_height_mask, original_width_mask = mask.shape
  
	  
	  for i in range(0,num_scales):
	    #------------------------------------------------------------------------------------
	    # Frames
	    # first scale
	    # random number between 1 and 10 for example
	    random_number = random.uniform(1,5)
	    frame_scaled = rescale(frame,random_number)
	   
	    scaled_height_frame = frame_scaled.shape[0]
	    scaled_width_frame = frame_scaled.shape[1]
	    
	    ## second crop
	    start_row_frame = scaled_height_frame/2 - original_height_frame/2 
	    end_row_frame = scaled_height_frame/2 + original_height_frame/2
	    start_col_frame = scaled_width_frame/2 - original_width_frame/2
	    end_col_frame = scaled_width_frame/2 + original_width_frame/2
	    
	    frame_cropped = frame_scaled[start_row_frame:end_row_frame,start_col_frame:end_col_frame]
	    #------------------------------------------------------------------------------------
	    # Masks
	    # first scale
	    mask_scaled = rescale(mask,random_number)
	    scaled_height_mask, scaled_width_mask = mask_scaled.shape
	    
	    ## second crop
	    start_row_mask = scaled_height_mask/2 - original_height_mask/2 
	    end_row_mask = scaled_height_mask/2 + original_height_mask/2
	    start_col_mask = scaled_width_mask/2 - original_width_mask/2
	    end_col_mask = scaled_width_mask/2 + original_width_mask/2
	    
	    mask_cropped = mask_scaled[start_row_mask:end_row_mask,start_col_mask:end_col_mask]
	    
	     # downsampling the frame and the mask
	    frame_cropped = cv2.resize(frame_cropped,(TARGET_X_DIM,TARGET_Y_DIM))
	    mask_cropped = cv2.resize(mask_cropped,(TARGET_X_DIM,TARGET_Y_DIM))
	    
	    # cast to float32 for Caffe
	    frame_cropped = frame_cropped.astype(dtype=np.float32) 
	    mask_cropped = mask_cropped.astype(dtype=np.float32)
	    
	    
	    # adding cropped mask and frame to the sets
	    if color:
	      frame_scaled_set[i,0,:,:] = frame_cropped[:,:,0]
	      frame_scaled_set[i,1,:,:] = frame_cropped[:,:,1]
	      frame_scaled_set[i,2,:,:] = frame_cropped[:,:,2]
	    
	    else: 
	      frame_scaled_set[i,0,:,:] = frame_cropped 
	    mask_scaled_set[i,0,:,:] = mask_cropped
	   
	# debugging after scaling is finished
	if False:
	  print "-----------------------------------------"
	  print "SCALE augmentation"
	  print "-Frame set shape", frame_scaled_set.shape
	  print "-Mask set shape", mask_scaled_set.shape
	  for i in range(0,num_scales):
	    plt.figure
	    if color:
	      frame_debug = np.zeros((frame_scaled_set.shape[2],frame_scaled_set.shape[3],frame_scaled_set.shape[1]))
	      frame_debug[:,:,0] = frame_scaled_set[i,0,:,:]
	      frame_debug[:,:,1] = frame_scaled_set[i,1,:,:]
	      frame_debug[:,:,2] = frame_scaled_set[i,2,:,:]
	      
	      plt.subplot(121)
	      plt.imshow(frame_debug)
	      
	    else:
	      frame_debug = frame_scaled_set[i,0,:,:]
	      plt.subplot(121)
	      plt.imshow(frame_debug,cmap='gray')
	      
	    mask_debug = np.copy(mask_scaled_set[i,0,:,:])
	      
	    plt.subplot(122)
	    plt.imshow(mask_debug,cmap='gray')
	    plt.show
	    plt.pause(1)	    
	#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
	# Rotation
	if num_rotations != 0:
	  
	  if color:
	    frame_rotated_set = np.zeros((num_rotations,3,TARGET_Y_DIM,TARGET_X_DIM),dtype=np.float32)
	  else:
	    frame_rotated_set = np.zeros((num_rotations,1,TARGET_Y_DIM,TARGET_X_DIM),dtype=np.float32)  
	  mask_rotated_set = np.zeros((num_rotations,1,TARGET_MASK_Y_DIM,TARGET_MASK_X_DIM),dtype=np.float32)
	  
	  
	  for i in range(0,num_rotations):	   
	    random_number = random.uniform(1,360)
	    frame_rot = rotate(frame,random_number)
	    
	    mask_rot = mask
	    mask_rot = rotate(mask_rot,random_number)
	    
	    # downsampling the frame and the mask
	    frame_rot = cv2.resize(frame_rot,(TARGET_X_DIM,TARGET_Y_DIM))
	    mask_rot = cv2.resize(mask_rot,(TARGET_X_DIM,TARGET_Y_DIM))
	    
	    # cast to float32 for Caffe
	    frame_rot = frame_rot.astype(dtype=np.float32) 
	    mask_rot = mask_rot.astype(dtype=np.float32)
	    
	    # adding rotated mask and frame to the sets
	    if color:
	      frame_rotated_set[i,0,:,:] = frame_rot[:,:,0]
	      frame_rotated_set[i,1,:,:] = frame_rot[:,:,1]
	      frame_rotated_set[i,2,:,:] = frame_rot[:,:,2]
	    
	    else: 
	      frame_rotated_set[i,0,:,:] = frame_rot
	    mask_rotated_set[i,0,:,:] = mask_rot
	    
	# debugging after rotation is finished
	if False:
	  print "-----------------------------------------"
	  print "Rotation augmentation"
	  print "-Frame set shape", frame_rotated_set.shape
	  print "-Mask set shape", mask_rotated_set.shape
	  for i in range(0,num_rotations):
	    plt.figure
	    if color:
	      frame_debug = np.zeros((frame_rotated_set.shape[2],frame_rotated_set.shape[3],frame_rotated_set.shape[1]))
	      frame_debug[:,:,0] = frame_rotated_set[i,0,:,:]
	      frame_debug[:,:,1] = frame_rotated_set[i,1,:,:]
	      frame_debug[:,:,2] = frame_rotated_set[i,2,:,:]
	      
	      plt.subplot(121)
	      plt.imshow(frame_debug)
	      
	    else:
	      frame_debug = frame_rotated_set[i,0,:,:]
	      plt.subplot(121)
	      plt.imshow(frame_debug,cmap='gray')
	      
	    mask_debug = np.copy(mask_rotated_set[i,0,:,:])
	      
	    plt.subplot(122)
	    plt.imshow(mask_debug,cmap='gray')
	    plt.show
	    plt.pause(1)
	       
	#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	# plotting mask before downsampling (debugging)
	if False:
	  plt.figure
	  #plt.pcolor( net.blobs['data'].data[0,0,:,:])
	  plt.imshow(mask,cmap='gray')
	  plt.show
	  plt.pause(1)
	
	# debugging
	# shapes of the frame and the mask before downsampling
	if verbose:
	  print "-----------------------------------------------------------------------------------------------"
	  print "\n -Save data from file ",frame_name,"\n into hdf5 file for Caffe input"
	  print "\n -Save label from file ",annotation_video_name,"\n into hdf5 file for Caffe input"
	  print "\n-Frame shape before: ",frame.shape
	  print "-Mask shape before: ",mask.shape
	
	# downsampling the frame and the mask to the new dimension
	frame = cv2.resize(frame,(TARGET_X_DIM,TARGET_Y_DIM))
	mask = cv2.resize(mask,(TARGET_MASK_X_DIM,TARGET_MASK_Y_DIM))
      
	# shapes of the frame and the mask after downsampling
	if verbose:
	  print "\n-Frame shape after: ", frame.shape
	  print "-Mask shape after: ", mask.shape
	
	# cast to float32 for Caffe
	mask = mask.astype(dtype=np.float32)
	frame = frame.astype(dtype=np.float32) 
	
	# don't understand (why 0 ?) # (not implemented for the augmentation options)
	#if extract_shape:
	  #mask[np.where(frame>(extract_th-mean_value) )] = 0  # change max value to 1 !
	 
	# plotting mask after downsampling (debugging)
	if False:
	  plt.figure
	  plt.imshow(mask,cmap='gray')
	  plt.show
	  plt.pause(2)
	
	
	if False:
	  plt.figure
	  plt.subplot(121)
	  if color:
	    plt.imshow(frame)
	  else:
	    plt.imshow(frame,cmap='gray')
	  plt.subplot(122)
	  plt.imshow(mask,cmap='gray')
	  plt.show()
	  plt.pause(1)
	    
	
	# put the data and the mask in the sets
	if color:
	  data_set[fid,0,:,:] = frame[:,:,0]
	  data_set[fid,1,:,:] = frame[:,:,1]
	  data_set[fid,2,:,:] = frame[:,:,2]
	  label_set[fid,0,:,:] = mask
	  fid +=1
	  
	  #check rotation data
	  if num_rotations!=0:
	    data_set[fid:fid+num_rotations,0,:,:] = frame_rotated_set[:,0,:,:]
	    data_set[fid:fid+num_rotations,1,:,:] = frame_rotated_set[:.1,:,:]
	    data_set[fid:fid+num_rotations,2,:,:] = frame_rotated_set[:,2,:,:]
	    
	    label_set[fid:fid+num_rotations,0,:,:] = mask_rotated_set[:,0,:,:]
	    fid += num_rotations
	    
	  #check scaled data
	  if num_scales!=0:
	    data_set[fid:fid+num_scales,0,:,:] = frame_scaled_set[:,0,:,:]
	    data_set[fid:fid+num_scales,1,:,:] = frame_scaled_set[:.1,:,:]
	    data_set[fid:fid+num_scales,2,:,:] = frame_scaled_set[:,2,:,:]
	    
	    label_set[fid:fid+num_scales,0,:,:] = mask_scaled_set[:,0,:,:]
	    fid += num_scales
	    
	  #check contrast data
	  if num_colors!=0:
	    data_set[fid:fid+num_colors,0,:,:] = frame_contrasted_set[:,0,:,:]
	    data_set[fid:fid+num_colors,1,:,:] = frame_contrasted_set[:.1,:,:]
	    data_set[fid:fid+num_colors,2,:,:] = frame_contrasted_set[:,2,:,:]
	    
	    label_set[fid:fid+num_colors,0,:,:] = mask_contrasted_set[:,0,:,:]
	    fid += num_colors
	  
	else:
	  data_set[fid,0,:,:] = frame
	  label_set[fid,0,:,:] = mask
	  fid += 1
	  
	  #check rotation data
	  if num_rotations !=0:
	    data_set[fid:fid+num_rotations,0,:,:] = frame_rotated_set[:,0,:,:]
	    label_set[fid:fid+num_rotations,0,:,:] = mask_rotated_set[:,0,:,:]
	    fid += num_rotations
	    
	  #check scaled data
	  if num_scales !=0:
	    data_set[fid:fid+num_scales,0,:,:] = frame_scaled_set[:,0,:,:]
	    label_set[fid:fid+num_scales,0,:,:] = mask_scaled_set[:,0,:,:]
	    fid += num_scales
	    
	  #check contrast data
	  if num_colors !=0:
	    data_set[fid:fid+num_colors,0,:,:] = frame_contrasted_set[:,0,:,:]
	    label_set[fid:fid+num_colors,0,:,:] = mask_contrasted_set[:,0,:,:]
	    fid += num_colors
	  
	
	  
	  
	
	
	
	
	
      elif annotation_frame == 0:
	print "-----------------------------------------------------------------------------------------------"
	print "-Warning: annotations for frame {} in video {} \n  don't exist... ignoring this data".format(frame_number+1,video_name)
	print "-----------------------------------------------------------------------------------------------"
      elif not os.path.exists(frame_path):
	print "-----------------------------------------------------------------------------------------------"
	print "-Warning: frame {} for video {} doesn't exist\n  but annotations for it exist... ignoring this data".format(frame_number+1,video_name)
	print "-----------------------------------------------------------------------------------------------"
  print "\writing HDF file...",
  f.create_dataset("data", data=data_set)
  f.create_dataset("label", data=label_set)
  f.close()
  print "| done"


