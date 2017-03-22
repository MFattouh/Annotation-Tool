import h5py  # HDF5 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys 
import os
from skimage import io
from skimage import color
import glob

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


def generate_hd5f(TARGET_X_DIM, TARGET_Y_DIM, num_all_frames, annotation_folder_path, frames_folder_path, output_folder_path, color=False):
  
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
  print "---------------------- Begin HDF5 export -------------------------"
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
	  #mask = cv2.imread(mname)
	  # mean substraction
	  frame[:,:,0] = frame[:,:,0] - mean_value[0]
	  frame[:,:,1] = frame[:,:,1] - mean_value[1]
	  frame[:,:,2] = frame[:,:,2] - mean_value[2]
	  
	else:
	  frame = cv2.imread(frame_path,0)
	  #mask = cv2.imread(mname,0)
	  # mean substraction
	  frame[:,:] = frame[:,:] - mean_value[0]
	 
	mask = np.zeros_like(frame)

	# creating the mask array for the current frame
	# for each label in the current frame
	for label in range(0,len(annotation_frame)):
	   mask[annotation_frame[label][1]:annotation_frame[label][3], annotation_frame[label][0]:annotation_frame[label][2]] = annotation_frame[label][-1]
	
	# plotting mask before downsampling (debugging)
	if False:
	  plt.figure
	  #plt.pcolor( net.blobs['data'].data[0,0,:,:])
	  plt.imshow(mask,cmap='gray')
	  plt.show
	  plt.pause(1)
	
	
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
	mask_cp = mask.astype(dtype=np.float32)
	frame_cp = frame.astype(dtype=np.float32) 
      
	# don't understand (why 0 ?)
	if extract_shape:
	  mask_cp[np.where(frame_cp>(extract_th-mean_value) )] = 0  # change max value to 1 !
	 
	# plotting mask after downsampling (debugging)
	if False:
	  plt.figure
	  #plt.pcolor( net.blobs['data'].data[0,0,:,:])
	  plt.imshow(mask_cp,cmap='gray')
	  plt.show
	  plt.pause(2)

	if do_plot:
	  # if frames and masks have different shapes
	  if frame.shape != mask.shape:
	    mask_cp2 = mask_cp
	    mask_cp2 = cv2.resize(mask_cp2,(TARGET_X_DIM,TARGET_Y_DIM))
	    vis = np.concatenate((mask_cp2,frame),axis=1)
	  else:
	    # if frames and masks have same shapes
	    vis = np.concatenate((mask_cp,frame),axis=1)
	  plt.figure
	  #plt.pcolor( net.blobs['data'].data[0,0,:,:])
	  plt.imshow(mask_cp,cmap='gray')  
	  plt.show
	  plt.pause(2)
	  
	
	# put the data of the current mask and frame in the label structure
	if color:
	  label_set[fid,0,:,:] = mask_cp[:,:,0]
	  label_set[fid,1,:,:] = mask_cp[:,:,1]
	  label_set[fid,2,:,:] = mask_cp[:,:,2]
	  
	  data_set[fid,0,:,:] = frame_cp[:,:,0]
	  data_set[fid,1,:,:] = frame_cp[:,:,1]
	  data_set[fid,2,:,:] = frame_cp[:,:,2]
	  
	else:
	  label_set[fid,0,:,:] = mask_cp
	  data_set[fid,0,:,:] = frame_cp
	
	fid +=1
	
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

  
#if __name__ == "__main__":
  #generate_hd5f()


