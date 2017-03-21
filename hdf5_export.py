import h5py  # HDF5 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys 
import os
from skimage import io
from skimage import color
import glob

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


def generate_hd5f(color, TARGET_X_DIM, TARGET_Y_DIM, num_all_frames, masks_folder_path, frames_folder_path, output_folder_path):
  
  # calculate the mean over all pixels in all frames
  mean_value = calculate_mean_all_frames(color, num_all_frames, frames_folder_path)
  
  if extract_shape:
    if extract_th - mean_value <= 1:
      print "STOP:\n mean value too close to threshold extraction value"
      quit() 
  
  # mask shape = frame shape
  TARGET_MASK_X_DIM = TARGET_X_DIM
  TARGET_MASK_Y_DIM = TARGET_Y_DIM
  
  # get list of masks (png files in the mask folder)
  masks_list_full_path = glob.glob(os.path.join(masks_folder_path, "*.png"))
  
  masks_list =[]
  for mask_path in masks_list_full_path:
    masks_list.append(os.path.basename(mask_path))
  
  # number of masks in the mask folder
  max_examples = len(masks_list)

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


  # loop for each frame(data) and corresponding mask(label)
  # downsample + putting data and label in the corresponding structures
  for fid, mask_name in enumerate(masks_list): 
    
    # add frames and masks to hdf5 if only the frame has a corresponding mask (same name)
    # otherwise if a frame has no mask then ignore this frame
    mname = os.path.join(masks_folder_path,mask_name)
    fname = os.path.join(frames_folder_path, mask_name)
    
    # read the frame and the mask to arrays
    if color:
      frame = cv2.imread(fname)
      mask = cv2.imread(mname)
      # mean substraction
      frame[:,:,0] = frame[:,:,0] - mean_value[0]
      frame[:,:,1] = frame[:,:,1] - mean_value[1]
      frame[:,:,2] = frame[:,:,2] - mean_value[2]
      
    else:
      frame = cv2.imread(fname,0)
      mask = cv2.imread(mname,0)
      # mean substraction
      frame[:,:] = frame[:,:] - mean_value[0]
        
    # plotting mask before downsampling (debugging)
    if False:
      plt.figure
      #plt.pcolor( net.blobs['data'].data[0,0,:,:])
      plt.imshow(mask,cmap='gray')
      plt.show
      plt.pause(5)
    
    # shapes of the frame and the mask before downsampling
    if verbose:
      print "\t-Save data from file ",fname," into hdf5 file for Caffe input"
      print "\t-Save label from file ",mname," into hdf5 file for Caffe input"
      print "frame shape before ",frame.shape
      print "mask shape before ",mask.shape
    
    # downsampling the frame and the mask to the new dimension
    frame = cv2.resize(frame,(TARGET_X_DIM,TARGET_Y_DIM))
    mask = cv2.resize(mask,(TARGET_MASK_X_DIM,TARGET_MASK_Y_DIM))
  
    # shapes of the frame and the mask after downsampling
    if verbose:
      print "frame shape after ", frame.shape
      print "mask shape after ", mask.shape
    
    # cast to float32 for Caffe
    mask_cp = mask.astype(dtype=np.float32)
    frame_cp = frame.astype(dtype=np.float32) 
  
    # don't understand (why 0 ?)
    if extract_shape:
      mask_cp[np.where(frame_cp>(extract_th-mean_value) )] = 0  # change max value to 1 !
    
    # change max value to 1 # what if I have several labels ?
    #mask_cp[np.where(mask_cp>0)] = 1  
    
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
      plt.imshow(vis,cmap='gray')  
      plt.show
      plt.pause(2)
      
    
    # put the data of the current mask in the label structure
    if color:
      label_set[fid,0,:,:] = mask_cp[:,:,0]
      label_set[fid,1,:,:] = mask_cp[:,:,1]
      label_set[fid,2,:,:] = mask_cp[:,:,2]
    else:
      label_set[fid,0,:,:] = mask_cp
    
    # put the data of the current frame in the data structure
    if color:
      data_set[fid,0,:,:] = frame_cp[:,:,0]
      data_set[fid,1,:,:] = frame_cp[:,:,1]
      data_set[fid,2,:,:] = frame_cp[:,:,2]
    else:
      data_set[fid,0,:,:] = frame_cp

  print "writing HDF file...",
  f.create_dataset("data", data=data_set)
  f.create_dataset("label", data=label_set)
  f.close()
  print "| done"

  
#if __name__ == "__main__":
  #generate_hd5f()


