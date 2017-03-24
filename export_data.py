import os
from scipy.misc import imsave

def export(output_folder, data_set, label_set, num_colors, num_scales, num_rotations, video_names_list, annotated_frames_list, type_data, color):
  
  # ----------------------------------------------------------------------------------
  # creating mask folders if they doesn't exist
  mask_folder = os.path.join(output_folder,"masks")
  if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)
  
  if type_data == "image":
    image_folder_data = os.path.join(mask_folder,"images/data")
    image_folder_labels = os.path.join(mask_folder,"images/labels")
    if not os.path.exists(image_folder_data):
      os.makedirs(image_folder_data)
    if not os.path.exists(image_folder_labels):
      os.makedirs(image_folder_labels)
      
  if type_data == "mat":
    mat_folder_data = os.path.join(mask_folder, "mat/data")
    mat_folder_labels = os.path.join(mask_folder, "mat/labels")
    
    if not os.path.exists(mat_folder_data):
      os.makedirs(mat_folder_data)
    if not os.path.exists(mat_folder_labels):
      os.makedirs(mat_folder_labels)
    
  if type_data == "hdf5":
    hdf5_folder = os.path.join(mask_folder, "hd5f")
    if not os.path.exists(hdf5_folder):
      os.makedirs(hdf5_folder)
 # ---------------------------1-------------------------------------------------------
 
  # data, label sets shape (examples * channels * x * y)
  ''' it is important how the data is stored in data_set and label_set (e.g original frame store first then contrasted frame
      then scaled frame then rotation frame '''
  
  # for each video 
  example_number = 0
  for index, (video_name,video_frames) in enumerate(zip(video_names_list, annotated_frames_list)):
     print video_name
     print len(video_frames)
     print video_frames
     for frame_index, frame in enumerate(video_frames):
       
       # save only if the frame is annotated
       if frame == 1:
	 
	 
	 #----------------------------------------------------------------------------------------------------------------------------------------
	 # save as images
	 if type_data == "image":
	   # save original data (frame)
	   if color:
	     imsave(image_folder_data+"/"+video_name+"_"+str(frame_index+1)+".png",data_set[example_number,0:3,:,:])
	   else:
	     imsave(image_folder_data+"/"+video_name+"_"+str(frame_index+1)+".png",data_set[example_number,0,:,:])
	   # save original label (mask)
	   imsave(image_folder_labels+"/"+video_name+"_"+str(frame_index+1)+".png",label_set[example_number,0,:,:])
	   example_number += 1
	   
	   # save the contrasted data and labels if exist
	   if num_colors!=0:
	     for i in range(0,num_colors):
	       if color:
		 imsave(image_folder_data+"/"+video_name+"_"+str(frame_index+1)+"_augment_color_"+str(i+1)+".png",data_set[example_number,0:3,:,:])
	       else:
		 imsave(image_folder_data+"/"+video_name+"_"+str(frame_index+1)+"_augment_color_"+str(i+1)+".png",data_set[example_number,0,:,:])
	       # save original label (mask)
	       imsave(image_folder_labels+"/"+video_name+"_"+str(frame_index+1)+"_augment_color_"+str(i+1)+".png",label_set[example_number,0,:,:])
	       example_number +=1
	       
	    # save the scaled data and labels if exist
	   if num_scales!=0:
	     for i in range(0,num_scales):
	       if color:
		 imsave(image_folder_data+"/"+video_name+"_"+str(frame_index+1)+"_augment_scale_"+str(i+1)+".png",data_set[example_number,0:3,:,:])
	       else:
		 imsave(image_folder_data+"/"+video_name+"_"+str(frame_index+1)+"_augment_scale_"+str(i+1)+".png",data_set[example_number,0,:,:])
	       # save original label (mask)
	       imsave(image_folder_labels+"/"+video_name+"_"+str(frame_index+1)+"_augment_scale_"+str(i+1)+".png",label_set[example_number,0,:,:])
	       example_number +=1
		
	    # save the rotated data and labels if exist
	   if num_rotations!=0:
	     for i in range(0,num_rotations):
	       if color:
		 imsave(image_folder_data+"/"+video_name+"_"+str(frame_index+1)+"_augment_rot_"+str(i+1)+".png",data_set[example_number,0:3,:,:])
	       else:
		 imsave(image_folder_data+"/"+video_name+"_"+str(frame_index+1)+"_augment_rot_"+str(i+1)+".png",data_set[example_number,0,:,:])
	       # save original label (mask)
	       imsave(image_folder_labels+"/"+video_name+"_"+str(frame_index+1)+"_augment_rot_"+str(i+1)+".png",label_set[example_number,0,:,:])
	       example_number +=1
	
	   
  