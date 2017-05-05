import os
from scipy.misc import imsave
import scipy.io
import numpy as np
import h5py  # HDF5

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
    hdf5_folder = os.path.join(mask_folder, "hdf5")
    if not os.path.exists(hdf5_folder):
      os.makedirs(hdf5_folder)
 # ---------------------------1-------------------------------------------------------

  # data, label sets shape (examples * channels * x * y)
  ''' it is important how the data is stored in data_set and label_set (e.g original frame store first then contrasted frame
      then scaled frame then rotation frame '''


  # for each video
  example_number = 0
  for index, (video_name,video_frames) in enumerate(zip(video_names_list, annotated_frames_list)):

    print "Writing {} files for video: {} ...".format(type_data,video_name),

    # initialize some arrays for the hdf5
    if type_data == "hdf5":
      example_this_video = sum(video_frames)
      row = data_set.shape[2]
      col = data_set.shape[3]

      if color:
        hdf5_data_set = np.zeros((example_this_video,3,row,col),dtype=np.float32)
        hdf5_label_set = np.zeros((example_this_video,1,row,col),dtype=np.float32)
        index_original_data = 0

        if num_colors != 0:
          hdf5_data_color_set = np.zeros((num_colors*example_this_video,3,row,col),dtype=np.float32)
          hdf5_label_color_set = np.zeros((num_colors*example_this_video,1,row,col),dtype=np.float32)
          index_color_data = 0

        if num_scales !=0:
          hdf5_data_scale_set = np.zeros((num_scales*example_this_video,3,row,col),dtype=np.float32)
          hdf5_label_scale_set = np.zeros((num_scales*example_this_video,1,row,col),dtype=np.float32)
          index_scale_data = 0

        if num_rotations != 0:
          hdf5_data_rot_set = np.zeros((num_rotations*example_this_video,3,row,col),dtype=np.float32)
          hdf5_label_rot_set = np.zeros((num_rotations*example_this_video,1,row,col),dtype=np.float32)
          index_rot_data = 0

      else:
        hdf5_data_set = np.zeros((example_this_video,1,row,col),dtype=np.float32)
        hdf5_label_set = np.zeros((example_this_video,1,row,col),dtype=np.float32)
        index_original_data = 0

        if num_colors != 0:
          hdf5_data_color_set = np.zeros((num_colors*example_this_video,1,row,col),dtype=np.float32)
          hdf5_label_color_set = np.zeros((num_colors*example_this_video,1,row,col),dtype=np.float32)
          index_color_data = 0

        if num_scales !=0:
          hdf5_data_scale_set = np.zeros((num_scales*example_this_video,1,row,col),dtype=np.float32)
          hdf5_label_scale_set = np.zeros((num_scales*example_this_video,1,row,col),dtype=np.float32)
          index_scale_data = 0

        if num_rotations != 0:
          hdf5_data_rot_set = np.zeros((num_rotations*example_this_video,1,row,col),dtype=np.float32)
          hdf5_label_rot_set = np.zeros((num_rotations*example_this_video,1,row,col),dtype=np.float32)
          index_rot_data = 0

    # for each frame in the video
    for frame_index, frame in enumerate(video_frames):
      # save only if the frame is annotated
      if frame == 1:
        #----------------------------------------------------------------------------------------------------------------------------------------
        # save as images
        if type_data == "image":
          # save original data (frame)
          if color:
            imsave(image_folder_data+"/"+video_name + "_{0:05d}.png".format(frame_index + 1),data_set[example_number,0:3,:,:])
          else:
            imsave(image_folder_data+"/"+video_name+ "_{0:05d}.png".format(frame_index + 1),data_set[example_number,0,:,:])
          # save original label (mask)
          imsave(image_folder_labels+"/"+video_name+ "_{0:05d}.png".format(frame_index + 1),label_set[example_number,0,:,:])
          example_number += 1

          # save the contrasted data and labels if exist
          if num_colors!=0:
            for i in range(0,num_colors):
              if color:
                imsave(image_folder_data+"/"+video_name + "_{0:05d}".format(frame_index + 1)+"_augment_color_"+str(i+1)+".png",data_set[example_number,0:3,:,:])
              else:
                imsave(image_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_color_"+str(i+1)+".png",data_set[example_number,0,:,:])
              # save colored label (mask)
              imsave(image_folder_labels+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_color_"+str(i+1)+".png",label_set[example_number,0,:,:])
              example_number +=1

          # save the scaled data and labels if exist
          if num_scales!=0:
            for i in range(0,num_scales):
              if color:
                imsave(image_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_scale_"+str(i+1)+".png",data_set[example_number,0:3,:,:])
              else:
                imsave(image_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_scale_"+str(i+1)+".png",data_set[example_number,0,:,:])
              # save scaled label (mask)
              imsave(image_folder_labels+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_scale_"+str(i+1)+".png",label_set[example_number,0,:,:])
              example_number +=1

          # save the rotated data and labels if exist
          if num_rotations!=0:
            for i in range(0,num_rotations):
              if color:
                imsave(image_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_rot_"+str(i+1)+".png",data_set[example_number,0:3,:,:])
              else:
                imsave(image_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_rot_"+str(i+1)+".png",data_set[example_number,0,:,:])
              # save rotated label (mask)
              imsave(image_folder_labels+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_rot_"+str(i+1)+".png",label_set[example_number,0,:,:])
              example_number +=1
        #----------------------------------------------------------------------------------------------------------------------------------------

        # save as mat files
        if type_data == "mat":
          # save original data (frame)
          if color:
            MM = {'PartMask': data_set[example_number,0:3,:,:]}
            scipy.io.savemat(mat_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+".mat", mdict = {'MM': MM}, do_compression = True)
          else:
            MM = {'PartMask': data_set[example_number,0,:,:]}
            scipy.io.savemat(mat_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+".mat", mdict = {'MM': MM}, do_compression = True)
          # save original label (mask)
          MM = {'PartMask': label_set[example_number,0,:,:]}
          scipy.io.savemat(mat_folder_labels+"/"+video_name+"_{0:05d}".format(frame_index + 1)+".mat", mdict = {'MM': MM}, do_compression = True)
          example_number+=1

          # save the contrasted data and labels if exist
          if num_colors!=0:
            for i in range(0,num_colors):
              if color:
                MM = {'PartMask': data_set[example_number,0:3,:,:]}
                scipy.io.savemat(mat_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_color_"+str(i+1)+".mat", mdict = {'MM': MM}, do_compression = True)
              else:
                MM = {'PartMask': data_set[example_number,0,:,:]}
                scipy.io.savemat(mat_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_color_"+str(i+1)+".mat", mdict = {'MM': MM}, do_compression = True)
              # save colored label (mask)
              MM = {'PartMask': label_set[example_number,0,:,:]}
              scipy.io.savemat(mat_folder_labels+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_color_"+str(i+1)+".mat", mdict = {'MM': MM}, do_compression = True)
              example_number +=1

          # save the scaled data and labels if exist
          if num_scales!=0:
            for i in range(0,num_scales):
              if color:
                MM = {'PartMask': data_set[example_number,0:3,:,:]}
                scipy.io.savemat(mat_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_scale_"+str(i+1)+".mat", mdict = {'MM': MM}, do_compression = True)
              else:
                MM = {'PartMask': data_set[example_number,0,:,:]}
                scipy.io.savemat(mat_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_scale_"+str(i+1)+".mat", mdict = {'MM': MM}, do_compression = True)
              # save scaled label (mask)
              MM = {'PartMask': label_set[example_number,0,:,:]}
              scipy.io.savemat(mat_folder_labels+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_scale_"+str(i+1)+".mat", mdict = {'MM': MM}, do_compression = True)
              example_number +=1

          # save the rotated data and labels if exist
          if num_rotations!=0:
            for i in range(0,num_rotations):
              if color:
                MM = {'PartMask': data_set[example_number,0:3,:,:]}
                scipy.io.savemat(mat_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_rot_"+str(i+1)+".mat", mdict = {'MM': MM}, do_compression = True)
              else:
                MM = {'PartMask': data_set[example_number,0,:,:]}
                scipy.io.savemat(mat_folder_data+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_rot_"+str(i+1)+".mat", mdict = {'MM': MM}, do_compression = True)
              # save scaled label (mask)
              MM = {'PartMask': label_set[example_number,0,:,:]}
              scipy.io.savemat(mat_folder_labels+"/"+video_name+"_{0:05d}".format(frame_index + 1)+"_augment_rot_"+str(i+1)+".mat", mdict = {'MM': MM}, do_compression = True)
              example_number +=1
        #----------------------------------------------------------------------------------------------------------------------------------------
        # save as hdf5 files
        if type_data == "hdf5":

          # save original data
          if color:
            hdf5_data_set[index_original_data,0:3,:,:] = data_set[example_number,0:3,:,:]
          else:
            hdf5_data_set[index_original_data,0,:,:] = data_set[example_number,0,:,:]

          # save original label
          hdf5_label_set[index_original_data,0,:,:] = label_set[example_number,0,:,:]
          index_original_data +=1
          example_number +=1

          # save color data if exists
          if num_colors!= 0 :
            if color:
              hdf5_data_color_set[index_color_data:index_color_data+num_colors,0:3,:,:] = data_set[example_number:example_number+num_colors,0:3,:,:]
            else:
              hdf5_data_color_set[index_color_data:index_color_data+num_colors,0,:,:] = data_set[example_number:example_number+num_colors,0,:,:]

            # save color label
            hdf5_label_color_set[index_color_data:index_color_data+num_colors,0,:,:] = label_set[example_number:example_number+num_colors,0,:,:]
            index_color_data = index_color_data + num_colors
            example_number = example_number + num_colors

          # save scale data if exists
          if num_scales!= 0 :
            if color:
              hdf5_data_scale_set[index_scale_data:index_scale_data+num_scales,0:3,:,:] = data_set[example_number:example_number+num_scales,0:3,:,:]
            else:
              hdf5_data_scale_set[index_scale_data:index_scale_data+num_scales,0,:,:] = data_set[example_number:example_number+num_scales,0,:,:]

            # save scale label
            hdf5_label_scale_set[index_scale_data:index_scale_data+num_scales,0,:,:] = label_set[example_number:example_number+num_scales,0,:,:]
            index_scale_data = index_scale_data + num_scales
            example_number = example_number + num_scales



          # save rotation data if exists
          if num_rotations!= 0 :
            if color:
              hdf5_data_rot_set[index_rot_data:index_rot_data+num_rotations,0:3,:,:] = data_set[example_number:example_number+num_rotations,0:3,:,:]
            else:
              hdf5_data_rot_set[index_rot_data:index_rot_data+num_rotations,0,:,:] = data_set[example_number:example_number+num_rotations,0,:,:]

            # save rotation label
            hdf5_label_rot_set[index_rot_data:index_rot_data+num_rotations,0,:,:] = label_set[example_number:example_number+num_rotations,0,:,:]
            index_rot_data = index_rot_data + num_rotations
            example_number = example_number + num_rotations

    # writing hdf5 files
    if type_data == "hdf5":
      #original data
      #print "\nOriginal"
      #print hdf5_data_set.shape
      #print hdf5_label_set.shape

      f_name = hdf5_folder+"/"+video_name+".hdf5"
      f = h5py.File(f_name,"w")

      f.create_dataset("data", data= hdf5_data_set)
      f.create_dataset("label", data= hdf5_label_set)
      f.close()

      if num_colors!=0:
        # color data
        #print "\ncolor"
        #print hdf5_data_color_set.shape
        #print hdf5_label_color_set.shape
        f_name = hdf5_folder+"/"+video_name+"_color"+".hdf5"

        f = h5py.File(f_name,"w")
        f.create_dataset("data", data= hdf5_data_color_set)
        f.create_dataset("label", data= hdf5_label_color_set)
        f.close()

      if num_scales!=0:
        #print "\nscale"
        #print hdf5_data_scale_set.shape
        #print hdf5_label_scale_set.shape
        # scale data
        f_name = hdf5_folder+"/"+video_name+"_scale"+".hdf5"

        f = h5py.File(f_name,"w")
        f.create_dataset("data", data= hdf5_data_scale_set)
        f.create_dataset("label", data= hdf5_label_scale_set)
        f.close()

      if num_rotations!=0:
        #print "\n Rot"
        #print hdf5_data_rot_set.shape
        #print hdf5_label_rot_set.shape
        # rotated data
        f_name = hdf5_folder+"/"+video_name+"_rot"+".hdf5"

        f = h5py.File(f_name,"w")
        f.create_dataset("data", data= hdf5_data_rot_set)
        f.create_dataset("label", data= hdf5_label_rot_set)
        f.close()

    print "|done"
