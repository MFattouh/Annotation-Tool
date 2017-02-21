import Tkinter as tk  	# GUI 
import tkFileDialog  # for browsing files
import tkMessageBox # warning boxes
import tkSimpleDialog # pop up window with entry
import os	      	# system calls
from PIL import ImageTk, Image # images in GUI and image processing 
import dlib	      	# correlational tracker
#from skimage import io 	# image read and conversion to array 
import glob		# some linux command functions
import numpy as np	# matlab python stuff
import cPickle		# saving rectangle pairs (list i/o)
import datetime		# date time function 
from sklearn.feature_extraction import image # just another image processing lib
import extract_patches	# for patch extraction?
from extract_frames import extract_frames_from_videos # frame extraction function
from extract_frames import get_video_file_name
import argparse # for the arguments passed 
from compute_masks import create_masks_for_model # for creating the mask 


class SampleApp(tk.Tk):  # inherit from Tk class 
    '''Illustrate how to drag items on a Tkinter canvas'''

    def __init__(self):
        tk.Tk.__init__(self)
	
	# Folder settings
	
	# main folder
	current_dir = os.getcwd() + "/"
        main_folder = current_dir
        if args.main_folder is None:
	  print "WARNING!! main folder path was not passed (default: current directory)"
	else:
	  if args.main_folder[-1] != "/":
	    args.main_folder += "/"
	  main_folder += args.main_folder
	main_folder = os.path.abspath(main_folder) + "/"
	# change directory to main folder
        os.chdir(main_folder)
        
        # output folder
	output_folder = current_dir
	if args.output_folder is None:
	  print "WARNING!! output folder path was not passed (default: current directory)"
	else:
	  if args.output_folder[-1] != "/":
	    args.output_folder += "/"
	  output_folder += args.output_folder
	output_folder = os.path.abspath(output_folder) + "/"
	self.output_folder = output_folder
	
	# videos folder
        videos_folder = main_folder + "videos/"
        # check if the videos folders exists
        if not os.path.isdir(videos_folder):
	  print "Error: Main directory " + main_folder + " doesn't contain a folder named videos " 
	  exit(1)
	
        # frames folder
        frames_folder_path = output_folder + "frames/"
        # check if the frames folder exists
        if not os.path.exists(frames_folder_path):
	  # create a folder for the frames
	  os.makedirs(frames_folder_path)
        self.frames_folder = frames_folder_path
	
        # annnotations folder
        annotation_folder = output_folder + "annotations/"
        #check if already exists
        if not os.path.exists(annotation_folder):
	  os.makedirs(annotation_folder)
        self.annot_save_folder = os.path.abspath(annotation_folder)
        
        # masks folder
        mask_folder =  os.path.join(output_folder,"masks/")
        if not os.path.exists(mask_folder):
	  os.makedirs(mask_folder)
        self.mask_folder = mask_folder
	
	###### GUI #####
	self.title("Data annotation")
        # create a canvas
        self.canvas = tk.Canvas(width=1200, height=640) 
        self.canvas.pack(fill="both", expand=True)	
        
	'''create buttons section'''
	# add quit button
        #button1 = tk.Button(self.canvas, text = "Quit", command = self.quit,
                                                            #anchor = "w")
        #button1.configure(width = 10)
        #button1.pack()
        #button1_window = self.canvas.create_window(10, 10, anchor="nw", window=button1)
	
	# check if inbetween save and load the space is same
	button2 = tk.Button(self.canvas, text = "Load annotations", command = self.load, anchor = "w")
	button2.configure(width = 15)
	button2.pack()
	button2_window = self.canvas.create_window(100, 10, anchor="nw", window=button2)
	
	button3 = tk.Button(self.canvas, text = "Save annotations", command = self.save, anchor = "w")
	button3.configure(width = 15)
	button3.pack()
	button3_window = self.canvas.create_window(100, 50, anchor="nw", window=button3)
	
	
	button4 = tk.Button(self.canvas, text = "Extract rnd patches", command = self.extract_patches, anchor = "w")
	button4.configure(width = 15)
	button4.pack()
	button4_window = self.canvas.create_window(300, 10, anchor="nw", window=button4)

        button5 = tk.Button(self.canvas, text = "Segment patches for current frame", command = self.extract_patches_tf, anchor = "w")
        button5.configure(width = 27)
        button5.pack()
        button5_window = self.canvas.create_window(500, 10, anchor="nw", window=button5)

	
	
	#button5 = tk.Button(self.canvas, text = "Extract all patches", command = extract_patches.generate_patches_for_models((200,200)), anchor = "w")
	#button5.configure(width = 15)
	#button5.pack()
	#button5_window = self.canvas.create_window(530, 50, anchor="nw", window=button5)
	
	# Video label window
	self.video_info_label = tk.LabelFrame(self.canvas, text = "Video information", padx=5, pady=5)
	self.video_info_label.pack()
	self.canvas.create_window(800,140, anchor = "nw", window = self.video_info_label)
	tk.Label(self.video_info_label, text = "Video: No info").pack()
	tk.Label(self.video_info_label, text = "Video: No info").pack()
	
	# Frame label window
	self.frame_info_label = tk.LabelFrame(self.canvas, text="Frame information", padx=5, pady=5)
	self.frame_info_label.pack()
	self.canvas.create_window(800, 240, anchor="nw", window=self.frame_info_label)
	tk.Label(self.frame_info_label, text="Frame: No info").pack()
	tk.Label(self.frame_info_label, text="Frame: No info").pack()	
	
	# Annotation label window
	self.frame_annot_label = tk.LabelFrame(self.canvas, text="Annotation information", padx=5, pady=5)
	self.frame_annot_label.pack()
	self.canvas.create_window(800, 340, anchor="nw", window=self.frame_annot_label)
	tk.Label(self.frame_annot_label, text="Annotated frames: No info").pack()
	  
	# Action label window
	self.legend_label = tk.LabelFrame(self.canvas, text="Legend", padx=5, pady=5)
	self.legend_label.pack()
	self.canvas.create_window(800, 440, anchor="nw", window=self.legend_label)  
	tk.Label(self.legend_label, text = "<--  Left \n --> Right \n ReturnKey <--| Annotate \n BackSpace < Delete Annotation ").pack()
	
	# Mask window
	self.mask_label = tk.LabelFrame(self.canvas, text = "Create mask as:", padx = 5, pady =5)
	self.mask_label.pack()
	self.canvas.create_window(800,10, anchor = "nw", window = self.mask_label)
	button7 = tk.Button(self.mask_label, text = "Images", command = lambda: self.create_mask("image"), anchor = "w")
	button7.configure(width= 5)
	button7.pack(side = tk.LEFT)
	button8 = tk.Button(self.mask_label, text = "Mat", command = lambda: self.create_mask("mat"), anchor = "w")
	button8.configure(width= 5)
	button8.pack(side = tk.LEFT)
	
	
	# Rectangle size window
	self.rectangle_label = tk.LabelFrame(self.canvas, text = "Rectangle size:", padx = 5, pady =5)
	self.rectangle_label.pack()
	self.canvas.create_window(1000,10,anchor = "nw", window = self.rectangle_label,width = 100)
	button9 = tk.Button(self.rectangle_label, text = "w:",command = lambda: self.rectangle_change_size("w"))
	button9.configure(width = 25)
	button9.pack()
	button10 = tk.Button(self.rectangle_label, text = "h:",command = lambda: self.rectangle_change_size("h"))
	button10.configure(width = 25)
	button10.pack()


	
	
	
	#print self.frame_info_label.winfo_children()[0].config(text="asdas")
        #label_num_frames_window = self.canvas.create_window(800, 140, anchor="nw", window=self.label_num_frames)
        """
        self.label_frame_path = tk.Label(self.canvas, text="Frame path:{0}".format(os.path.join(self.frames_folder, "{0}.png".format(self.img_num+1))), fg="red", font=('Helvetica',14), highlightbackground = "black", highlightthickness=1)
	self.label_frame_path.pack()
        label_frame_path_window = self.canvas.create_window(800, 240, anchor="nw", window=self.label_frame_path)
	"""
	# ectract frames
        extract_frames_from_videos(videos_folder, args.videos, frames_folder_path, args.fps)
        
        # get list of videos and the number of their frames
        self.get_number_of_videos_and_frames()
        self.video_index = 0
        
        # check if a video file was passed to show its frames first
        video = None
        if args.videos is not None:
	  video = get_video_file_name(args.videos)
	  video = video + "_fps_" + str(args.fps)
        
        # get the index of the video passed as argument from the list
        if video is not None:
	  self.video_index = self.list_of_videos.index(video)
        
        # load the first video frames
        self.img_num = 0
        self.load_frames(self.list_of_videos[self.video_index])
        
        
    def rectangle_change_size(self, dimension,w = None, h = None):
      if dimension == "both":
	self.rectangle_size[0] = w
	self.rectangle_size[1] = h
	self.rectangle_label.winfo_children()[0].config(text = "w: {}".format(w))
	self.rectangle_label.winfo_children()[1].config(text = "h: {}".format(h))
	
      else:
	if dimension == "w":
	  w = tkSimpleDialog.askinteger("Width","Enter a number",parent = self.canvas)
	  if w is None:
	    return
	
	  self.rectangle_size[0] = w
	  self.rectangle_label.winfo_children()[0].config(text = "w: {}".format(w))
      
	elif dimension == "h":
	  h = tkSimpleDialog.askinteger("Height","Enter a number",parent = self.canvas)
	  if h is None:
	    return
	  self.rectangle_size[1] = h
	  self.rectangle_label.winfo_children()[1].config(text = "h: {}".format(h))
      
      
	# reset all annotations for the new size
	self.rectangle_frame_pairs = [0]*self.video_num_of_frames
	# update the annotation label
	self.frame_annot_label.winfo_children()[0].config(text="Annotated frames: {0:0{width}}/{1}".format(0, len(self.rectangle_frame_pairs), width=3))
	
      img_width = self.curr_photoimage.width()
      img_height = self.curr_photoimage.height()
      
      # delete the rectangle
      self.canvas.delete(self.polygon_id)
      # create a new rectangle with the new width
      self.create_token((img_width/2 + self.img_start_x, img_height/2 + self.img_start_y), "blue", self.rectangle_size)
      
      
      
        
    def load_annotations_from_file(self,file_name):
      f = file(file_name, 'rb')
      frame_rectangle_pairs = cPickle.load(f)
      f.close()
      return frame_rectangle_pairs 
    
    def create_mask(self,save_option):   
      # check if there is a frame folder loaded first
      if not self.frames_folder:
	tkMessageBox.showinfo(title = "Warning", message = "Load video frames directory first")
	return
      
      model_annot_name = os.path.join(self.annot_save_folder, self.video_name + ".model")
      if not os.path.exists(model_annot_name):
	tkMessageBox.showinfo(title = "Warning", message = "Annotated model doesn't exist for this video")
      else:
	img_width = self.curr_photoimage.width()
	img_height = self.curr_photoimage.height()
	
	# call create mask for model function
	create_masks_for_model(model_annot_name, self.mask_folder, self.video_name, img_width, img_height,save_option)
	
	if save_option == "image":
	  tkMessageBox.showinfo(title = "Mask created", message = "saved as images")
	elif save_option == "mat":
	  tkMessageBox.showinfo(title = "Mask created", message = "saved as mat files")	

		
    def create_token(self, center_rectangle, color, rectangle_size):
        # Create a token at the given coordinate in the given color'''
        (x,y) = center_rectangle
        ####print "X", x
        ####print "Y", y
        # left upper corner
        l_u_c_x = x - rectangle_size[0]/2
        l_u_c_y = y - rectangle_size[1]/2
         
        # left bottom corner
        l_b_c_x =  x - rectangle_size[0]/2
        l_b_c_y =  y + rectangle_size[1]/2
        
        # right upper corner
        r_u_c_x =  x + rectangle_size[0]/2
        r_u_c_y = y - rectangle_size[1]/2
        
        # right bottom corner
        r_b_c_x = x + rectangle_size[0]/2
        r_b_c_y = y + rectangle_size[1]/2
	self.polygon_id = self.canvas.create_polygon(l_u_c_x, l_u_c_y, 
						     r_u_c_x, r_u_c_y, 
						     r_b_c_x, r_b_c_y, 
						     l_b_c_x, l_b_c_y, 
						      outline=color, fill='', tags="token")
    	
	
    def get_number_of_videos_and_frames(self):
      # list of different video names
      self.list_of_videos = []
      
      # list of number of frames for each video of the above list
      self.list_number_of_frames = []
      video_frame_number = -1
      
      for root, dirs, frames in os.walk(self.frames_folder):
	for frame in frames:
	  video_name = frame[0:frame.rfind('_')]
	  if video_name not in self.list_of_videos:
	    self.list_of_videos.append(video_name)
	    self.list_number_of_frames.append(0)
	    video_frame_number += 1
	    
	    for other_frame in frames:
	      if video_name == other_frame[0:other_frame.rfind('_')]:
		self.list_number_of_frames[video_frame_number] += 1
		
      self.total_num_of_videos = len(self.list_of_videos)
      self.total_num_of_frames = sum(self.list_number_of_frames)
      print "Number of videos: ", self.total_num_of_videos
      print "Number of frames for each video", self.list_number_of_frames
      print "Total Number of frames", self.total_num_of_frames
      # Next couple of lines to get the name of the video
      
    
    def load_frames(self,video_name):   
      self.video_name = video_name
    
      # set image counter
      self.read_image_from_file()  # read figure in self.curr_photoimage
      self.create_photo_from_raw()
      
      self.video_num_of_frames = self.list_number_of_frames[self.video_index]
      
      self.rectangle_frame_pairs = [0]*self.video_num_of_frames
      self.img_start_x = 100
      self.img_start_y = 100
      self.img_id = self.canvas.create_image(self.img_start_x, self.img_start_y, image = self.curr_photoimage, anchor="nw") #, anchor = NW
      
      
      # this data is used to keep track of an 
      # item being dragged
      self._drag_data = {"x": 0, "y": 0, "item": None}
      
      # create a couple movable objects
      self.polygon_id = 0
      
      # [changeable]
      # rectangle size in x and y
      rec_h = 50
      rec_w = 100
      self.rectangle_size = [rec_w, rec_h]
      self.rectangle_label.winfo_children()[0].config(text = "w: {}".format(rec_w))
      self.rectangle_label.winfo_children()[1].config(text = "h: {}".format(rec_h))

      
      img_width = self.curr_photoimage.width()
      img_height = self.curr_photoimage.height()
      self.create_token((img_width/2 + self.img_start_x, img_height/2 + self.img_start_y), "blue", self.rectangle_size) # in here tags="token" assigned 
      
      # add bindings for clicking, dragging and releasing over
      # any object with the "token" tag
      self.canvas.tag_bind("token", "<ButtonPress-1>", self.OnTokenButtonPress)
      self.canvas.tag_bind("token", "<ButtonRelease-1>", self.OnTokenButtonRelease)
      self.canvas.tag_bind("token", "<B1-Motion>", self.OnTokenMotion)
      
      # add bindings for arrow keys when changing the image to right
      self.canvas.bind("<Return>", self.returnKey)
      self.canvas.bind("<Right>", self.rightKey)
      self.canvas.bind("<Left>", self.leftKey)
      self.canvas.bind("<Key-BackSpace>", self.backspaceKey)
      self.canvas.focus_set()
      
      # initialize tracker
      self.tracker = dlib.correlation_tracker()
      self.prev_tracker = self.tracker
      self.flag = 0
      
      # Video window update
      self.video_info_label.winfo_children()[0].config(text = "Video: {0:0{width}}/{1}".format(self.video_index+1, self.total_num_of_videos, width = 2))        
      self.video_info_label.winfo_children()[1].config(text = "Video: " + self.video_name)
      
      # Frames widnow update
      self.frame_info_label.winfo_children()[0].config(text = "Frame: {0:0{width}}/{1}".format(self.img_num+1, self.video_num_of_frames, width = 3))
      self.frame_info_label.winfo_children()[1].config(text="Frame: {0}".format(self.video_name + "_{0}.png".format(self.img_num+1)))
      
      # Annotation window update
      self.frame_annot_label.winfo_children()[0].config(text="Annotated frames: 000/{0}".format(self.video_num_of_frames))        
     
      #check if there is an annotation model for this video
      model_annot_name = os.path.join(self.annot_save_folder, self.video_name + ".model")
      if os.path.exists(model_annot_name):
	self.load()

    
  
    def read_image_from_file(self, image_num = -1):
      # if no parameter is passed set to self.img_num + 1
      if (image_num == -1):
	    image_num = self.img_num
      f = os.path.join(self.frames_folder,self.video_name + "_{0}.png".format(image_num+1))
      self.curr_image_raw = io.imread(f)
    


    def create_photo_from_raw(self):
      self.curr_photoimage = ImageTk.PhotoImage(image = Image.fromarray(self.curr_image_raw))
      #print "Size of image: width: ", self.curr_photoimage.width(), ", height ", self.curr_photoimage.height()
    
    def get_coord_rectangle(self):
      ''' Get coordinates of rectangle relative to image '''
      coords_rectangle = self.canvas.coords(self.polygon_id)
      coords_rectangle = [long(c) for c in coords_rectangle]
      coords_image = self.canvas.coords(self.img_id) 
      coords_image = [long(c) for c in coords_image]
      coords_relative = [coords_rectangle[0]-coords_image[0],coords_rectangle[1]-coords_image[1],coords_rectangle[4]-coords_image[0],coords_rectangle[5]-coords_image[1]]
      return coords_relative      
    
    def change_image(self):
    
      # put here change rectangle  
      self.read_image_from_file()
      self.create_photo_from_raw()
      self.canvas.itemconfig(self.img_id, image = self.curr_photoimage)
      self.frame_info_label.winfo_children()[0].config(text = "Frame: {0:0{width}}/{1}".format(self.img_num+1, self.video_num_of_frames, width = 3))
      self.frame_info_label.winfo_children()[1].config(text="Frame: {0}".format(self.video_name + "_{0}.png".format(self.img_num+1)))
      if (self.rectangle_frame_pairs[self.img_num] == 0):
	self.canvas.itemconfig(self.polygon_id, outline = "blue")
      else:
	self.canvas.itemconfig(self.polygon_id, outline = "red") 
      
      counter = 0
      for x in self.rectangle_frame_pairs:
	if x is not 0:
	  counter = counter + 1
      
      self.frame_annot_label.winfo_children()[0].config(text="Annotated frames: {0:0{width}}/{1}".format(counter, len(self.rectangle_frame_pairs), width=3))
      
    def change_rectangle(self):
      rel_position = self.rectangle_frame_pairs[self.img_num]
      curr_position = self.get_coord_rectangle()
      #print (rel_position.left())
      self.canvas.move(self.polygon_id, -curr_position[0]+rel_position[0], -curr_position[1]+rel_position[1])
    
       
    def image_segmentation(self, window_size, overlap = 0):
      # segmentations folder
      segmentation_folder = self.output_folder + "segmentations_"+ str(self.rectangle_size[0]) + "x" + str(self.rectangle_size[1]) +"/"
      if not os.path.exists(segmentation_folder):
	os.makedirs(segmentation_folder)
      
      # make sure to delete any segments that existed before
      counter = 0
      while(1):
	segment_file = "{0}/{1}_{2}_{3}.png".format(segmentation_folder,self.video_name, self.img_num+1,counter+1)
	if os.path.exists(segment_file):
	  os.remove(segment_file)
	else:
	  break
	counter += 1
	
      img = self.curr_image_raw
      img_width = self.curr_photoimage.width()
      img_height = self.curr_photoimage.height()

      coord_x = 0
      coord_y = 0     
      counter = 0

      end_loop = 0
      if overlap > 0:
	while end_loop == 0:
	  Image.fromarray(img[coord_x:coord_x + window_size[0],
		          coord_y:coord_y + window_size[1],:]).save("{0}/{1}_{2}_{3}.png".format(segmentation_folder,self.video_name,self.img_num+1,counter+1))
	  counter = counter + 1
	  print "Counter:", counter, "| rows:", coord_x, "->",coord_x + window_size[1], "| cols:", coord_y, "->", coord_y + window_size[0]
	  coord_y = coord_y+overlap
	  if coord_y >= img_width-window_size[1]:
	    coord_x = coord_x+overlap
	    coord_y = 0
	  if coord_x >= img_height-window_size[0]:
	    end_loop = 1
	    #break
	    
	  
      elif overlap == 0:
	while counter < (img_width/window_size[1])*(img_height/window_size[0]):
	  Image.fromarray(img[coord_x:coord_x + window_size[0],
			  coord_y:coord_y + window_size[1],:]).save("{0}/{1}_{2}_{3}.png".format(segmentation_folder,self.video_name,self.img_num+1,counter+1))
	  counter = counter + 1
	  print "Counter:", counter, "| rows:", coord_x, "->",coord_x + window_size[0], "| cols:", coord_y, "->", coord_y + window_size[1]
	  coord_y = coord_y+window_size[1]
	  if coord_y >= img_width-window_size[1]:
	    coord_x = coord_x+window_size[0]
	    coord_y = 0
	  if coord_x >= img_height-window_size[0]:
	    break
	  
	
				       	
      print "Video:" , self.video_name
      print "-Segmentation with overlap",overlap,"done for frame:", self.img_num + 1, "|height:", self.rectangle_size[1],"|width:", self.rectangle_size[0] 
      tkMessageBox.showinfo(title = "Congrats", message = "Segmentation done!")
    
    
    
   
    def sliding_window(self, window_size, image_num = 0, overlap = 0):
      
      # pos patches folder
      patches_pos_size = "patches_pos_" + str(self.rectangle_size[0]) + "x" + str(self.rectangle_size[1]) 
      patches_folder_pos = self.output_folder +  patches_pos_size + "/"
      if not os.path.exists(patches_folder_pos):
	os.makedirs(patches_folder_pos)
	
      # neg patches folder
      patches_neg_size = "patches_neg_" + str(self.rectangle_size[0]) + "x" + str(self.rectangle_size[1])
      patches_folder_neg = self.output_folder + patches_neg_size + "/"
      if not os.path.exists(patches_folder_neg):
	os.makedirs(patches_folder_neg)    
      
      
      file_pos = os.path.join(self.output_folder, "pos.txt")
      file_neg = os.path.join(self.output_folder, "neg.txt")
      
      fop = open(file_pos, "a")
      fon = open(file_neg, "a")
      self.read_image_from_file(image_num)
      img = self.curr_image_raw
      img_width = self.curr_photoimage.width()
      img_height = self.curr_photoimage.height()
      
      coord_relative = self.rectangle_frame_pairs[image_num]
      coord_x = coord_relative[1]
      coord_y = coord_relative[0]
      p = 0
      counter = 0
      
      while p < (img_width/window_size[1])*(img_height/window_size[0]):
	
	if p == 0:
	  fop.write("{0}/{1}_{2}_{3}.png 1 \n".format(patches_pos_size,self.video_name, image_num+1, counter+1))
	  Image.fromarray(img[coord_x:coord_x+window_size[0],
			coord_y:coord_y+window_size[1],:]).save("{0}/{1}_{2}_{3}.png".format(patches_folder_pos,self.video_name,image_num+1,counter+1))
	else:
	  fon.write("{0}/{1}_{2}_{3}.png 0 \n".format(patches_neg_size,self.video_name, image_num+1, counter+1))
	  Image.fromarray(img[coord_x:coord_x+window_size[0],
			coord_y:coord_y+window_size[1],:]).save("{0}/{1}_{2}_{3}.png".format(patches_folder_neg,self.video_name,image_num+1,counter+1))
	  
	#print "Counter:", p+1, "| rows:", coord_x, "->",coord_x + window_size[0], "| cols:", coord_y, "->", coord_y + window_size[1]
	coord_y = coord_y+window_size[1]
	#proverka dali odi vo nov red, proveri na mal primer
	if coord_y >= img_width-window_size[1]:
	  coord_x = coord_x+window_size[0]
	  coord_y = 0
	if coord_x >= img_height-window_size[0]:
	  break
	
	p = p+1
	
	counter = counter + 1
	
      p = 0
      coord_x = coord_relative[1]
      coord_y = coord_relative[0]
      while p < (img_width/window_size[1])*(img_height/window_size[0]):
	p = p+1
	coord_y = coord_y-window_size[1]
	# it should print it down for (0,0), 
	if coord_y < 0:
	  coord_x = coord_x-window_size[0]
	  coord_y = img_width-window_size[1]
	if coord_x < 0:
	  break      

	Image.fromarray(img[coord_x:coord_x+window_size[0],
			coord_y:coord_y+window_size[1],:]).save("{0}/{1}_{2}_{3}.png".format(patches_folder_neg,self.video_name,image_num + 1,counter+1))
	fon.write("{0}/{1}_{2}_{3}.png 0 \n".format(patches_neg_size,self.video_name, image_num+1, counter+1))

	
	counter = counter + 1
      
      fop.close()
      fon.close()
      # or maybe instead of this, create non-overlapping windows, but that way it can miss
      # the positive sample
      print "patches successfully saved for image: ", image_num + 1






    def extract_patches(self):
      # check if there is a frame folder loaded first
      if not self.frames_folder:
	tkMessageBox.showinfo(title = "Warning", message = "Load video frames directory first")
	return
       
      # check if there is annotated model for the current video
      model_annot_name = os.path.join(self.annot_save_folder, self.video_name + ".model")
      if not os.path.exists(model_annot_name):
	tkMessageBox.showinfo(title = "Warning", message = "Annotated model doesn't exist for this video")
      
      else:
	annotated_frames_rectangle_pairs = self.load_annotations_from_file(model_annot_name)
	
	num_of_annotations = sum([int(i !=0) for i in annotated_frames_rectangle_pairs])
	print num_of_annotations, "annotations"
  
	index_annotation = 0
	image_number = 0
	
	while index_annotation < num_of_annotations:
	  if annotated_frames_rectangle_pairs[image_number] <> 0:
	    self.sliding_window ((self.rectangle_size[1],self.rectangle_size[0]), image_number, 0)
	    index_annotation += 1
	  image_number += 1
	
	# check verbose
	print "patches were saved congrats to you!!!"
	
   
    def extract_patches_tf(self):
      # check if there is a frame folder loaded first
      if not self.frames_folder:
	tkMessageBox.showinfo(title = "Warning", message = "Load video frames directory first")
	return
      
      overlap = tkSimpleDialog.askinteger("Overlap","Choose an overlap",parent = self.canvas)
      if overlap == None:
	tkMessageBox.showinfo(title = "Error", message = "Overlap was not entered")
	return
      
      if overlap < 0:
	tkMessageBox.showinfo(title = "Error", message = "Overlap must be a positive number or 0")
	return
     

      self.image_segmentation((self.rectangle_size[1],self.rectangle_size[0]), overlap)
 
      
    
    def save(self):
      # check if there is a frame folder loaded first
      if not self.frames_folder:
	tkMessageBox.showinfo(title = "Warning", message = "Load video frames directory first")
	return
      
      # don"t save if there is not at least one annotated frame
      number_of_annotaded_frames = sum([int(i !=0) for i in self.rectangle_frame_pairs])
      if number_of_annotaded_frames == 0:
	tkMessageBox.showinfo(title = "Warning", message = "At least annotate one frame!")
	return
      
      
      # check if there is already a model 
      result = "yes"
      model_annot_name = os.path.join(self.annot_save_folder, self.video_name + ".model")
      if os.path.exists(model_annot_name):
	result = tkMessageBox.askquestion("Overwrite", "Are you sure?", icon = "warning")
	
      if result == "yes":
	f = file(model_annot_name, 'wb')
	cPickle.dump(self.rectangle_frame_pairs, f, protocol = cPickle.HIGHEST_PROTOCOL)
	f.close()
	tkMessageBox.showinfo(title = "Info", message = "Annotation model saved")	
      else:
	tkMessageBox.showinfo(title = "Info", message = "Annotation model not saved")
    
    
    
    def load(self):
      if not self.frames_folder:
	tkMessageBox.showinfo(title = "Warning", message = "Load video frames directory first")
	return
      
      
      model_annot_name = os.path.join(self.annot_save_folder, self.video_name + ".model")
      #check if there is a model for the loaded frames
      if not os.path.exists(model_annot_name):
	tkMessageBox.showinfo(title = "Info", message = "No existing annotation model")
	return
           
      previous_frames = self.load_annotations_from_file(model_annot_name)
      self.rectangle_frame_pairs[0:len(previous_frames)] = previous_frames  
      
      counter = 0
      done = False
      w = 0
      h = 0
      for x in self.rectangle_frame_pairs:
	if x is not 0:
	  counter = counter + 1
	  # get the width and the height of the rectangle from the annotations
	  if done is False:
	    done = True
	    w = x[2] - x[0]
	    h = x[3] - x[1]

      self.rectangle_change_size("both", w , h)
      if (self.rectangle_frame_pairs[self.img_num] is not 0):
	self.change_rectangle()
	self.canvas.itemconfig(self.polygon_id, outline = "red")
      self.frame_annot_label.winfo_children()[0].config(text="Annotated frames: {0:0{width}}/{1}".format(counter, len(self.rectangle_frame_pairs), width=3))
      tkMessageBox.showinfo(title = "Info", message = "Annotation model loaded")
      
       
    
    def rightKey(self, event):      
      self.img_num +=1 
      if self.img_num >= self.video_num_of_frames:
	
	#delete rectangle
	self.canvas.delete(self.polygon_id)
	
	save_annot = "no"
	save_annot = tkMessageBox.askquestion("End of video frames", "Save annotations?", icon = "warning")
	if save_annot == "yes":
	  self.save()
	else:
	  tkMessageBox.showinfo(title = "Info", message = "Annotation model not saved")
	  
	self.img_num = 0
	self.video_index+=1
	if self.video_index == self.total_num_of_videos:
	  self.video_index = 0
	  
	self.load_frames(self.list_of_videos[self.video_index])
      
	# if rectangle exists redraw it
      if (self.rectangle_frame_pairs[self.img_num] is not 0):
	self.change_rectangle() 
      self.change_image()
      #print self.img_num/float(20*3600)
      
    
    # same code in both right and left key, refactor
    
    def leftKey(self, event):
      self.img_num -=1 
      if self.img_num < 0: 
	# delete rectangle
	self.canvas.delete(self.polygon_id)
	
	self.video_index-=1
	if self.video_index == -1:
	  self.video_index = self.total_num_of_videos-1
	
	# load frames of the previous video
	self.img_num = self.list_number_of_frames[self.video_index] - 1
	
	self.load_frames(self.list_of_videos[self.video_index])
	
	# if rectangle exists redraw it
      if (self.rectangle_frame_pairs[self.img_num] is not 0):
	self.change_rectangle() 
      self.change_image()
    
    
    
    def returnKey(self, event):
        #save rectangle position
        img_width = self.curr_photoimage.width()
	img_height = self.curr_photoimage.height()
	
        self.rectangle_frame_pairs[self.img_num] = self.get_coord_rectangle()      
	coords_relative = self.get_coord_rectangle()
	
	#check if there was bad annotations
	# rectangle is defined by left top corner and right bottom corner
	left_upper_x = coords_relative[0]
	left_upper_y = coords_relative[1]
	right_bottom_x = coords_relative[2]
	right_bottom_y = coords_relative[3]
	
	# check if the rectangle is outside the image borders
	if left_upper_x < 0 or left_upper_y <0 or right_bottom_x >= img_width or right_bottom_y >= img_height:
	  tkMessageBox.showinfo(title = "Info", message = "Bad annotation, try again")
	  return
	
	if (self.flag == 0):  
	  #proveri uste ednas koordinatite
	  self.tracker.start_track(self.curr_image_raw, dlib.rectangle(coords_relative[0],coords_relative[1],coords_relative[2],coords_relative[3]))
	  #self.tracker.start_track(self.images_raw[0], dlib.rectangle(170, 200, 240, 240))
	  self.flag = 1 
	  
        else: 
	  
	  #update filter
	  self.tracker.update(self.curr_image_raw, dlib.rectangle(coords_relative[0],coords_relative[1],coords_relative[2],coords_relative[3]))
	  
	  # update rectangle (overlay it)
	  rel_position = self.tracker.get_position()
	  curr_position = self.get_coord_rectangle()
	
	  # refactor this code as well
	  self.canvas.move(self.polygon_id, -curr_position[0]+rel_position.left(), -curr_position[1]+rel_position.top())
      
	self.img_num += 1
	
	# check if this is the last frame
	if self.img_num >= self.video_num_of_frames:
	  # delete polygon
	  self.canvas.delete(self.polygon_id)
	  save_annot = "no"
	  save_annot = tkMessageBox.askquestion("End of video frames", "Save annotations?", icon = "warning")
	  if save_annot == "yes":
	    self.save()
	  else:
	    tkMessageBox.showinfo(title = "Info", message = "Annotation model not saved")
	  self.img_num = 0
	  self.video_index+=1
	  # check if the current video is the last video
	  if self.video_index == self.total_num_of_videos:
	    self.video_index = 0
	    
	  # load frames of the next video
	  self.load_frames(self.list_of_videos[self.video_index])
	  
	self.change_image()
	
	
    def backspaceKey(self, event):
      self.rectangle_frame_pairs[self.img_num] = 0
      self.canvas.itemconfig(self.polygon_id, outline = "blue")
      counter = 0
      for x in self.rectangle_frame_pairs:
	if x is not 0:
	  counter = counter + 1
      self.frame_annot_label.winfo_children()[0].config(text="Annotated frames: {0:0{width}}/{1}".format(counter, len(self.rectangle_frame_pairs), width=3))

    
    def OnTokenButtonPress(self, event):
        '''Being drag of an object'''
        # record the item and its location
        self._drag_data["item"] = self.canvas.find_closest(event.x, event.y)[0]
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y 
      
        
        # put item to front
        self.canvas.tag_raise(self._drag_data["item"])

    def OnTokenButtonRelease(self, event):
        '''End drag of an object'''
        # reset the drag information
        self._drag_data["item"] = None
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0

    def OnTokenMotion(self, event):
        '''Handle dragging of an object'''
        # compute how much this object has moved
        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]
        
        # move the object the appropriate amount
        self.canvas.move(self._drag_data["item"], delta_x, delta_y)
       
        # record the new position
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = "Annotation Program")
  parser.add_argument("-mf", dest = "main_folder", type = str,
		      help = "path to the main folder containing the videos folder (default: current directory)")
  parser.add_argument("-v", dest = "videos", type = str,
		      help = "path to a video file (default: all videos)")
  parser.add_argument("-fps", dest = "fps", type = int, default = 24, 
		      help = "frames per second (default: 24 fps)")
  parser.add_argument("-of", dest = "output_folder", type = str, 
		      help = "path to an output folder (default: current directory)")
  args = parser.parse_args()
  
  from skimage import io # image read and conversion to array # had to import it here (conflicts with the args passed)
  app = SampleApp()
  app.mainloop()
