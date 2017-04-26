import Tkinter as tk      # GUI
import tkFileDialog  # for browsing files
import tkMessageBox # warning boxes
import tkSimpleDialog # pop up window with entry
import os              # system calls
from PIL import ImageTk, Image # images in GUI and image processing
import dlib              # correlational tracker
import glob        # some linux command functions
import numpy as np    # matlab python stuff
import cPickle        # saving rectangle pairs (list i/o)
import datetime        # date time function
import time
import argparse # for the arguments passed
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image # just another image processing lib
from cv2 import imread

from extract_frames import extract_frames_from_videos # frame extraction function
from extract_frames import get_video_file_name
from compute_masks import create_mask_for_image # mask for single image
from augmentation import augment, augment_bg
from export_data import export

COLOR = False

class SampleApp(tk.Tk):  # inherit from Tk class
    '''Illustrate how to drag items on a Tkinter canvas'''

    def __init__(self):
        tk.Tk.__init__(self)


        ###### GUI #####
        self.title("Data annotation")
        # create a canvas
        self.canvas = tk.Canvas(width=1400, height=640)
        self.canvas.pack(fill="both", expand=True)

        self.menubar = tk.Menu(self)
        # file menu bar
        filemenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Exit", command=self.quit)

        # help menu bar
        self.show_actions_flag = tk.IntVar()

        helpmenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_checkbutton(label="Actions", variable=self.show_actions_flag, command=self.show_actions)


        self.config(menu=self.menubar)

        '''create Frames section'''

        # Video label window
        self.video_info_label = tk.LabelFrame(self.canvas, text = "Video information", padx=5, pady=5)
        self.canvas.create_window(800,320, anchor = "nw", window = self.video_info_label, width=400, height=90)
        tk.Label(self.video_info_label, text = "Video: No info").pack()
        tk.Label(self.video_info_label, text = "Video: No info").pack()

        next_video_btn = tk.Button(self.video_info_label, text="Next", command=lambda: self.next_video())
        next_video_btn.configure(width=10)
        next_video_btn.pack(side = "right",anchor="ne")

        prev_video_btn = tk.Button(self.video_info_label, text="Previous", command=lambda: self.next_video(forward=False))
        prev_video_btn.configure(width=10)
        prev_video_btn.pack(anchor="nw")

        # Frame label window
        self.frame_info_label = tk.LabelFrame(self.canvas, text="Frame information", padx=5, pady=5)
        self.canvas.create_window(800, 430, anchor="nw", window=self.frame_info_label, width=400, height=70)
        tk.Label(self.frame_info_label, text="Frame: No info").pack()
        tk.Label(self.frame_info_label, text="Frame: No info").pack()

        #------------------------------------------------------------------------------------------------------------------------------------------#
        # Annotation information frame
        self.frame_annot_label = tk.LabelFrame(self.canvas, text="Annotation information", padx=5, pady=5)
        self.canvas.create_window(800, 510, anchor="nw", window=self.frame_annot_label, width=240, height=75)
        tk.Label(self.frame_annot_label, text="Annotated frames: No info").pack()

        # load_btn
        load_btn = tk.Button(self.frame_annot_label, text = "Load", command = self.load)
        load_btn.pack(side="right",anchor="center")

        # save_btn
        save_btn = tk.Button(self.frame_annot_label, text = "Save", command = self.save)
        save_btn.pack(side="left")
        #------------------------------------------------------------------------------------------------------------------------------------------#
        # Patches frame
        self.patches_frame = tk.LabelFrame(self.canvas, text="Extract patches per", padx=5, pady=5)
        self.canvas.create_window(1110,235, anchor="nw", window=self.patches_frame, width=170, height=75)

        extract_patches_all_frames_btn = tk.Button(self.patches_frame, text = "Video", command = self.extract_patches)
        extract_patches_all_frames_btn.config(width=5,height=2)
        extract_patches_all_frames_btn.pack(side="right")


        extract_patches_this_frame_btn = tk.Button(self.patches_frame, text = "Frame", command = self.extract_patches_tf)
        extract_patches_this_frame_btn.config(width=5,height=2)
        extract_patches_this_frame_btn.pack(side="left")

        #------------------------------------------------------------------------------------------------------------------------------------------#
        # Annotation options window
        self.annotation_options_label = tk.LabelFrame(self.canvas, text = "Annotation options", padx = 5, pady =5)
        self.canvas.create_window(800, 10,anchor = "nw", window = self.annotation_options_label,width = 260,height = 305)

        self.shape = tk.IntVar()
        #set rectangle as default choice
        self.shape.set(0)
        # Radio buttons for the rectangle and the circle
        radio_rectangle = tk.Radiobutton(self.annotation_options_label,text ="rectangle",
                         value=0, variable=self.shape, command=self.get_shape).pack(side="left", anchor="nw")
        radio_circle = tk.Radiobutton(self.annotation_options_label,text ="circle",
                         value=1, variable=self.shape, command=self.get_shape).pack(side="left", anchor="ne", padx=20)
        #--------------------------
        # rectangle frame
        self.rectangle_frame = tk.LabelFrame(self.canvas, padx=5, pady=5)
        self.canvas.create_window(810,55, anchor="nw", window=self.rectangle_frame, width=100, height=180)

        # width frame
        self.rec_width_frame = tk.LabelFrame(self.canvas, padx=5, pady=5)
        self.canvas.create_window(820,60, anchor="nw", window=self.rec_width_frame, width=80, height=80)

        # rectangle width button
        rec_width_btn = tk.Button(self.rec_width_frame, text="width",command=lambda: self.rectangle_change_size(w_flag=True, ask=True), width=5)
        rec_width_btn.pack(side="top", anchor="nw")

        # rectangle width slider
        self.slider_rec_w = tk.Scale(self.rec_width_frame, orient=tk.HORIZONTAL, length=70, sliderlength=10, from_=1, to=640, command=lambda _:self.rectangle_change_size(w_flag=True))
        self.slider_rec_w.pack(side="left", anchor="nw")

        # rectangle height frame
        self.rec_height_frame = tk.LabelFrame(self.canvas, padx=5, pady=5)
        self.canvas.create_window(820,145, anchor="nw", window=self.rec_height_frame, width=80, height=80)

        # rectangle heigth button
        rec_height_btn = tk.Button(self.rec_height_frame, text="height",command=lambda: self.rectangle_change_size(h_flag=True, ask=True), width=5)
        rec_height_btn.pack(side="top", anchor="n")

        # rectangle height slider
        self.slider_rec_h = tk.Scale(self.rec_height_frame, orient=tk.VERTICAL, length=70, sliderlength=5, from_=1, to=480, command= lambda _:self.rectangle_change_size(h_flag = True))
        self.slider_rec_h.pack(side="bottom", anchor="nw")
        #----------------
        # circle frame
        self.circle_frame = tk.LabelFrame(self.canvas, padx=5, pady=5)
        self.canvas.create_window(920,55, anchor = "nw", window = self.circle_frame, width = 80, height = 80)

        # circle radius button
        circle_rad_btn = tk.Button(self.circle_frame, text = "radius", width=5)
        circle_rad_btn.pack(side="top",anchor="nw")

        # circle radius slider
        self.slider_circle_r = tk.Scale(self.circle_frame, orient=tk.HORIZONTAL, length=70, sliderlength=10, from_=1, to=50)
        self.slider_circle_r.pack(side= "bottom", anchor = "nw")
        #-------------------
        # labels frame
        self.annotation_labels = tk.LabelFrame(self.canvas, text="Labels", padx=5, pady=5)
        self.canvas.create_window(810,240, anchor="nw", window=self.annotation_labels, width=240, height=40)

        self.label = tk.IntVar()
        self.label.set(1)
        self.label_number = self.label.get()
        self.label_colors = ['black','red','green','yellow','magenta','cyan']
        self.num_labels = 5
        label_0 = tk.Radiobutton(self.annotation_labels,text ="0",
                         value = 0, variable=self.label,fg=self.label_colors[0], command=self.update_label).pack(side="left")
        label_1 = tk.Radiobutton(self.annotation_labels,text ="1",
                         value = 1, variable=self.label, fg=self.label_colors[1], command=self.update_label).pack(side="left")
        label_2 = tk.Radiobutton(self.annotation_labels,text ="2",
                         value = 2, variable=self.label, fg=self.label_colors[2], command=self.update_label).pack(side="left")
        label_3 = tk.Radiobutton(self.annotation_labels,text ="3",
                         value = 3, variable=self.label, fg=self.label_colors[3], command=self.update_label).pack(side="left")
        label_4 = tk.Radiobutton(self.annotation_labels,text ="4",
                         value = 4, variable=self.label, fg=self.label_colors[4], command=self.update_label).pack(side="left")
        label_5 = tk.Radiobutton(self.annotation_labels,text ="5",
                         value = 5, variable=self.label, fg=self.label_colors[5], command=self.update_label).pack(side="left")

        # check box
        self.show_mask_flag = tk.IntVar()
        check_box_mask = tk.Checkbutton(self.annotation_options_label, text="Show labels only", variable=self.show_mask_flag, command=self.show_masks)
        check_box_mask.place(x=0, y=250)

        #------------------------------------------------------------------------------------------------------------------------------------------#
        # Augmentation frame
        self.augmentation_frame = tk.LabelFrame(self.canvas, text="Augmentation")
        self.canvas.create_window(1110, 10, anchor="nw", window=self.augmentation_frame, width=170, height=150)

        # aumentation options label
        augmentation_options_label = tk.Label(self.augmentation_frame, text="Options")
        augmentation_options_label.place(x=5)

        # check boxes
        self.rotation = tk.IntVar()
        check_box_1 = tk.Checkbutton(self.augmentation_frame,text="rotation", variable=self.rotation, command=self.check_augmentation_boxes)
        check_box_1.place(x=0,y=20)

        self.color = tk.IntVar()
        check_box_2 = tk.Checkbutton(self.augmentation_frame,text="color", variable=self.color, command=self.check_augmentation_boxes)
        check_box_2.place(x=0,y=40)

        self.scale = tk.IntVar()
        check_box_3 = tk.Checkbutton(self.augmentation_frame, text="scale", variable=self.scale, command=self.check_augmentation_boxes)
        check_box_3.place(x=0, y=60)

        # Random number label
        random_number_label = tk.Label(self.augmentation_frame, text="# Random")
        random_number_label.place(x=90)

        # Rotation entry
        self.rotation_rand_num = tk.StringVar()
        self.rotation_entry = tk.Entry(self.augmentation_frame, width=5, textvariable=self.rotation_rand_num, state='disabled')
        self.rotation_entry.place(x=100, y=20)

        # Color entry
        self.color_rand_num = tk.StringVar()
        self.color_entry = tk.Entry(self.augmentation_frame, width=5, textvariable=self.color_rand_num, state='disabled')
        self.color_entry.place(x=100, y=40)

        # Scale entry
        self.scale_rand_num = tk.StringVar()
        self.scale_entry = tk.Entry(self.augmentation_frame, width=5, textvariable=self.scale_rand_num, state='disabled')
        self.scale_entry.place(x=100, y=60)

        # augment button
        augment_btn = tk.Button(self.augmentation_frame, text="Augment", command=self.augment_data)
        augment_btn.place(x=40, y=90)
        self.augmentation_flag = 0

        #-----------------------------------------------------------------------------------------------------------------------------------------#
        # Backgournd Freme
        self.bg_frame = tk.LabelFrame(self.canvas, text="Background")
        self.canvas.create_window(1290, 10, anchor="nw", window=self.bg_frame, width=170, height=150)
        # aumentation options label
        augmentation_options_label = tk.Label(self.bg_frame, text="Options")
        augmentation_options_label.place(x=5)

        # BG Color canvas
        self.bgcolor = tk.IntVar()
        self.bgcolor_check_box = tk.Checkbutton(self.bg_frame, text="bg subtraction", variable=self.bgcolor, command=self.check_augmentation_boxes)
        self.bgcolor_check_box.place(x=0, y=20)
        self.sensitivity = 0

        self.custom_bg = tk.IntVar()
        self.custom_bg_img = None
        self.custom_bg_check_box = tk.Checkbutton(self.bg_frame, text="custom bg", variable=self.custom_bg, command=self.check_augmentation_boxes,state='disabled')
        self.custom_bg_check_box.place(x=0, y=40)
        self.bgcolor_rgb = []
        self.bgcolor_canvas = tk.Canvas(self.bg_frame, height = 20, width=40)
        self.bgcolor_canvas.place(x=100, y=20)

        # Cutom background button
        self.custom_bg_btn = tk.Button(self.bg_frame, text="bg", command=self.add_custom_bg, padx=5, pady=3)
        self.custom_bg_btn.config(width=4, state='disabled')
        self.custom_bg_btn.place(x=100, y=40)

        #-----------------------------------------------------------------------------------------------------------------------------------------#
        # Export Frame
        self.export_frame = tk.LabelFrame(self.canvas, text="Export data as:", padx=5, pady=5)
        self.canvas.create_window(1110, 170, anchor="nw", window=self.export_frame, width=170, height=50)

        # export hdf5 button
        hdf5_export_btn = tk.Button(self.export_frame, text="HDF5", command=lambda: self.export_fun("hdf5"), padx=5, pady=5)
        hdf5_export_btn.configure(width= 4)
        hdf5_export_btn.pack(side="right")

        # export images button
        images_export_btn = tk.Button(self.export_frame, text = "PNG", command = lambda: self.export_fun("image"), padx=5, pady=5)
        images_export_btn.configure(width= 4)
        images_export_btn.pack(side="left")

        # export mat button
        mat_export_btn = tk.Button(self.export_frame, text = "Mat", command = lambda: self.export_fun("mat"), padx=10, pady=5)
        mat_export_btn.configure(width= 4)
        mat_export_btn.pack(anchor="center")

        #------------------------------------------------------------------------------------------------------------------------------------------#
        # Folder settings

        # main folder
        current_dir = os.getcwd() + "/"
        main_folder = current_dir
        if args.main_folder is None:
          print "WARNING!! main folder path was not passed (default: current directory)"
        else:
          main_folder = args.main_folder
        main_folder = os.path.abspath(main_folder) + "/"
        # change directory to main folder
        os.chdir(main_folder)

        # output folder
        output_folder = current_dir
        if args.output_folder is None:
          print "WARNING!! output folder path was not passed (default: current directory)"
        else:
          output_folder = args.output_folder
        output_folder = os.path.abspath(output_folder) + "/"
        self.output_folder = output_folder

        # videos folder
        videos_folder = main_folder + "videos/"
        # check if the videos folders exists
        if not os.path.isdir(videos_folder):
          print "Error: Main directory " + main_folder + " doesn't contain a folder named videos "
          exit(1)

        # annnotations folder
        annotation_folder = output_folder + "annotations/"
        #check if already exists
        if not os.path.exists(annotation_folder):
          os.makedirs(annotation_folder)
        self.annotation_folder = os.path.abspath(annotation_folder)

        # masks folder
        mask_folder =  os.path.join(output_folder,"masks/")
        if not os.path.exists(mask_folder):
          os.makedirs(mask_folder)
        self.mask_folder = mask_folder

        # frames folder
        if args.frames_folder is not None:
          frames_folder_path = args.frames_folder
          self.frames_folder = os.path.abspath(frames_folder_path) + "/"
          # check if the directory is empty
          if os.listdir(self.frames_folder) == []:
            print "Error: No frames in", self.frames_folder
            exit(1)

        else:
          frames_folder_path = output_folder + "frames/"
          # check if the frames folder exists
          if not os.path.exists(frames_folder_path):
            # create a folder for the frames
            os.makedirs(frames_folder_path)
          self.frames_folder = frames_folder_path
          # extract frames
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


    def show_masks(self):
    # don't show masks
      if self.show_mask_flag.get() == 0:
        self.curr_photo_image_mask_only = ImageTk.PhotoImage(image = Image.fromarray(self.curr_image_raw))
        self.canvas.itemconfig(self.img_id, image = self.curr_photo_image_mask_only)
        if self.label_number != 0:
          self.canvas.itemconfig(self.polygon_id[0], tags='token')

      # Show masks only
      else:
        self.curr_image_raw_masks_only = create_mask_for_image(self.curr_image_raw, self.rectangle_frame_pairs[self.img_num], self.label_number)
        # no annotations
        self.curr_photo_image_mask_only = ImageTk.PhotoImage(image = Image.fromarray(self.curr_image_raw_masks_only))
        self.canvas.itemconfig(self.img_id, image = self.curr_photo_image_mask_only)
        self.canvas.itemconfig(self.polygon_id[0], tags='')

    def augment_data(self):
      num_scales = 0
      num_rotations = 0
      num_colors = 0
      if self.bgcolor.get() and self.bgcolor_rgb == []:
        tkMessageBox.showerror(title="BG Color",
                               message="choose background color first!")
        return None

      if self.custom_bg.get() and self.custom_bg_img is None:
          tkMessageBox.showerror(title="Custom BG",
                                 message="choose custom background first!")
          return

      if self.rotation_rand_num.get() != "":
        # check if it is a positive int digit
        if not self.rotation_rand_num.get().isdigit():
          tkMessageBox.showinfo(title="Rotation number", message="Please enter an integer")
          return
        else:
          num_rotations = int(self.rotation_rand_num.get())

      if self.color_rand_num.get() != "":
        # check if it is a positive int digit
        if not self.color_rand_num.get().isdigit():
          tkMessageBox.showinfo(title="Color number", message="Please enter an integer")
          return
        else:
          num_colors = int(self.color_rand_num.get())

      if self.scale_rand_num.get() != "":
        # check if it is a positive int digit
        if not self.scale_rand_num.get().isdigit():
          tkMessageBox.showinfo(title="Scale number", message="Please enter an integer")
          return
        else:
          num_scales = int(self.scale_rand_num.get())

      downsample_x = 300
      downsample_y = 300

      self.data_set, self.label_set, self.num_colors, self.num_scales, self.num_rotations, self.video_names_list, self.annotated_frames_list, self.augmentation_flag = augment(self.augmentation_flag,
    downsample_x, downsample_y, self.total_num_of_frames, self.annotation_folder,
    self.frames_folder, self.output_folder, num_scales=num_scales,
    num_rotations=num_rotations, num_colors=num_colors, bg_sub=
    self.bgcolor.get(), bg_color=self.bgcolor_rgb, sensitivity=self.sensitivity,
    custom_bg=self.custom_bg.get(), custom_bg_img=self.custom_bg_img,
    color=COLOR)

      if self.augmentation_flag == -1:
        self.augmentation_flag = 0
        tkMessageBox.showinfo(title="Info", message="No annotation models exist")
        return

      tkMessageBox.showinfo(title="Info", message="Augmentation done !")


    def export_fun(self,type_data):
      if self.augmentation_flag == 0:
        tkMessageBox.showinfo(title="Info",message="Augment data first")
        return

      export(self.output_folder, self.data_set, self.label_set, self.num_colors, self.num_scales, self.num_rotations, self.video_names_list, self.annotated_frames_list, type_data, COLOR)
      tkMessageBox.showinfo(title="Info",message="Data exported !")

    def check_augmentation_boxes(self):
      if self.rotation.get():
        self.rotation_entry.config(state='normal')
      else:
        self.rotation_entry.delete(0, tk.END)
        self.rotation_entry.config(state='disabled')

      if self.color.get():
        self.color_entry.config(state='normal')
      else:
        self.color_entry.delete(0, tk.END)
        self.color_entry.config(state='disabled')

      if self.scale.get():
        self.scale_entry.config(state='normal')
      else:
        self.scale_entry.delete(0, tk.END)
        self.scale_entry.config(state='disabled')

      if self.bgcolor.get():
        if self.bgcolor_rgb == []:
            # ask user to select BG color
            if tkMessageBox.showinfo(title="BG color", message="Please select bg color"):
                self.canvas.bind('<Button-1>', self.OnPickColorCoord, add='+')

      else:
        self.bgcolor_rgb = []
        self.bgcolor_canvas.delete("all")
        self.bgcolor_canvas.config(state='disabled')
        self.custom_bg_check_box.deselect()
        self.custom_bg_check_box.config(state='disabled')
        self.custom_bg_btn.config(state='disabled')
        self.custom_bg_img = None
        self.read_image_from_file(self.img_num)
        self.create_photo_from_raw()
        self.canvas.itemconfig(self.img_id, image = self.curr_photoimage)

      if self.custom_bg.get():
        self.custom_bg_btn.config(state='normal')
      else:
        self.custom_bg_btn.config(state='disabled')
        if self.bgcolor.get() and self.bgcolor_rgb != []:
            self.custom_bg_img = None
            self.read_image_from_file(self.img_num)
            self.create_photo_from_raw()
            self.canvas.itemconfig(self.img_id, image = self.curr_photoimage)

      # return the focus to the canvas
      self.canvas.focus_set()

    def quit(self):
      exit(1)

    def show_actions(self):
      if self.show_actions_flag.get():
        # Action label window
        self.legend_label = tk.LabelFrame(self.canvas, text="Actions", padx=5, pady=5)
        self.canvas.create_window(1100, 510, anchor="nw", window=self.legend_label,width=220, height=90)
        tk.Label(self.legend_label, text = "<--  Left \n --> Right \n ReturnKey <--| Annotate \n BackSpace < Delete Annotation ").pack()
      else:
        self.legend_label.destroy()

    def create_circle(self,x,y,rad):
      pass
      #self.polygon_id = self.canvas.create_oval(200, 300, 250, 340,outline='blue',fill='',tags="token")

    def create_rectangle(self, center_rectangle, color, rectangle_size, new = True, tags_flag = True, index = 0):
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
      # create new polygon
      if new == True:
        if tags_flag == True:
          self.polygon_id[index] = self.canvas.create_polygon(l_u_c_x, l_u_c_y, r_u_c_x, r_u_c_y, r_b_c_x, r_b_c_y, l_b_c_x, l_b_c_y, outline=color, fill='', tags="token")
        if tags_flag == False:
          self.polygon_id[index] = self.canvas.create_polygon(l_u_c_x, l_u_c_y, r_u_c_x, r_u_c_y, r_b_c_x, r_b_c_y, l_b_c_x, l_b_c_y, outline=color, fill='')

      #change coords
      else:
        self.canvas.coords(self.polygon_id[index],l_u_c_x, l_u_c_y, r_u_c_x, r_u_c_y, r_b_c_x, r_b_c_y, l_b_c_x, l_b_c_y)

    def update_image_annotated_with_label(self,label_index):
      if label_index != -1:
        width = self.rectangle_frame_pairs[self.img_num][label_index][2] - self.rectangle_frame_pairs[self.img_num][label_index][0]
        height = self.rectangle_frame_pairs[self.img_num][label_index][3] - self.rectangle_frame_pairs[self.img_num][label_index][1]
        self.rectangle_size[0] = width
        self.rectangle_size[1] = height
        self.rectangle_change_size(w_flag=True, h_flag=True, w=width, h=height)
        self.move_rectangle(label_index)
        self.canvas.itemconfig(self.polygon_id[0], outline = self.label_colors[self.label_number])
      else:
        self.canvas.itemconfig(self.polygon_id[0], outline = "blue")


    def all_annotations_mode(self):
      # delete rectangle(s)
      for i in range(0, self.num_labels):
        self.canvas.delete(self.polygon_id[i])
      # check if this image is annotate
      if (self.rectangle_frame_pairs[self.img_num] is not 0):
        # get number of labels for this image
        size_labels = len(self.rectangle_frame_pairs[self.img_num])
        # go through the labels for this image and make annotations for all of them
        for i in range(0, size_labels):
          label = self.rectangle_frame_pairs[self.img_num][i][-1]
          rec_width = self.rectangle_frame_pairs[self.img_num][i][2] - self.rectangle_frame_pairs[self.img_num][i][0]
          rec_height = self.rectangle_frame_pairs[self.img_num][i][3] - self.rectangle_frame_pairs[self.img_num][i][1]
          rec_x_center = self.rectangle_frame_pairs[self.img_num][i][0] + rec_width/2 + self.img_start_x
          rec_y_center = self.rectangle_frame_pairs[self.img_num][i][1] + rec_height/2 + self.img_start_y
          self.create_rectangle((rec_x_center,rec_y_center),self.label_colors[label], [rec_width,rec_height] ,tags_flag=False, index = label-1)

    def update_label(self):
      self.label_number = self.label.get()

      if self.label_number == 0:
        self.all_annotations_mode()

      else:
        # delete rectangle(s)
        for i in range(0, self.num_labels):
          self.canvas.delete(self.polygon_id[i])

        # create new rectangle (default)
        img_width = self.curr_photoimage.width()
        img_height = self.curr_photoimage.height()
        rec_w = 100
        rec_h = 50
        self.create_rectangle((img_width/2 + self.img_start_x, img_height/2 + self.img_start_y), "blue", [rec_w,rec_h])
        self.rectangle_change_size(w_flag = True, h_flag = True, ask = False,w=rec_w, h=rec_h, change_size = False)
        # if this image is annotated
        if (self.rectangle_frame_pairs[self.img_num] is not 0):
          # check the annotation with the label selected
          label_index = self.get_label_index_in_list()
          # update image with the annotation of the label
          self.update_image_annotated_with_label(label_index)

          self.show_masks()

    def get_shape(self):
      shape = self.shape.get()
      img_width = self.curr_photoimage.width()
      img_height = self.curr_photoimage.height()

      if shape == 0:
        print "Rectangle"
        #self.canvas.delete(self.polygon_id)
        #self.create_rectangle((img_width/2 + self.img_start_x, img_height/2 + self.img_start_y), "blue", self.rectangle_size)

      elif shape == 1:
        print "Circle"
        #self.canvas.delete(self.polygon_id)
        #self.create_circle(300,200,40)

    def circle_change_size(self):
      print "circle change size"

    def rectangle_change_size(self, w_flag = False, h_flag = False, ask = False, w=0, h=0, change_size = True):

      if self.label_number == 0:
        return

      # get the relative coords of rectangle to the image
      rec_coord = self.get_coord_rectangle()
      rect_x_center = rec_coord[0] + self.rectangle_size[0]/2
      rect_y_center = rec_coord[1] + self.rectangle_size[1]/2

      width = None
      height = None
      img_width = self.curr_photoimage.width()
      img_height = self.curr_photoimage.height()

      # flag to change the width
      if w_flag == True:

        # case user changed the width from the button
        if ask == True:
          width = tkSimpleDialog.askinteger("Width","Enter a number",parent = self.canvas)

          if width is None:
            return

          # check if the user entered a large number
          if width >img_width or width < 1:
            tkMessageBox.showinfo(title="Warning",message="Width shouldn't be greater than image width or less than 1")
            return

        else:
          # get the width input from the user
          width = self.slider_rec_w.get()

        if w is not 0:
          width = w

        self.rectangle_size[0] = width
        self.slider_rec_w.set(width)

      # flag to change the height
      if h_flag == True:

        # take height from the user
        if ask == True:
          height = tkSimpleDialog.askinteger("Height","Enter a number",parent = self.canvas)

          if height is None:
            return

          # check if the user enteered a large number
          if height >img_height or height < 1:
            tkMessageBox.showinfo(title="Warning",message="Height shouldn't be greater than image height or less than 1")
            return

        else:
          # get the height input from the user
          height = self.slider_rec_h.get()

        if h is not 0:
          height = h

        self.rectangle_size[1] = height
        self.slider_rec_h.set(height)

      if change_size == True:
        # change the coordinates of the polygon
        self.create_rectangle((rect_x_center + self.img_start_x, rect_y_center + self.img_start_y), "blue", self.rectangle_size, new = False)


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

      model_annot_name = os.path.join(self.annotation_folder, self.video_name + ".model")
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
      self.img_start_y = 70
      self.img_id = self.canvas.create_image(self.img_start_x, self.img_start_y, image = self.curr_photoimage, anchor="nw")

      # this data is used to keep track of an item being dragged
      self._drag_data = {"x": 0, "y": 0, "item": None}

      # create a couple movable objects
      self.polygon_id = [0]*self.num_labels

      # rectangle size in x and y (default)
      rec_h = 50
      rec_w = 100
      self.rectangle_size = [rec_w, rec_h]

      self.slider_rec_w.set(rec_w)
      self.slider_rec_h.set(rec_h)


      img_width = self.curr_photoimage.width()
      img_height = self.curr_photoimage.height()

      self.create_rectangle((img_width/2 + self.img_start_x, img_height/2 + self.img_start_y), "blue", self.rectangle_size)

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
      model_annot_name = os.path.join(self.annotation_folder, self.video_name + ".model")
      if os.path.exists(model_annot_name):
        self.load()


    def read_image_from_file(self, image_num = -1):
      # if no parameter is passed set to self.img_num + 1
      if (image_num == -1):
        image_num = self.img_num
      f = os.path.join(self.frames_folder,self.video_name + "_{0}.png".format(image_num+1))
      # check if augment background
      if self.bgcolor.get():
            src = imread(f)
            output, _, _ =\
            augment_bg(src, self.bgcolor_rgb, self.sensitivity,
                       self.custom_bg.get(), self.custom_bg_img)
            #redord to RGB
            self.curr_image_raw = np.dstack((output[:, :, 2],output[:, :, 1],
                                             output[:, :, 0]))
      else:
            self.curr_image_raw = io.imread(f)


    def create_photo_from_raw(self):
      self.curr_photoimage = ImageTk.PhotoImage(image = Image.fromarray(self.curr_image_raw))
      #print "Size of image: width: ", self.curr_photoimage.width(), ", height ", self.curr_photoimage.height()

    def get_coord_rectangle(self):
      ''' Get coordinates of rectangle relative to image '''
      coords_rectangle = self.canvas.coords(self.polygon_id[0])
      coords_rectangle = [long(c) for c in coords_rectangle]
      coords_image = self.canvas.coords(self.img_id)
      coords_image = [long(c) for c in coords_image]
      coords_relative = [coords_rectangle[0]-coords_image[0],coords_rectangle[1]-coords_image[1],coords_rectangle[4]-coords_image[0],coords_rectangle[5]-coords_image[1]]
      return coords_relative

    def change_image(self):

      self.read_image_from_file()
      self.create_photo_from_raw()
      self.canvas.itemconfig(self.img_id, image = self.curr_photoimage)
      self.frame_info_label.winfo_children()[0].config(text = "Frame: {0:0{width}}/{1}".format(self.img_num+1, self.video_num_of_frames, width = 3))
      self.frame_info_label.winfo_children()[1].config(text="Frame: {0}".format(self.video_name + "_{0}.png".format(self.img_num+1)))


      if (self.rectangle_frame_pairs[self.img_num] == 0):
        self.canvas.itemconfig(self.polygon_id[0], outline = "blue")
      else:
        label_index = self.get_label_index_in_list()
        if label_index == -1:
          self.canvas.itemconfig(self.polygon_id[0], outline = "blue")
        else:
          self.canvas.itemconfig(self.polygon_id[0], outline = self.label_colors[self.label_number])

      # get number of annootated frames
      number_of_annotaded_frames = sum([int(i !=0) for i in self.rectangle_frame_pairs])

      self.frame_annot_label.winfo_children()[0].config(text="Annotated frames: {0:0{width}}/{1}".format(number_of_annotaded_frames, len(self.rectangle_frame_pairs), width=3))

    def move_rectangle(self, label_index):
      rel_position = self.rectangle_frame_pairs[self.img_num][label_index]

      rect_x_center = rel_position[0] + self.rectangle_size[0]/2
      rect_y_center = rel_position[1] + self.rectangle_size[1]/2

      curr_position = self.get_coord_rectangle()
      self.canvas.move(self.polygon_id[0], -curr_position[0]+rel_position[0], -curr_position[1]+rel_position[1])


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
      #if background option is used display the image augmented bacground
      self.read_image_from_file(image_num)
      img_width = self.curr_photoimage.width()
      img_height = self.curr_photoimage.height()

      coords_relative = self.rectangle_frame_pairs[image_num]
      coord_x = coords_relative[1]
      coord_y = coords_relative[0]
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
      coord_x = coords_relative[1]
      coord_y = coords_relative[0]
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

      if self.label_number == 0:
        tkMessageBox.showinfo(title = "Warning", message = "Can not extract while label is 0")
        return

      # check if there is annotated model for the current video
      model_annot_name = os.path.join(self.annotation_folder, self.video_name + ".model")
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

      if self.label_number == 0:
        tkMessageBox.showinfo(title = "Warning", message = "Can not segment while label is 0")
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
        tkMessageBox.showinfo(title = "Warning", message = "0 annotations!\nModel not saved!")
        return


      # check if there is already a model
      result = "yes"
      model_annot_name = os.path.join(self.annotation_folder, self.video_name + ".model")
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
      #check if there is a model for the current video frames
      model_annot_name = os.path.join(self.annotation_folder, self.video_name + ".model")
      if not os.path.exists(model_annot_name):
        tkMessageBox.showinfo(title = "Info", message = "No existing annotation model")
        return

      # get the rectangle coordinates for each frame of the loaded model
      previous_frames = self.load_annotations_from_file(model_annot_name)

      self.rectangle_frame_pairs = [0]*len(previous_frames)
      self.rectangle_frame_pairs[0:len(previous_frames)] = previous_frames

      w = 0
      h = 0
      # get number of annootated frames
      number_of_annotaded_frames = sum([int(i !=0) for i in self.rectangle_frame_pairs])

      # check if the current frame is an annotated one
      if (self.rectangle_frame_pairs[self.img_num] is not 0):
        label_index = self.get_label_index_in_list()
        self.update_image_annotated_with_label(label_index)

      self.frame_annot_label.winfo_children()[0].config(text="Annotated frames: {0:0{width}}/{1}".format(number_of_annotaded_frames, len(self.rectangle_frame_pairs), width=3))
      tkMessageBox.showinfo(title = "Info", message = "Annotation model loaded")



    def next_video(self, forward=True):
      save_annot = "no"
      save_annot = tkMessageBox.askquestion("End of video frames", "Save annotations?", icon = "warning")
      if save_annot == "yes":
        self.save()

      else:
        tkMessageBox.showinfo(title = "Info", message = "Annotation model not saved")

      #delete rectangle
      for i in range(0, self.num_labels):
        self.canvas.delete(self.polygon_id[i])

      if forward is True:
        self.video_index+=1
        if self.video_index == self.total_num_of_videos:
          self.video_index = 0
      else:
        self.video_index-=1
        if self.video_index == -1:
          self.video_index = self.total_num_of_videos-1


      self.img_num = 0
      self.load_frames(self.list_of_videos[self.video_index])

      self.change_image()
      # check if label 0 is chosen
      if self.label_number == 0:
        self.all_annotations_mode()

      # check if this image is annotated
      elif (self.rectangle_frame_pairs[self.img_num] is not 0):
        label_index = self.get_label_index_in_list()
        self.update_image_annotated_with_label(label_index)

      self.show_masks()


    def rightKey(self, event):
      self.img_num +=1

      if self.img_num > self.video_num_of_frames-1:

        save_annot = "no"
        save_annot = tkMessageBox.askquestion("End of video frames", "Save annotations?", icon = "warning")
        if save_annot == "yes":
          self.save()

        else:
          tkMessageBox.showinfo(title = "Info", message = "Annotation model not saved")

        #delete rectangle
        for i in range(0, self.num_labels):
          self.canvas.delete(self.polygon_id[i])

        self.video_index+=1
        if self.video_index == self.total_num_of_videos:
          self.video_index = 0

        self.img_num = 0
        self.load_frames(self.list_of_videos[self.video_index])

      self.change_image()
      # check if label 0 is chosen
      if self.label_number == 0:
        self.all_annotations_mode()

      # check if this image is annotated
      elif (self.rectangle_frame_pairs[self.img_num] is not 0):
        label_index = self.get_label_index_in_list()
        self.update_image_annotated_with_label(label_index)

      self.show_masks()


    def leftKey(self, event):
      self.img_num -=1

      if self.img_num < 0:

        save_annot = "no"
        save_annot = tkMessageBox.askquestion("End of video frames", "Save annotations?", icon = "warning")
        if save_annot == "yes":
          self.save()

        else:
          tkMessageBox.showinfo(title = "Info", message = "Annotation model not saved")

        # delete rectangle
        for i in range(0, self.num_labels):
          self.canvas.delete(self.polygon_id[i])

        self.video_index-=1
        if self.video_index == -1:
          self.video_index = self.total_num_of_videos-1

        self.img_num = self.list_number_of_frames[self.video_index] - 1
        self.load_frames(self.list_of_videos[self.video_index])

      self.change_image()
      # check if label 0 is chosen
      if self.label_number == 0:
        self.all_annotations_mode()

      # check if this image is annotated
      elif (self.rectangle_frame_pairs[self.img_num] is not 0):
        label_index = self.get_label_index_in_list()
        self.update_image_annotated_with_label(label_index)

      self.show_masks()

    def get_label_index_in_list(self):
       # number of previous labels for this image
      size_labels = len(self.rectangle_frame_pairs[self.img_num])
      label_exists = False
      # check if there was a previous annotation for the given label number
      for i in range(0, size_labels):
        if self.label_number == self.rectangle_frame_pairs[self.img_num][i][-1]:
          label_exists = True
          return i
      return -1



    def returnKey(self, event):
      # don't annotate if you are in the mask only mode
      if self.show_mask_flag.get() == 1:
        tkMessageBox.showinfo(title="Info",message="Can not annotate in this mode")
        return

      # check if the label is not 0
      if self.label_number == 0:
        tkMessageBox.showinfo(title="Warning",message="Can not annotate while label is 0")
        return

      img_width = self.curr_photoimage.width()
      img_height = self.curr_photoimage.height()
      #save rectangle position
      coords_relative = self.get_coord_rectangle()
      # add the label number at the end of the coords
      coords_relative.append(self.label_number)
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

      # if first time to annotate this frame
      if self.rectangle_frame_pairs[self.img_num] is 0:
        self.rectangle_frame_pairs[self.img_num] = []
        self.rectangle_frame_pairs[self.img_num].append(coords_relative)

      # else check for previous annotations
      else:
        label_index = self.get_label_index_in_list()

        if label_index != -1:
          self.rectangle_frame_pairs[self.img_num][label_index] = coords_relative
        else:
          self.rectangle_frame_pairs[self.img_num].append(coords_relative)

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

        self.canvas.move(self.polygon_id[0], -curr_position[0]+rel_position.left(), -curr_position[1]+rel_position.top())

      self.img_num += 1

      # check if this is the last frame
      if self.img_num >= self.video_num_of_frames:
        # delete polygon
        self.canvas.delete(self.polygon_id[0])
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
      # update according to the label number selected
      if self.rectangle_frame_pairs[self.img_num] is not 0:
        label_index = self.get_label_index_in_list()
        self.update_image_annotated_with_label(label_index)



    def backspaceKey(self, event):

      # don't delete if you are in the mask only mode
      if self.show_mask_flag.get() == 1:
        tkMessageBox.showinfo(title="Info",message="Can not delete in this mode")
        return

      # delete all annotations for all the labels if label is 0
      if self.label_number == 0:
        self.rectangle_frame_pairs[self.img_num] = 0
        for i in range(0, self.num_labels):
          self.canvas.delete(self.polygon_id[i])

      else:
        label_index = self.get_label_index_in_list()
        if label_index != -1:
          del self.rectangle_frame_pairs[self.img_num][label_index]
          if len(self.rectangle_frame_pairs[self.img_num]) == 0:
            self.rectangle_frame_pairs[self.img_num] = 0
          self.canvas.itemconfig(self.polygon_id[0], outline = "blue")

      # get number of annootated frames
      number_of_annotaded_frames = sum([int(i !=0) for i in self.rectangle_frame_pairs])
      self.frame_annot_label.winfo_children()[0].config(text="Annotated frames: {0:0{width}}/{1}".format(number_of_annotaded_frames, len(self.rectangle_frame_pairs), width=3))


    def OnTokenButtonPress(self, event):
        '''Being drag of an object'''
        # record the item and its location
        self._drag_data["item"] = self.canvas.find_closest(event.x, event.y)[0]
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

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

    def OnPickColorCoord(self, event):
        '''Pick the background color coords'''
        #check if clicked on the canvas
        relative_x = event.x - self.img_start_x
        relative_y = event.y - self.img_start_y
        img_width = self.curr_photoimage.width()
        img_height = self.curr_photoimage.height()
        self.bgcolor_rgb = self.curr_image_raw[relative_y, relative_x].tolist()
        bgcolor_hex = '#%02x%02x%02x' % tuple(self.bgcolor_rgb)
        self.bgcolor_canvas.delete("all")
        self.bgcolor_canvas.create_rectangle(0, 0, 40, 20, fill=bgcolor_hex)
        self.sensitivity = 10
        self.canvas.unbind('<Button-1>')
        self.custom_bg_check_box.config(state='normal')
        self.read_image_from_file(self.img_num)
        self.create_photo_from_raw()
        self.canvas.itemconfig(self.img_id, image = self.curr_photoimage)

    def add_custom_bg(self):
        options = {}
        options['defaultextension'] = '.*'
        options['filetypes'] = [('image files', '.*')]
        options['initialdir'] = '.'
        options['initialfile'] = ''
        options['parent'] = self.canvas
        options['title'] = 'Select Background'
        custom_bg_filename = tkFileDialog.askopenfilename(**options)
        self.custom_bg_img = imread(custom_bg_filename)
        if self.custom_bg_img is None:
            tkMessageBox.showerror(title='Bad file',
                                   message='Can not open the selected file')
        else:
            self.read_image_from_file(self.img_num)
            self.create_photo_from_raw()
            self.canvas.itemconfig(self.img_id, image = self.curr_photoimage)

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
  parser.add_argument("-ff", dest = "frames_folder", type = str,
          help = "path to frame folder (pass if no extraction needed)")
  args = parser.parse_args()

  from skimage import io # image read and conversion to array # had to import it here (conflicts with the args passed)
  app = SampleApp()
  app.mainloop()
