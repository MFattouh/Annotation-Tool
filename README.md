- Annotation tool used for Deep Tracking

- Semi-manual video annotation tool based on correlation tracker.

- Continuing on the work from https://github.com/stefanm91/annotation_tool

The correlation tracker works by combining filters for translation and scaling so it can detect the object at different sizes and different locations. It manages to mark the object in the clear situations but does not perform accurately in all cases.  

Please visit the project [web page](http://www.optophysiology.uni-freiburg.de/Research/Deep-Tracking) for more details.

[1] Shahbaz Khan Danelljan Haeger. Accurate scale estimation for robust visual tracking. In: BMVC (2014).

----------------------------------------------------------------------------------------------------------------------------------------------------------------
PS: Before you use this tool you should have already extracted frames for the videos of interest
----------------------------------------------------------------------------------------------------------------------------------------------------------------

- How to extract the frames for a video:
----------------------------------------

* Usage of main.py can be known by typing "main.py -h" in the command line which has 5 arguments as follows:

 -mf: [path to the main folder that contains a folder named 'videos' (where the videos folder contains the video files and .txt files for the
cuts of the videos, if the cut of a  video is not given then it will extract frames for the whole video)]  if the main folder is not passed then
default is current directory

 -v : [video to extract frames from, if not passed then extract frames from all video files in the 'videos' folder]

 -fps: [frames per second, if not passed default fps =24]

 -of: [Output folder: folder containing the generated folders (patches, masks, frames, annotation). If not given then default is current directory ]

 -ff: [frames folder path: if passed then no frames extraction will take place]

* frames are saved with respect to the video name and the frame number (e.g. if the video name is toys_camera01.avi with fps = 25 then for example frame number 6 will be saved as -> toys_camera01_25_6.png) 
----------------------------------------------------------------------------------------------------------------------------------------------------------------

- How to use the tool:
----------------------
	1- Annotation
	2- Augmentation
	3- Export


1- Annotation:
--------------
	*For each frame in the 'frames' folder do the following: place a rectangle (with adjustable length and height) on the object of interest (eg. a frog in an image) and press enter to annotate this data. 

	*There is also the option that allows the annotation of multiple objects in one frame (up to 5 objects(classes))

	*The annotation data is necessary to later train deep neural networks. (This way the network can be 'told' where the object is (supervised learning) in each training image).

	*After you are finished with annotating the frames of interest in the video then you can save these annotations to a file which will be save in the 'annotation' folder

	*Repeat the above steps for all the videos and then you are done with the annotation part


2- Augmentation:
----------------
	* The annotation step is a must before this step as it reads all the annotation files that are found in the 'annotation' folder.

	* Data augmentation is useful when it comes to training deep neural networks so that the data do not over fit and it can generalize well.

	* From the annotation files of each video, a mask is calculated representing the annotated object with its number (e.g. if a frog in an frame is annotated with a label = 2, then the mask will be 2's in the part where the frog is and 0's elsewhere)

	* In the tool there are three options for your data(frames, masks): rotate, color and scale
	
	* You can then choose which options from these you want to apply to your data and how many times and press "Augment" (e.g rotate = 20; means that each annotated frame and its corresponding mask are rotated randomly 20 times) and then you end up with your original data + the augmented data
	
	* If no option is chosen then it simply produces the frames and their corresponding masks

3- Export:
----------
	* After augmenting the data now you can export it either as images, mat or hdf5 files to be used as supervised learning data for training the network !
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

- Scripts functionality 
-------------------

* There are 5 scripts along with the main script which is responsible for the GUI part and calling functions from other scripts

1- extract_frames.py: responsible for extracting the frames of the videos

2- compute_masks.py: There is an option in the annotation tool "show masks only" that makes use of this script which shows the frame with only the annotated objects visible and the background is white

3- check_hdf5.py: this is only used for debugging to check if the data written to the hdf5 files are correct (adjust the file path u want to read inside this script)

4- augmentation.py: used for augmenting the data as explained above which checks if the user entered numbers for rotation, scale and color 
NOTE VERY IMP: in the script, the order of checking of these options(rotation, scale and color) is important as they data is saved to one big numpy array and then the data in this array is used in the export_data.py and it generates the png ,mat or hdf5 files by checking for these options in the same order they were put in the array.

5- export_data.py: responsible for generating png, mat or hdf5 files for the augmented data. Files produced from this script are saved according to the video name and frame number. (e.g for saving png file for frame number 2 with augmentation option scale = 20 for a video name "toys_camera01.avi" with fps = 25 ---> The frame name will be: toys_camera01_25_2_augment_scale_x.png ;  where x is a number from 1 to 20)









