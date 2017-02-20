# Annotation Tool used for Deep Tracking

Semi-manual video annotation tool based on correlation tracker.
The correlation tracker works by combining filters for translation and scaling so it can detect the object at different sizes and
different locations. It manages to mark the object in the clear situations but does not perform accurately in all cases.

Please visit the project [web page](http://www.optophysiology.uni-freiburg.de/Research/Deep-Tracking) for more details.

[1] Shahbaz Khan Danelljan Haeger. Accurate scale estimation for robust visual tracking. In: BMVC (2014).

# HowTo use the tool
Annotation of the data can be done by positioning a rectangle around the object to indicate the objects's location. 
The annotation data is necessary to later train deep neural networks. (This way the network can be 'told' where the object is (supervised learning) in each training image).


# Work in Progress / Mohamed (2017)
some modifications to the extract_frames.py and to the main.py. and compute_mask.py.

1- Usage of the main.py can be known if you type in the terminal ( python main.py -h )

2- Usage : main.py -mf -v -fps -of

 -mf: [path to the main folder that contains a folder named 'videos' (where the videos folder contains the video files and .txt files for the
cuts of the videos, if the cut of a  video is not given then it will extract frames for the whole video)]  if the main folder is not passed then
default is current directory

 -v : [video to extract frames from, if not passed then extract frames from all video files in the 'videos' folder]

 -fps: [frames per second, if not passed default fps =24]

 -of: [Output folder: folder containing the generated folders (patches, masks, frames ,etc). If not given then default is current directory ]

3- you don't need to run the extract_frames.py  or compute_masks.py (they are called from the main.py)

4- frames folder generated is something like this ---> (output_folder/frames/name_of_video/fps_number/frames_of_the_video )

5- I modified the GUI where I added a button to choose the DIRECTORY of the video frames to be annotated (e.g. output_folder/frames/name_of_video/fps_number/ )

6- I also added a button to create a mask for the annotated frames if a model exists and checks if a mask is already created for this model

6- Annotations are saved in output_folder/annotations/video_name_fps_number.model

7-If there is an annotation model for the chosen video frames then it is automatically loaded after you load the frame directory from step 5

8- Before saving the annotations I check for consistency of number of frames annotated with the number of frames of the video and I also check if there is already a saved model

I attached the three scripts and you can try the program and give me a feedback (maybe the code in the main.py needs to be cleaned but I will do it later as I adjusted quite a few lines)

------------------------------------------
Still to do:
 - segmentation folder
 - patches positive and patches negative folders

