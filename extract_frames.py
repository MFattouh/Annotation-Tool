import os
import shutil
import argparse


def extract_frames_from_one_video(video_file, frames_path, fps):

  images_counter = 1
  counter = 1
  # get the video name without the extension and path
  video_file_name = get_video_file_name(video_file)
  # get the name of cut file of the video
  cut_video_file = os.path.dirname(video_file) +  "/" + video_file_name + ".txt"

  # path for video folder containing its frames  according to thhe fps
  frames_video_name_fps = frames_path + video_file_name + "_fps_" + str(fps)


  #check if the video file exists
  if not os.path.exists(video_file):
    print "Error: Video " + video_file_name + " doesn't exist"
    exit(1)

  #check if frames for this video and fps were already created
  if os.path.exists(frames_video_name_fps + "_00001.png"):
    print "Warning: frames already created for " + video_file_name + " with fps: " + str(fps)

  else:
    # extract frames
    print "Extracting frames from video: " + video_file_name + " (fps: " + str(fps) +" )"

    # check if the video cut exists or not
    if not os.path.exists(cut_video_file):
      print("Cut file " + os.path.basename(cut_video_file) + " for video " + video_file_name
        + " doesn't exist,  will extract frames for the whole video")

      array = []
      new_path = frames_path + "model" + str(counter) + "/"

      # create temporary folder so all segments can be extracted there and then copied
      # with the consecutive name  ... , 10034, 10035, ... in the frames_path
      os.system("mkdir " + new_path)

      os.system("ffmpeg -i " + video_file + " -qscale:v 2 -r {} -f image2 '".format(fps) + new_path + "%05d.png'")

      counter = 1
      for f in os.listdir(new_path):
        shutil.copy2(os.path.join(new_path, "{0:05d}.png".format(counter)), (frames_video_name_fps + "_{0:05d}.png".format(images_counter)))
        images_counter += 1
        counter += 1
      os.system("rm -R " + new_path)

    else:
      with open(cut_video_file, "r") as ins:
        array = []
        for line in ins:
          # read segments from file and skip the newlines
          no_newline = line[0:len(line)-1]
          new_path = frames_video_name_fps + "model" + str(counter) + "/"

          # create temporary folder so all segments can be extracted there and then copied
          # with the consecutive name  ... , 10034, 10035, ... in the frames_path
          os.system("mkdir " + new_path)

          # [changeable][optional]
          # 00:00:02 indicates number of seconds to take from starting point
          os.system("ffmpeg -i " + video_file + " -ss " + no_newline +
            " -t 00:00:02 -qscale:v 2 -r {} -f image2 '".format(fps) + new_path + "%d.png'")

          counter = 1
          for f in os.listdir(new_path):
            shutil.copy2(os.path.join(new_path, "{0}.png".format(counter)), (frames_video_name_fps + "_{0}.png".format(images_counter)))
            images_counter += 1
            counter += 1
          os.system("rm -R " + new_path)

    print "Frames extraction done"




def extract_frames_from_videos(videos_folder, video_file_path, frames_path, fps):

  # first check if a video was not passed as an argument
  if video_file_path is None:
    print "Extracting frames from all videos"
    #loop through all the videos in the video folder
    for root, dirs, files in os.walk(videos_folder):
      for file in files:
        # check if it is a video
        if file.endswith(".h264"):
          # set the whole path of the video
          video_file = os.path.join(root, file)
          #extract frames from the video
          extract_frames_from_one_video(video_file, frames_path, fps)
  else:
    # get the video name with without the path (eg video.avi)
    video_file_name_with_ext = os.path.basename(video_file_path)
    # set the whole path of the video
    video_file = os.path.join(videos_folder, video_file_name_with_ext)
    # extract frames from the video
    extract_frames_from_one_video(video_file, frames_path, fps)



# takes the path of the video file
# outputs the name of the video
def get_video_file_name(video_file):
  # get the video name with without the path (eg video.avi)
  video_file_with_extension =  os.path.basename(video_file)

  #remove the extension (splits from the last ".")
  video_file_name = os.path.splitext(video_file_with_extension)[0]

  return video_file_name
