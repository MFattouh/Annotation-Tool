import os
import cPickle
#from skimage import io  # imported in the main file
from PIL import Image

#maybe make class
def read_image_from_file(frames_folder, image_num):
      f = os.path.join(frames_folder, "{0}.jpg".format(image_num))
      return io.imread(f)

def sliding_window(image, image_num, patch_position, window_size, save_folder, first, overlap = 0):
      file_pos = os.path.join(save_folder, "../pos.txt")
      file_neg = os.path.join(save_folder, "../neg.txt")
      
      if (first == 0):
        fop = open(file_pos, "w")
        fon = open(file_neg, "w")
      else:
        fop = open(file_pos, "a")
        fon = open(file_neg, "a")  
      
      #print patch_position[0]
      coord_x = patch_position[1]
      coord_y = patch_position[0]
      p = 0
      patch_num = 0
      
      while p < (640/window_size[1])*(480/window_size[0]):
        #print coord_x, coord_y
	Image.fromarray(image[coord_x:coord_x+50,coord_y:coord_y+100,:]).save("{0}/{1}_{2}.jpg".format(save_folder,image_num, patch_num))
	if p == 0:
	  fop.write("patches/{0}_{1}.jpg 1 \n".format(image_num, patch_num))
	else:
	  fon.write("patches/{0}_{1}.jpg 0 \n".format(image_num, patch_num))
	  
	coord_y = coord_y+window_size[1]
	if coord_y >= 640-window_size[1]:
	  coord_x = coord_x+window_size[0]
	  coord_y = 0
	if coord_x >= 480-window_size[0]:
	  break
	
	p = p+1
	patch_num = patch_num + 1
      p = 0
      coord_x = patch_position[1]
      coord_y = patch_position[0]
      #print 'asdsa'
      while p < (640/window_size[1])*(480/window_size[0]):
        
	coord_y = coord_y-window_size[1]
	if coord_y < 0:
	  coord_x = coord_x-window_size[0]
	  coord_y = 640-window_size[1]
	if coord_x < 0:
	  break      
	
	Image.fromarray(image[coord_x:coord_x+50,coord_y:coord_y+100,:]).save("{0}/{1}_{2}.jpg".format(save_folder,image_num,patch_num))
	#print coord_x, coord_y
	fon.write("patches/{0}_{1}.jpg 0 \n".format(image_num, patch_num))

	p = p+1
	patch_num = patch_num + 1
      
      fop.close()
      fon.close()


def load_annotations_from_file(file_name):
  f = file(file_name, 'rb')
  frame_rectangle_pairs = cPickle.load(f)
  #print frame_rectangle_pairs
  f.close()
  return frame_rectangle_pairs

def extract_patches_for_frames(frame_folder_name, annotations, save_folder):
  flag = 0
  for x in range(0, len(annotations)):
    if annotations[x] <> 0:
      image = read_image_from_file(frame_folder_name, x+1)
      sliding_window(image, x+1, annotations[x], (50,100), save_folder, flag, 0)
      flag = 1   
def generate_patches_for_models():
  models_path = os.path.abspath("/home/deeplearning/Documents/paw_tracking_caffe/models/")
  for f in os.listdir(models_path):
   if os.path.isdir(os.path.join(models_path, f)) is True:
     file_name = os.path.join(os.path.join(models_path, f),"annotations.model")
     annot = load_annotations_from_file(file_name)
     frame_folder_name = os.path.join(os.path.join(models_path, f), "frames")
     
     save_folder = os.path.join(os.path.join(models_path, f), "patches")
     extract_patches_for_frames(frame_folder_name, annot, save_folder)
     print "patch numero: ", f

#def load_all_annotations_from_folder(folder_name):
 
if __name__ == '__main__':
 generate_patches_for_models()

