import cv2
import os
import glob
import pdb
from tqdm import tqdm
import pdb

data_dir='../Kinetics/kinetics-dataset/k400/transfer_train/'
files = glob.glob(data_dir+'*')

count=1
for file in files:
   cap = cv2.VideoCapture(file)
   fps = int(cap.get(cv2.CAP_PROP_FPS))
   frame_skip = 5 # number of frames to skip  
   print(file)
   frame_count = 0
   writen_frames=0
   #pdb.set_trace()
   directoryTrain = '../Kinetics/kinetics-dataset/k400images/train/'+file.split('/')[-1].split('.')[0]
   directoryVal = '../Kinetics/kinetics-dataset/k400images/val/'+file.split('/')[-1].split('.')[0]

   if int(file.split('/')[-1].split('.')[0].split('_')[1][1:]) % 5 == 0:
      directory = directoryVal
   else:
      directory = directoryTrain

   if not os.path.exists(directory):
      os.makedirs(directory)

   length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   for i in tqdm(range(length)):
      #while cap.isOpened():
   
      ret, frame = cap.read()
      if ret:
          frame_count +=1
          if frame_count % (frame_skip+1) == 0:
             count += 1

             cv2.imwrite(directory+"/image"+str(count)+".png", frame)

             writen_frames +=1

      # Break the loop
      #else:
      #    break

   print(f"{writen_frames} frames in {file} have been extracted")
   cap.release()
   cv2.destroyAllWindows()

