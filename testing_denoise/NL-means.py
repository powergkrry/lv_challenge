import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

dir_name = "./testing_image/"
images = os.listdir(dir_name)

title = "patchsize"
var = [1,2,4,8,16,32,64,128,256,512,1024]
#var = [2,4,8,16,32,64,128,256,512,1024,2048] #h
for i, imname in enumerate(images):
  f, a = plt.subplots(3, 4, figsize=(7,7))
  f.suptitle(title)
  showimg = []
  img = cv2.imread(dir_name+imname)
  showimg.append(img)

  for j in var:
    dst = cv2.fastNlMeansDenoising(img,None,10,j,7)
    showimg.append(dst)

  for j in range(len(var)+1):
    if j!=0:
      a[j//4][j%4].set_title(var[j-1])
    a[j//4][j%4].imshow(showimg[j], cmap="gray")
    a[j//4][j%4].set_xticks(()); a[j//4][j%4].set_yticks(());
      
  plt.show()
  #f.savefig("./result/find_interval/"+imname+"NLmeans_"+title+".png")
