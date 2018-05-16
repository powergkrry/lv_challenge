import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

dir_name = "./testing_image/"
images = os.listdir(dir_name)

f, a = plt.subplots(3, 4, figsize=(7,7))
var = [50,100,200]
title = "d"
f.suptitle(title)
for i, imname in enumerate(images):
  showimg = []
  img = cv2.imread(dir_name+imname)
  showimg.append(img)

  for j in var:
    dst = cv2.bilateralFilter(img)
    showimg.append(dst)

  for j in range(len(var)+1):
    if j!=0:
      a[j//4][j%4].set_title(var[j-1])
    a[j//4][j%4].imshow(showimg[j], cmap="gray")
    a[j//4][j%4].set_xticks(()); a[j//4][j%4].set_yticks(());

  plt.show()
  f.savefig("./result/find_interval/"+imname+"Bilateral_"+title+".png")
