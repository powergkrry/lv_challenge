import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

dir_name = "./test_image/"
images = os.listdir(dir_name)

f, a = plt.subplots(len(images), 4, figsize=(7,7))
title = "sigmaXY_1_10_100"
f.suptitle(title)
for i, imname in enumerate(images):
  showimg = []
  img = cv2.imread(dir_name+imname)
  showimg.append(img)

  dst1 = cv2.GaussianBlur(img,(5,5),1,sigmaY=1)
  dst2 = cv2.GaussianBlur(img,(5,5),10,sigmaY=10)
  dst3 = cv2.GaussianBlur(img,(5,5),100,sigmaY=100)
  showimg.append(dst1)
  showimg.append(dst2)
  showimg.append(dst3)

  for j in range(4):
    a[i][j].imshow(showimg[j], cmap="gray")
    a[i][j].set_xticks(()); a[i][j].set_yticks(());

plt.show()
f.savefig("./result_image/"+"Gaussian_"+title+".png")
