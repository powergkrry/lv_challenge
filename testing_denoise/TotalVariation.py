import numpy as np
import cv2
from skimage import io
from skimage.restoration import denoise_tv_bregman
from matplotlib import pyplot as plt
import os

dir_name = "./testing_image/"
images = os.listdir(dir_name)

f, a = plt.subplots(len(images), 4, figsize=(7,7))
#var = [0.1, 1, 10]
#var = [0.01, 0.001, 0.0001]
var = [True, False, True]
title = "eps_"+str(var[0])+"_"+str(var[1])+"_"+str(var[2])
f.suptitle(title)
for i, imname in enumerate(images):
  showimg = []
  #img = cv2.imread(dir_name+imname)
  #img, img_header = load(dir_name+imname)
  img = io.imread(dir_name+imname)
  showimg.append(img)

  dst1 = denoise_tv_bregman(img, weight=1, max_iter=100, eps=0.001, isotropic=var[0])
  dst2 = denoise_tv_bregman(img, weight=1, max_iter=100, eps=0.001, isotropic=var[1])
  dst3 = denoise_tv_bregman(img, weight=1, max_iter=100, eps=0.001, isotropic=var[2])
  showimg.append(dst1)
  showimg.append(dst2)
  showimg.append(dst3)

  for j in range(4):
    a[i][j].imshow(showimg[j], cmap="gray")
    a[i][j].set_xticks(()); a[i][j].set_yticks(());

plt.show()
f.savefig("./result/"+"TotalVariation_"+title+".png")
