import numpy as np
from skimage import io
from skimage.restoration import denoise_tv_bregman
from matplotlib import pyplot as plt
import os

dir_name = "/home/yeonjee/lv_challenge/testing_denoise/testing_image/"
images = os.listdir(dir_name)

f, a = plt.subplots(len(images), 9, figsize=(7,7))
#var = [2, 4, 8, 16, 32]
#var = [1, 0.01, 0.001, 0.0001, 0.000001]
#var = [True, False, True, False, True]i
var1 = [0.1, 1, 2, 4]
var2 = [0.04, 0.01]

#title = "weight_"+str(var[0])+"_"+str(var[1])+"_"+str(var[2])+"_"+str(var[3])+"_"+str(var[4])
title = "test"
f.suptitle(title)

for i, imname in enumerate(images):
  showimg = []
  #img = cv2.imread(dir_name+imname)
  #img, img_header = load(dir_name+imname)
  img = io.imread(dir_name+imname)
  print(img.shape)
  print(type(img))
  showimg.append(img)

  #dst1 = denoise_tv_bregman(img, weight=var[0], max_iter=100, isotropic=False)
  #dst2 = denoise_tv_bregman(img, weight=var[1], max_iter=100, isotropic=False)
  #dst3 = denoise_tv_bregman(img, weight=var[2], max_iter=100, isotropic=False)
  #dst4 = denoise_tv_bregman(img, weight=var[3], max_iter=100, isotropic=False)
  #dst5 = denoise_tv_bregman(img, weight=var[4], max_iter=100, isotropic=False)

  #dst1 = denoise_tv_bregman(img, weight=1, eps=var[0], max_iter=500, isotropic=False)
  #dst2 = denoise_tv_bregman(img, weight=1, eps=var[1], max_iter=500, isotropic=False)
  #dst3 = denoise_tv_bregman(img, weight=1, eps=var[2], max_iter=500, isotropic=False)
  #dst4 = denoise_tv_bregman(img, weight=1, eps=var[3], max_iter=500, isotropic=False)
  #dst5 = denoise_tv_bregman(img, weight=1, eps=var[4], max_iter=500, isotropic=False)

  #dst1 = denoise_tv_bregman(img, weight=10, isotropic=var[0])
  #dst2 = denoise_tv_bregman(img, weight=10, isotropic=var[1])
  #dst3 = denoise_tv_bregman(img, weight=10, isotropic=var[2])
  #dst4 = denoise_tv_bregman(img, weight=10, isotropic=var[3])
  #dst5 = denoise_tv_bregman(img, weight=10, isotropic=var[4])

  dst1 = denoise_tv_bregman(img, weight=var1[0], eps=var2[0], isotropic=False)
  dst2 = denoise_tv_bregman(img, weight=var1[1], eps=var2[0], isotropic=False)
  dst3 = denoise_tv_bregman(img, weight=var1[2], eps=var2[0], isotropic=False)
  dst4 = denoise_tv_bregman(img, weight=var1[3], eps=var2[0], isotropic=False)
  dst5 = denoise_tv_bregman(img, weight=var1[0], eps=var2[1], isotropic=False)
  dst6 = denoise_tv_bregman(img, weight=var1[1], eps=var2[1], isotropic=False)
  dst7 = denoise_tv_bregman(img, weight=var1[2], eps=var2[1], isotropic=False)
  dst8 = denoise_tv_bregman(img, weight=var1[3], eps=var2[1], isotropic=False)

  showimg.append(dst1)
  showimg.append(dst2)
  showimg.append(dst3)
  showimg.append(dst4)
  showimg.append(dst5)
  showimg.append(dst6)
  showimg.append(dst7)
  showimg.append(dst8)

  for j in range(9):
    a[i][j].imshow(showimg[j], cmap="gray")
    a[i][j].set_xticks(()); a[i][j].set_yticks(());

plt.show()
f.savefig("./result/find_parameter/"+"TotalVariation_"+title+".png")
