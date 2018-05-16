import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion
from medpy.io import load
from matplotlib import pyplot as plt 
import os

dir_name = "./testing_image/"
images = os.listdir(dir_name)

f, a = plt.subplots(3, 4, figsize=(7,7))
var = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#var = [(8,8),(64,64),(512,512)]
#var = [2, 1, 2]
title = "niter"
f.suptitle(title)
for i, imname in enumerate(images):
  showimg = []
  img, img_header = load(dir_name+imname)
  showimg.append(img)

  img2 = img.astype('float32') 
  img2 *= (255.0/img.max())

  for j in var:
    dst = anisotropic_diffusion(img2,niter=j)
    showimg.append(dst)

  for j in range(len(var)+1):
    if j!=0:
      a[j//4][j%4].set_title(var[j-1])
    a[j//4][j%4].imshow(showimg[j], cmap="gray")
    a[j//4][j%4].set_xticks(()); a[j//4][j%4].set_yticks(());

  plt.show()
  f.savefig("./result/find_interval/"+imname+"_Anisotropic_"+title+".png")
