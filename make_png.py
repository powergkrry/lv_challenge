import numpy as np
import skimage
from skimage import io
from skimage.restoration import denoise_tv_bregman
from skimage.filters import gaussian
from matplotlib import pyplot as plt
import os
import torch
import shutil


def imflip(grayimage, direction):
    if direction == "o":
        flipped = grayimage
    elif direction == 'rl1': # rotate left 1
        flipped = np.rot90(grayimage, 1)
    elif direction == 'rl2': # rotate left 2
        flipped = np.rot90(grayimage, 2)
    elif direction == 'rl3': # rotate left 3
        flipped = np.rot90(grayimage, 3)
    elif direction == "ov": # vertical
        flipped = np.flip(grayimage, 0)
    elif direction == 'rl1v': # rotate left 1 and vertical
        flipped = np.flip(np.rot90(grayimage, 1), 0)
    elif direction == 'rl2v': # rotate left 2 and vertical
        flipped = np.flip(np.rot90(grayimage, 2), 0)
    elif direction == 'rl3v': # rotate left 3 and vertical
        flipped = np.flip(np.rot90(grayimage, 3), 0)
    else:
        return -1

    return flipped


# not use Imagefolder
#  dirname_save1=["ioriginal_TV_png", "ooriginal_TV_png", "poriginal_TV_png"]
#  dirname_save2=["/home/yeonjee/lv_challenge/data/dataset/dataset08/ioriginal_TV_png/",
#      "/home/yeonjee/lv_challenge/data/dataset/dataset08/ooriginal_TV_png/",
#      "/home/yeonjee/lv_challenge/data/dataset/dataset08/poriginal_TV_png/"]
#  dirname_load1=["/home/yeonjee/lv_challenge/data/ioriginal_png/",
#      "/home/yeonjee/lv_challenge/data/ooriginal_png/",
#      "/home/yeonjee/lv_challenge/data/poriginal_png/"]
#  dirname_load2=["/home/yeonjee/lv_challenge/data/icontour_png/",
#      "/home/yeonjee/lv_challenge/data/ocontour_png/",
#      "/home/yeonjee/lv_challenge/data/pcontour_png/"]
#  TV_weight=[4]
#  TV_eps=[0.01]
#  flip=["o", "rl1", "rl2", "rl3", "ov", "rl1v", "rl2v", "rl3v"];
#  
#  os.chdir("/home/yeonjee/lv_challenge/data/dataset/")
#  if not os.path.exists("dataset08"):
#      os.makedirs("dataset08")
#  
#  #for input
#  for i in range(len(dirname_save1)):
#      os.chdir("/home/yeonjee/lv_challenge/data/dataset/dataset08/")
#      if not os.path.exists(dirname_save1[i]):
#          os.makedirs(dirname_save1[i])
#      os.chdir(dirname_save2[i])
#  
#      original_names = os.listdir(dirname_load1[i])
#      original_names.sort()
#  
#      for j, imname in enumerate(original_names):
#          img = io.imread(dirname_load1[i]+imname)
#  
#          for k in TV_weight:
#              for l in TV_eps:
#                  for m in flip:
#                      #mkdir_name = str(k) + "_" + str(l) + "_" + str(m)
#                      mkdir_name = "stack" + "_" + str(m)
#  
#                      if j == 0:
#                          if not os.path.exists(mkdir_name):
#                              os.makedirs(mkdir_name)
#  
#                      os.chdir(dirname_save2[i] + mkdir_name)
#  
#                      dst = denoise_tv_bregman(img, weight=k, eps=l, isotropic=False)
#                      #dst = skimage.exposure.rescale_intensity(dst, out_range=(0, 255))
#                      #dst = dst.astype(np.uint8)
#  
#                      img_arr = np.expand_dims(np.expand_dims(imflip(img.astype('float32')/255.0 ,m), axis=0), axis=0)
#                      dst_arr = np.expand_dims(np.expand_dims(imflip(skimage.exposure.rescale_intensity(img.astype('float32') - dst.astype('float32'), out_range=(0, 255))/255.0, m), axis=0), axis=0)  
#                      tensor_arr = [torch.from_numpy(img_arr.copy()), torch.from_numpy(dst_arr.copy())]
#  
#                      all_diffusion = torch.cat(tensor_arr,dim=1)
#  
#                      out_dir_name = imname[0:3] + ".pt"
#  
#                      torch.save(all_diffusion, out_dir_name)
#  
#                      print("input ", imname)
#                      
#                      #io.imsave("./{:03d}.png".format(j + 1), dst)
#  
#                      os.chdir("../")
#  
#  
#  #for output
#  for i in range(len(dirname_save1)):
#      os.chdir("/home/yeonjee/lv_challenge/data/dataset/dataset08/")
#      if not os.path.exists(dirname_save1[i]):
#          os.makedirs(dirname_save1[i])
#      os.chdir(dirname_save2[i])
#  
#      original_names = os.listdir(dirname_load2[i])
#      original_names.sort()
#  
#      for j, imname in enumerate(original_names):
#          img = io.imread(dirname_load2[i]+imname)
#  
#          for k in TV_weight:
#              for l in TV_eps:
#                  for m in flip:
#                      #mkdir_name = str(k) + "_" + str(l) + "_" + str(m)
#                      mkdir_name = "output" + "_" + str(m)
#  
#                      if j == 0:
#                          if not os.path.exists(mkdir_name):
#                              os.makedirs(mkdir_name)
#  
#                      os.chdir(dirname_save2[i] + mkdir_name)
#  
#                      tensor_arr = torch.from_numpy(np.expand_dims(np.expand_dims(imflip(img.astype('float32')/255.0 ,m), axis=0), axis=0).copy())
#  
#                      out_dir_name = imname[0:3] + ".pt"
#  
#                      torch.save(tensor_arr, out_dir_name)
#  
#                      print("output ", imname)
#                      
#                      #io.imsave("./{:03d}.png".format(j + 1), dst)
#  
#                      os.chdir("../")


# use Imagefolder
dirname_save1=["ioriginal_png"] #, "ooriginal_G_png", "poriginal_G_png"]
dirname_save2=["/home/yeonjee/lv_challenge/data/raw/ioriginal_png/",
    "/home/yeonjee/lv_challenge/data/raw/ooriginal_png/",
    "/home/yeonjee/lv_challenge/data/raw/poriginal_png/"]
dirname_load1=["/home/yeonjee/lv_challenge/data/ioriginal_png/",
    "/home/yeonjee/lv_challenge/data/ooriginal_png/",
    "/home/yeonjee/lv_challenge/data/poriginal_png/"]
dirname_load2=["/home/yeonjee/lv_challenge/data/icontour_png/",
    "/home/yeonjee/lv_challenge/data/ocontour_png/",
    "/home/yeonjee/lv_challenge/data/pcontour_png/"]
#TV_weight=[4, 16]
#TV_eps=[0.01]
empty=[0]
G_sigma=[1, 4]
flip=["o", "rl1", "rl2", "rl3", "ov", "rl1v", "rl2v", "rl3v"]

os.chdir("/home/yeonjee/lv_challenge/data/")

shutil.rmtree('raw')

if not os.path.exists("raw"):
    os.makedirs("raw")

# for input
for i in range(len(dirname_save1)):
    os.chdir("/home/yeonjee/lv_challenge/data/raw/")
    if not os.path.exists(dirname_save1[i]):
        os.makedirs(dirname_save1[i])
    os.chdir(dirname_save2[i])

    original_names = os.listdir(dirname_load1[i])
    original_names.sort()

    for j, imname in enumerate(original_names):
        img = io.imread(dirname_load1[i]+imname)
        img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))
        img = img.astype(np.uint8)

        for k in empty:#TV_weight:#empty:
            for l in empty:#TV_eps:#G_sigma:
                for m in flip:
                    #mkdir_name1 = str(l) + "_" + str(m)
                    mkdir_name2 = "original" + "_" + str(m)
                    #mkdir_name3 = str(k) + "_" + str(l) + "_" + "noise" + "_" + str(m)
                    #mkdir_name3 = str(k) + "_" + str(l) + "_" + str(m)
                    #mkdir_name = "stack" + "_" + str(m)

                    if j == 0:
                        #if not os.path.exists(mkdir_name1):
                        #    os.makedirs(mkdir_name1)
                        if not os.path.exists(mkdir_name2):
                            os.makedirs(mkdir_name2)
                        #if not os.path.exists(mkdir_name3):
                        #    os.makedirs(mkdir_name3)

                    #os.chdir(dirname_save2[i] + mkdir_name1)

                    #dst = gaussian(img, sigma=l)
                    #dst = skimage.exposure.rescale_intensity(dst, out_range=(0, 255))
                    #dst = dst.astype(np.uint8)
                    #dst = imflip(dst, m)

                    #io.imsave("./{:03d}.png".format(j + 1), dst)

                    os.chdir(dirname_save2[i] + mkdir_name2)

                    img = imflip(img, m)

                    io.imsave("./{:03d}.png".format(j + 1), img)

                    #os.chdir(dirname_save2[i] + mkdir_name3)

                    #dst = denoise_tv_bregman(img, weight=k, eps=l, isotropic=False)
                    #dst = skimage.exposure.rescale_intensity(dst, out_range=(0, 255))
                    #dst = dst.astype(np.uint8)
                    #dst = imflip(dst, m)

                    #noise = skimage.exposure.rescale_intensity(img - dst, out_range=(0, 255))
                    #noise = noise.astype(np.uint8)
                    #noise = imflip(noise, m)

                    #io.imsave("./{:03d}.png".format(j + 1), dst)
                    #io.imsave("./{:03d}.png".format(j + 1), noise)


                    #img_arr = np.expand_dims(np.expand_dims(imflip(img.astype('float32')/255.0 ,m), axis=0), axis=0)
                    #dst_arr = np.expand_dims(np.expand_dims(imflip(skimage.exposure.rescale_intensity(img.astype('float32') - dst.astype('float32'), out_range=(0, 255))/255.0, m), axis=0), axis=0)  
                    #tensor_arr = [torch.from_numpy(img_arr.copy()), torch.from_numpy(dst_arr.copy())]

                    #all_diffusion = torch.cat(tensor_arr,dim=1)

                    #out_dir_name = imname[0:3] + ".pt"

                    #torch.save(all_diffusion, out_dir_name)

                    #print("input ", imname)

                    os.chdir("../")


#for output
for i in range(len(dirname_save1)):
    os.chdir("/home/yeonjee/lv_challenge/data/raw/")
    if not os.path.exists(dirname_save1[i]):
        os.makedirs(dirname_save1[i])
    os.chdir(dirname_save2[i])

    original_names = os.listdir(dirname_load2[i])
    original_names.sort()

    for j, imname in enumerate(original_names):
        img = io.imread(dirname_load2[i]+imname)

        for k in empty:
            for l in empty:
                for m in flip:
                    #mkdir_name = str(k) + "_" + str(l) + "_" + str(m)
                    mkdir_name = "output" + "_" + str(m)

                    if j == 0:
                        if not os.path.exists(mkdir_name):
                            os.makedirs(mkdir_name)

                    os.chdir(dirname_save2[i] + mkdir_name)

                    dst = imflip(img, m)

                    #tensor_arr = torch.from_numpy(np.expand_dims(np.expand_dims(imflip(img.astype('float32')/255.0 ,m), axis=0), axis=0).copy())

                    #out_dir_name = imname[0:3] + ".pt"

                    #torch.save(tensor_arr, out_dir_name)

                    #print("output ", imname)
                    
                    io.imsave("./{:03d}.png".format(j + 1), dst)
                    
                    os.chdir("../")

