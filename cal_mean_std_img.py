import numpy as np
import skimage
from skimage import io
import os

dirname_load="/home/yeonjee/lv_challenge/data/raw/ioriginal_png/0.1_noise_o/"

original_names = os.listdir(dirname_load)

mean_overall = 0
img_stack = []

for imname in original_names:
    img = io.imread(dirname_load+imname)

    mean = img.mean()

    mean_overall += mean
    img_stack = np.append(img_stack, img)

print("mean_overall :", mean_overall/len(original_names)/256)
print("std_overall :", img_stack.std()/256)
