import torch
import os

flip = ['_o','_h','_v','hv']
img_dir = "/home/yeonjee/lv_challenge/data/dataset/dataset01/p/train/"
dir_list = [[
    dir_name for dir_name in os.listdir(img_dir) if f == dir_name[-2:]
] for f in flip]
img_num = len(os.listdir(img_dir+dir_list[0][0]+"/folder"))
for j in range(img_num):
    for idx in range(4):
        tens = torch.load("tensor_{:03}_{}".format(j,idx))
        print(tens)
