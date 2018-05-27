import folder2
from torch.utils import data
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

num_diffuse = 9
flip = ['_o','_h','_v','hv']
# input pipeline
img_dir = "/home/yeonjee/lv_challenge/data/dataset/dataset04/"
out_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/"
dir_list_iop = ["p/"]
dir_list_traintest = ["train_output/","test_output/"]
dir_list_fliprot = ["h/","hv/","o/","rl/","rr/","v/"]
"""
dir_list = [[
    dir_name for dir_name in os.listdir(img_dir) if f == dir_name[-2:]
] for f in flip]
for lst in dir_list: lst.sort()
"""

for iop in dir_list_iop:
    #os.mkdir(out_dir + iop)
    for traintest in dir_list_traintest:
        os.mkdir(out_dir + iop + traintest)

        img_name_list = os.listdir(img_dir + iop + traintest + "o/output_o/")
        img_name_list.sort()
        img_num = len(img_name_list)
        print(img_num)
        for fliprot in dir_list_fliprot:
            os.mkdir(out_dir + iop + traintest + fliprot)

            img_folder = folder2.ImageFolder(
                root = img_dir + iop + traintest + fliprot,
                transform = transforms.ToTensor(),
            )
            img_dataloader = data.DataLoader(
                img_folder,
                batch_size = 1,
                num_workers = 2,
            )

            for i_batch, (img,cat) in enumerate(img_dataloader):
                out_dir_name = out_dir + iop + traintest + fliprot + img_name_list[i_batch][:3] + ".pt"
                torch.save(img, out_dir_name)
                print(out_dir_name)
            
           

