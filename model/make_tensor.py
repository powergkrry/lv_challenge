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
out_dir = "/home/yeonjee/lv_challenge/data/dataset/dataset04_tensor/"
dir_list_iop = ["p/"]
dir_list_traintest = ["train/","test/"]
dir_list_fliprot = ["h/","hv/","o/","rl/","rr/","v/"]

for iop in dir_list_iop:
    os.mkdir(out_dir + iop)
    for traintest in dir_list_traintest:
        os.mkdir(out_dir + iop + traintest)

        img_name_list = os.listdir(img_dir + iop + traintest + "o/original_o/")
        img_name_list.sort()
        img_num = len(img_name_list)
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
            
            for idx in range(img_num):
                tensor_arr = []
                for i_batch, (img,cat) in enumerate(img_dataloader):
                    if(i_batch % img_num == idx):
                        tensor_arr.append(img)
                all_diffusion = torch.cat(tensor_arr,dim=1)
                out_dir_name = out_dir + iop + traintest + fliprot + img_name_list[idx][:3] + ".pt"
                torch.save(all_diffusion, out_dir_name)
                print(out_dir_name)


