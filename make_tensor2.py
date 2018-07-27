import folder2
from torch.utils import data
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# input pipeline
img_dir = "/home/yeonjee/lv_challenge/data/dataset/dataset49/"
out_dir = "/home/yeonjee/lv_challenge/data/dataset/dataset49_tensor/"
dir_list_iop = ["i/"]
dir_list_traintest = ["train/","test/"]
dir_list_fliprot = ["o/", "rl1/", "rl2/", "rl3/", "ov/", "rl1v/", "rl2v/", "rl3v/"]

os.mkdir(out_dir)
os.mkdir(out_dir + dir_list_iop[0])


#input
for iop in dir_list_iop:
    #os.mkdir(out_dir + iop)
    for traintest in dir_list_traintest:
        os.mkdir(out_dir + iop + traintest)

        img_name_list = os.listdir(img_dir + iop + traintest + "o/original_o/")
        #img_name_list = os.listdir(img_dir + iop + traintest + "o/1_o/")
        #img_name_list = os.listdir(img_dir + iop + traintest + "o/0.1_o/")

        img_name_list.sort()
        img_num = len(img_name_list)

        for fliprot in dir_list_fliprot:
            os.mkdir(out_dir + iop + traintest + fliprot)

            channel_list = os.listdir(img_dir + iop + traintest + fliprot)

            for tensor in img_name_list:
                tensor_arr = []

                for channel in channel_list:
                    tensor_arr.append(torch.load(img_dir + iop + traintest + fliprot + channel + "/" + tensor))
                
                all_diffusion = torch.cat(tensor_arr, dim = 1)
                out_dir_name = out_dir + iop + traintest + fliprot + tensor
                torch.save(all_diffusion, out_dir_name)
                print(out_dir_name)


#output
dir_list_traintest = ["train_output/","test_output/"]

for iop in dir_list_iop:
    #os.mkdir(out_dir + iop)
    for traintest in dir_list_traintest:
        os.mkdir(out_dir + iop + traintest)

        img_name_list = os.listdir(img_dir + iop + traintest + "o/output_o/")

        img_name_list.sort()
        img_num = len(img_name_list)

        for fliprot in dir_list_fliprot:
            os.mkdir(out_dir + iop + traintest + fliprot)

            channel_list = os.listdir(img_dir + iop + traintest + fliprot)

            for tensor in img_name_list:
                tensor_arr = []

                for channel in channel_list:
                    all_diffusion = torch.load(img_dir + iop + traintest + fliprot + channel + "/" + tensor)

                out_dir_name = out_dir + iop + traintest + fliprot + tensor
                torch.save(all_diffusion, out_dir_name)
                print(out_dir_name)
