# Semantic Segmentation
# Code by GunhoChoi

from FusionNet import * 
from UNet import *
import numpy as np
import matplotlib.pyplot as plt
import argparse
import folder2
from torch.utils.data.sampler import SequentialSampler
import os
from logger import Logger
import torchvision.transforms.functional as F


parser = argparse.ArgumentParser()
parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
parser.add_argument("--batch_size",type=int,default=1,help="batch size")
parser.add_argument("--lr",type=float,default=0.001,help="learning rate")
parser.add_argument("--epoch",type=int,default=300,help="epoch")

parser.add_argument("--num_gpu",type=int,default=2,help="number of gpus")
args = parser.parse_args()


# hyperparameters
batch_size = args.batch_size
lr = args.lr
epoch = args.epoch
img_size = 256
train_error = []
r_train_error =[]
test_error = []
r_test_error = []

# initiate Generator
if args.network == "fusionnet":
	#generator = nn.DataParallel(FusionGenerator(9,1,64),device_ids=[i for i in range(args.num_gpu)]).cuda()
	generator = nn.DataParallel(FusionGenerator(1,1,64),device_ids=None).cuda()
	#generator = nn.DataParallel(FusionGenerator(8,1,64),device_ids=[i for i in range(args.num_gpu)])
	#generator = nn.DataParallel(FusionGenerator(9,1,4),device_ids=[i for i in range(args.num_gpu)])
elif args.network == "unet":
	generator = nn.DataParallel(UnetGenerator(9,1,64),device_ids=[i for i in range(args.num_gpu)]).cuda()


# load pretrained model
try:
    generator = torch.load('./result/trained/54_{}_{}.pkl'.format(args.network, '059'))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass


# loss function & optimizer
recon_loss_func = nn.MSELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(),lr=lr)


# training
img_dir_train = "/home/powergkrry/lv_challenge/data/dataset/dataset06/i/train/"
img_dir_train_out = "/home/powergkrry/lv_challenge/data/dataset/dataset06/i/train_output/"
img_dir_test = "/home/powergkrry/lv_challenge/data/dataset/dataset06/i/test/"
img_dir_test_out = "/home/powergkrry/lv_challenge/data/dataset/dataset06/i/test_output/"
dir_list = os.listdir(img_dir_train)
dir_list.sort()
img_list = os.listdir(img_dir_test_out + "o/output_o/")
img_list.sort()

for flip in dir_list:
    img_data_arr_test = folder2.ImageFolder(root=img_dir_test + flip, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    img_batch_arr_test = data.DataLoader(img_data_arr_test, batch_size=batch_size, num_workers=2)

    img_output_arr_test = folder2.ImageFolder(root=img_dir_test_out + flip, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    img_output_batch_arr_test = data.DataLoader(img_output_arr_test, batch_size=batch_size, num_workers=2)

    iter_arr_test = iter(img_batch_arr_test)
    iter_output_test = iter(img_output_batch_arr_test)

    for j in range(len(iter_output_test)):
        tensor_list = next(iter_arr_test)[0]
        tensor_output = next(iter_output_test)[0]

        x = Variable(tensor_list).cuda()
        y_ = Variable(tensor_output).cuda()
        y = generator.forward(x)

        folder2.save_image(y.cpu().data,"./result/res_img/{}_{}_output.png".format(img_list[j][0:3],flip))
