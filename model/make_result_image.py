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
	generator = nn.DataParallel(FusionGenerator(9,1,64),device_ids=None).cuda()
	#generator = nn.DataParallel(FusionGenerator(8,1,64),device_ids=[i for i in range(args.num_gpu)])
	#generator = nn.DataParallel(FusionGenerator(9,1,4),device_ids=[i for i in range(args.num_gpu)])
elif args.network == "unet":
	generator = nn.DataParallel(UnetGenerator(9,1,64),device_ids=[i for i in range(args.num_gpu)]).cuda()


# load pretrained model
try:
    generator = torch.load('./result/trained/{}_{}.pkl'.format(args.network, 299))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass


# loss function & optimizer
recon_loss_func = nn.MSELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(),lr=lr)


# training
#tensor_train_input_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/train/"
tensor_train_input_dir = "/home/image/lv_challenge/data/dataset_tensor/dataset/p/train/"
#tensor_train_output_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/train_output/"
tensor_train_output_dir = "/home/image/lv_challenge/data/dataset_tensor/dataset/p/train_output/"
#tensor_test_input_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/test/"
tensor_test_input_dir = "/home/image/lv_challenge/data/dataset_tensor/dataset/p/test/"
#tensor_test_output_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/test_output/"
tensor_test_output_dir = "/home/image/lv_challenge/data/dataset_tensor/dataset/p/test_output/"
fliprot_list = ["h/","hv/","o/","rl/","rr/","v/"]
num_img = len(os.listdir(tensor_test_input_dir + "o/"))
num_batch = num_img // batch_size

for fliprot in fliprot_list:
    fliprot_input_dir = tensor_test_input_dir + fliprot
    fliprot_output_dir = tensor_test_output_dir + fliprot
    tensor_list = os.listdir(fliprot_input_dir)
    tensor_list.sort()

    idx = 0
    for batch in range(num_batch):
        x = [torch.load(fliprot_input_dir + tensor_list[idx+k]) for k in range(batch_size)]
        x = torch.cat(x, dim=0)
        x = Variable(x).cuda()

        y = generator.forward(x)
        y2 = torch.round(y)

        v_utils.save_image(y2.cpu().data,"./result/res_img/{}_{}_output.png".format(tensor_list[batch][0:3],fliprot[0:len(fliprot)-1]))

        idx += batch_size
