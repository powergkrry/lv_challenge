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
from skimage.exposure import rescale_intensity
import cv2


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
	generator = nn.DataParallel(FusionGenerator(2,1,32),device_ids=None).cuda()
	#generator = nn.DataParallel(FusionGenerator(8,1,64),device_ids=[i for i in range(args.num_gpu)])
	#generator = nn.DataParallel(FusionGenerator(9,1,4),device_ids=[i for i in range(args.num_gpu)])
elif args.network == "unet":
	generator = nn.DataParallel(UnetGenerator(9,1,64),device_ids=[i for i in range(args.num_gpu)]).cuda()


# load pretrained model
try:
    generator = torch.load('./result/trained/63_{}_{}.pkl'.format(args.network, '060'))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass


# loss function & optimizer
recon_loss_func = nn.MSELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(),lr=lr)


# training
#tensor_train_input_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/train/"
tensor_train_input_dir = "/home/image/lv_challenge/data/dataset/dataset12_tensor/i/train/"
#tensor_train_output_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/train_output/"
tensor_train_output_dir = "/home/image/lv_challenge/data/dataset/dataset12_tensor/i/train_output/"
#tensor_test_input_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/test/"
tensor_test_input_dir = "/home/image/lv_challenge/data/dataset/dataset12_tensor/i/test/"
#tensor_test_output_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/test_output/"
tensor_test_output_dir = "/home/image/lv_challenge/data/dataset/dataset12_tensor/i/test_output/"
fliprot_list = ["o/", "rl1/", "rl2/", "rl3/", "ov/", "rl1v/", "rl2v/", "rl3v/"]
num_img = len(os.listdir(tensor_test_input_dir + "o/"))
num_batch = num_img // batch_size
num_test_img = 256


dice = 0
f_measure = 0
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

        y_ = [torch.load(fliprot_output_dir + tensor_list[idx+k]) for k in range(batch_size)]
        y_ = torch.cat(y_, dim=0)        
        y_ = Variable(y_).cuda()

        for k in range(batch_size):
            truth = y_.cpu().data.numpy()[k][0]
            truth = truth.astype(np.bool)

            pred = y.cpu().data.numpy()[k][0]
            pred = rescale_intensity(pred, out_range=(0, 255))
            pred = pred.astype(np.uint8)
            ret, predf = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            predf = predf.astype(np.bool)

            #calulate dice
            tp = sum(sum(np.logical_and(truth, predf)))
            truth_sum = sum(sum(np.logical_and(truth, 1)))
            predf_sum = sum(sum(np.logical_and(predf, 1)))
            fp = predf_sum - tp
            fn = truth_sum - tp

            dice += 2. * tp / (2 * tp + fp + fn)

            #calculate f-measure
            if tp == 0:
                print("true positive is zero")
                continue

            precision=float(tp)/(tp+fp)
            recall=float(tp)/(tp+fn)

            f_measure += 2 * ((precision*recall)/(precision+recall))

        folder2.save_image(y.cpu().data,"./result/res_img/{}_{}_output.png".format(tensor_list[batch][0:3],fliprot[0:len(fliprot)-1]))

        idx += batch_size

dice_output = dice / (num_test_img * len(fliprot_list))
f_measure_output = f_measure / (num_test_img * len(fliprot_list))

print(dice_output)
print(f_measure_output)
