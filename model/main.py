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


def cal_test_error(i):
    iter_arr_test = [iter(batch_test) for batch_test in img_batch_arr_test]
    iter_output_test = iter(img_output_batch_arr_test)
    loss = 0

    for j in range(len(iter_output_test)):
        tensor_list_test = [next(it)[0] for it in iter_arr_test]
        tensor_output_test = next(iter_output_test)[0]

        all_diffusion_test = torch.cat(tensor_list_test,dim=1)

        x_test = Variable(all_diffusion_test).cuda()
        y__test = Variable(tensor_output_test).cuda()
        y_test = generator.forward(x_test)
        ###y_test = torch.round(y_test)

        loss += recon_loss_func(y_test, y__test).data[0]

    test_error_output = loss / len(iter_output_test)

    if i == 0:
        v_utils.save_image(y__test.cpu().data,"./result/label_image.png")
    v_utils.save_image(y_test.cpu().data,"./result/gen_image_{}.png".format(i))

    return test_error_output


parser = argparse.ArgumentParser()
parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
parser.add_argument("--batch_size",type=int,default=1,help="batch size")
parser.add_argument("--lr",type=float,default=0.01,help="learning rate")
parser.add_argument("--epoch",type=int,default=10,help="epoch")

parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
args = parser.parse_args()


# hyperparameters
batch_size = args.batch_size
lr = args.lr
epoch = args.epoch
img_size = 256
train_error = []
test_error = []


# input pipeline
img_dir = "/home/powergkrry/lv_challenge/data/dataset/dataset02/p/train/"
dir_list = os.listdir(img_dir)
dir_list.remove('output')

img_data_arr = [folder2.ImageFolder(root=img_dir+dirname, transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            ])) for dirname in dir_list]

img_batch_arr = [data.DataLoader(img_data, batch_size=batch_size,num_workers=2) for img_data in img_data_arr]

img_output_arr = folder2.ImageFolder(root="/home/powergkrry/lv_challenge/data/dataset/dataset02/p/train/output", transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            ]))

img_output_batch_arr = data.DataLoader(img_output_arr, batch_size=batch_size,num_workers=2)


img_dir_test = "/home/powergkrry/lv_challenge/data/dataset/dataset02/p/test/"
dir_list_test = os.listdir(img_dir_test)
dir_list_test.remove('output')

img_data_arr_test = [folder2.ImageFolder(root=img_dir_test+dirname_test, transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            ])) for dirname_test in dir_list_test]

img_batch_arr_test = [data.DataLoader(img_data_test, batch_size=batch_size,num_workers=2) for img_data_test in img_data_arr_test]

img_output_arr_test = folder2.ImageFolder(root="/home/powergkrry/lv_challenge/data/dataset/dataset02/p/test/output", transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            ]))

img_output_batch_arr_test = data.DataLoader(img_output_arr_test, batch_size=batch_size,num_workers=2)


# initiate Generator
if args.network == "fusionnet":
	generator = nn.DataParallel(FusionGenerator(33,1,64),device_ids=[i for i in range(args.num_gpu)]).cuda()
elif args.network == "unet":
	generator = nn.DataParallel(UnetGenerator(3,3,64),device_ids=[i for i in range(args.num_gpu)]).cuda()


"""
# load pretrained model
try:
    generator = torch.load('./model/{}.pkl'.format(args.network))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass
"""
print("\n--------model not restored--------\n")


# loss function & optimizer
recon_loss_func = nn.MSELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(),lr=lr)


# training
file = open('./{}_{}_{}_{}_loss'.format(args.network,args.batch_size,args.lr,args.epoch), 'w')

for i in range(epoch):
    iter_arr = [iter(batch) for batch in img_batch_arr]
    iter_output = iter(img_output_batch_arr)

    iter_arr_test = [iter(batch_test) for batch_test in img_batch_arr_test]
    iter_output_test = iter(img_output_batch_arr_test)
    loss_sum = 0

    for j in range(len(iter_output)):
        tensor_list = [next(it)[0] for it in iter_arr]
        tensor_output = next(iter_output)[0]

        all_diffusion = torch.cat(tensor_list,dim=1)
        
        gen_optimizer.zero_grad()

        x = Variable(all_diffusion).cuda()
        y_ = Variable(tensor_output).cuda()
        y = generator.forward(x)
        ###y = torch.round(y)

        loss = recon_loss_func(y,y_)
        loss_sum += loss.data[0]
        loss.backward()
        gen_optimizer.step()

    print(i)
    print(loss_sum/len(iter_output))
    print(cal_test_error(i))
    print("\n")
    train_error.append(loss_sum/len(iter_output))
    test_error.append(cal_test_error(i))

    #chunk, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = torch.chunk(x, chunks=33, dim=1)
    #plt.imshow(chunk.data.cpu().numpy().reshape(256,256), cmap='gray')
    #plt.show()
    #v_utils.save_image(chunk.cpu().data,"./result/original_image_{}.png".format(i))
    #v_utils.save_image(y_.cpu().data,"./result/label_image_{}.png".format(i))
    #v_utils.save_image(y.cpu().data,"./result/gen_image_{}.png".format(i))
    #torch.save(generator,'./trained/{}_{}.pkl'.format(args.network,i)) 

file.write("train"+"\n"+str(train_error)+"\n"+"test"+"\n"+str(test_error))
