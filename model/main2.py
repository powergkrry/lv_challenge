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

class fourflip(object):
    def __call__(self,img):
        result1 = img
        result2 = F.hflip(img)
        result3 = F.vflip(img)
        result4 = F.vflip(F.hflip(img))
    
        return (result1, result2, result3, result4)

def cal_test_error(i):
    iter_arr_test = [[iter(batch_test) for batch_test in img_batch_a_test] for img_batch_a_test in img_batch_arr_test]
    iter_output_test = iter(img_output_batch_arr_test)
    loss_sum = 0
    loss2_sum = 0

    #f,a = plt.subplots(2,len(iter_arr),figsize=(5,5))
    for j in range(len(iter_output_test)):
        tensor_list_test = [[next(it)[0] for it in iter_a_test] for iter_a_test in iter_arr_test]
        tensor_o_test = next(iter_output_test)[0]
        tensor_o_test = tensor_o_test.view(-1,4,256,256)

        tensor_output_test = torch.chunk(tensor_o_test,chunks=4,dim=1)
        for idx in range(len(iter_arr_test)):

            all_diffusion_test = torch.cat(tensor_list_test[idx],dim=1)

            #x_test = Variable(all_diffusion_test).cuda()
            x_test = Variable(all_diffusion_test)
            #y__test = Variable(tensor_output_test[idx]).cuda()
            y__test = Variable(tensor_output_test[idx])
            y_test = generator.forward(x_test)
            ###y_test = torch.round(y_test)
            y2_test = torch.round(y_test)

            loss = recon_loss_func(y_test, y__test)
            loss2 = recon_loss_func(y2_test, y__test)
            loss_sum += loss.data[0]
            loss2_sum += loss2.data[0]
            
            if j == 0 and idx == 0:
                #(1) Log the scalar values
                logger.scalar_summary('test_loss', loss.data[0], i+1)

            """
            if j == 0:
                a[0][idx].imshow(x_test.data.cpu().numpy()[0][0].reshape(256,256),cmap='gray')
                a[0][idx].set_xticks(()); a[0][idx].set_yticks(())
                a[1][idx].imshow(y_test.data.cpu().numpy().reshape(256,256),cmap='gray')
                a[1][idx].set_xticks(()); a[1][idx].set_yticks(())
        
        plt.show() 
        """    

    if i == 0:
        v_utils.save_image(y__test.cpu().data,"./result/res_img/label_image.png")
    v_utils.save_image(y2_test.cpu().data,"./result/res_img/gen_image_{:03d}.png".format(i))


    test_error_output = loss_sum / (len(iter_output_test)*len(iter_arr_test))
    test_error_output2 = loss2_sum / (len(iter_output_test)*len(iter_arr_test))

    #print("test_len :",len(iter_output_test))
    return test_error_output, test_error_output2


parser = argparse.ArgumentParser()
parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
parser.add_argument("--batch_size",type=int,default=4,help="batch size")
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

flip = ['_o','_h','_v','hv']
# input pipeline
img_dir = "/home/yeonjee/lv_challenge/data/dataset/dataset01/p/train/"
# 디렉토리를 4개로 분리시키는 건 어떨까?
dir_list = [[
    dir_name for dir_name in os.listdir(img_dir) if f == dir_name[-2:]
] for f in flip]
for lst in dir_list: lst.sort()

img_data_arr = [[
    folder2.ImageFolder(
        root=img_dir+dirname, 
        transform = transforms.ToTensor()
    ) for dirname in dir_l
] for dir_l in dir_list]

img_batch_arr = [[
    data.DataLoader(
        img_data, 
        batch_size=batch_size,num_workers=2
    ) for img_data in img_data_a
] for img_data_a in img_data_arr]

img_output_arr = folder2.ImageFolder(
    root="/home/yeonjee/lv_challenge/data/dataset/dataset01/p/train/output",
    transform = transforms.Compose([
        fourflip(),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.ToTensor()(crop) for crop in crops
        ]))
    ]))

img_output_batch_arr = data.DataLoader(img_output_arr, batch_size=batch_size,num_workers=2)


img_dir_test = "/home/yeonjee/lv_challenge/data/dataset/dataset01/p/test/"

dir_list_test = [[
    dir_name for dir_name in os.listdir(img_dir_test) if f == dir_name[-2:]
] for f in flip]
for lst in dir_list_test : lst.sort()

img_data_arr_test = [[
    folder2.ImageFolder(
        root=img_dir_test+dirname_test, 
        transform = transforms.ToTensor()
    ) for dirname_test in dir_l_test
] for dir_l_test in dir_list_test]

img_batch_arr_test = [[
    data.DataLoader(
        img_data_test, 
        batch_size=batch_size,
        num_workers=2
    ) for img_data_test in img_data_a_test
] for img_data_a_test in img_data_arr_test]

img_output_arr_test = folder2.ImageFolder(
    root="/home/yeonjee/lv_challenge/data/dataset/dataset01/p/test/output",
    transform = transforms.Compose([
        fourflip(),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.ToTensor()(crop) for crop in crops
        ]))
    ]))

img_output_batch_arr_test = data.DataLoader(img_output_arr_test, batch_size=batch_size,num_workers=2)


# initiate Generator
if args.network == "fusionnet":
	generator = nn.DataParallel(FusionGenerator(8,1,64),device_ids=[i for i in range(args.num_gpu)]).cuda()
elif args.network == "unet":
	generator = nn.DataParallel(UnetGenerator(8,1,64),device_ids=[i for i in range(args.num_gpu)]).cuda()


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
file = open('./result/res_error/{}_{}_{}_{:03d}_loss'.format(args.network,args.batch_size,args.lr,args.epoch), 'w')
logger = Logger('./result/logs')

for i in range(epoch):
    iter_arr = [[iter(batch) for batch in img_batch_a] for img_batch_a in img_batch_arr]
    iter_output = iter(img_output_batch_arr)

    #iter_arr_test = [iter(batch_test) for batch_test in img_batch_arr_test]
    #iter_output_test = iter(img_output_batch_arr_test)
    loss_sum = 0
    loss_sum2 = 0

    #f,a = plt.subplots(2,len(iter_arr),figsize=(5,5))
    for j in range(len(iter_output)):
        tensor_list = [[next(it)[0] for it in iter_a] for iter_a in iter_arr]
        tensor_o = next(iter_output)[0]
        tensor_o = tensor_o.view(-1,4,256,256)
        
        tensor_output = torch.chunk(tensor_o,chunks=4, dim=1)
        for idx in range(len(iter_arr)):
            all_diffusion = torch.cat(tensor_list[idx],dim=1)
            #x = Variable(all_diffusion).cuda()
            x = Variable(all_diffusion)
            #y_ = Variable(tensor_output[idx]).cuda()
            y_ = Variable(tensor_output[idx])
            y = generator.forward(x)
            ###y = torch.round(y)
            y2 = torch.round(y)
            gen_optimizer.zero_grad()

            loss = recon_loss_func(y,y_)
            loss_sum += loss.data[0]
            loss2 = recon_loss_func(y2,y_)
            loss_sum2 += loss2.data[0]
        
            loss.backward()
            gen_optimizer.step()
            
            """
            if j == 0:
                a[0][idx].imshow(x.data.cpu().numpy()[0][0].reshape(256,256),cmap='gray')
                a[0][idx].set_xticks(()); a[0][idx].set_yticks(())
                a[1][idx].imshow(y_.data.cpu().numpy().reshape(256,256),cmap='gray')
                a[1][idx].set_xticks(()); a[1][idx].set_yticks(())
        plt.show()
        """

            #TensorBoard logging
            if j == 0 and idx == 0:
                #(1) Log the scalar values
                logger.scalar_summary('loss', loss.data[0], i+1)

                #(2) Log values and gradients of the parameters (histogram)
                for tag, value in generator.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), i+1)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), i+1)

                """
                #(3) Log the images
                info = {
                    'train_images': y2.view(-1, 256, 256).data.cpu().numpy()
                }

                for tag, images in info.items():
                    logger.image_summary(tag, images, epoch+1)
                """

    print(i)
    tr_er = loss_sum/(len(iter_output)*len(iter_arr))
    r_tr_er = loss_sum2/(len(iter_output)*len(iter_arr))
    tst_er, r_tst_er = cal_test_error(i)
    #print("len :",len(iter_output)*len(iter_arr))
    print("train error :",tr_er)
    print("rounded train error :",r_tr_er)
    print("test error :",tst_er)
    print("rounded test error :",r_tst_er)
    print("\n")
    train_error.append(tr_er)
    r_train_error.append(r_tr_er)
    test_error.append(tst_er)
    r_test_error.append(r_tst_er)

    #chunk, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = torch.chunk(x, chunks=33, dim=1)
    #plt.imshow(chunk.data.cpu().numpy().reshape(256,256), cmap='gray')
    #plt.show()
    #v_utils.save_image(chunk.cpu().data,"./result/original_image_{}.png".format(i))
    #v_utils.save_image(y_.cpu().data,"./result/label_image_{}.png".format(i))
    #v_utils.save_image(y.cpu().data,"./result2/res_img/gen_image_{:03d}.png".format(i))
    #torch.save(generator,'./trained/{}_{}.pkl'.format(args.network,i)) 

file.write("train"+"\n"+str(train_error)+"\n"+"rounded train"+"\n"+str(r_train_error)+"\n"+"test"+"\n"+str(test_error)+"\n"+"rounded test"+"\n"+str(r_test_error))
