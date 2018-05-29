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

def cal_test_error(i):

    num_test_img = len(os.listdir(tensor_test_output_dir + "o/"))
    num_test_batch = num_test_img // batch_size
    test_loss_sum = 0
    test_loss_sum_rounded = 0

    print("num_test_img : "+str(num_test_img))

    for fliprot in fliprot_list:
        fliprot_test_input_dir = tensor_test_input_dir + fliprot
        fliprot_test_output_dir = tensor_test_output_dir + fliprot
        test_tensor_list = os.listdir(fliprot_test_input_dir)
        test_tensor_list.sort()

        idx = 0
        for batch in range(num_test_batch):
            #print("idx : "+str(idx))
            x_test = [torch.load(fliprot_test_input_dir + test_tensor_list[idx+k]) for k in range(batch_size)]
            x_test = torch.cat(x_test, dim=0)
            x_test = Variable(x_test).cuda()
            y__test = [torch.load(fliprot_test_output_dir + test_tensor_list[idx+k]) for k in range(batch_size)]
            y__test = torch.cat(y__test, dim=0)
            y__test = Variable(y__test).cuda()

            y_test = generator.forward(x_test)
            y2_test = torch.round(y_test)

            loss_test = recon_loss_func(y_test,y__test)
            #loss_sum += loss.data[0]
            test_loss_sum += (loss_test.item() * batch_size)
            test_loss_rounded = recon_loss_func(y2_test,y__test)
            #loss_sum_rounded += loss_rounded.data[0]
            test_loss_sum_rounded += (test_loss_rounded.item() * batch_size)

            idx += batch_size

        if idx < num_test_img:
            #print("idx : "+str(idx))
            x_test = [torch.load(fliprot_test_input_dir + test_tensor_list[idx+k]) for k in range(num_test_img - idx)]
            x_test = torch.cat(x_test, dim=0)
            x_test = Variable(x_test).cuda()
            y__test = [torch.load(fliprot_test_output_dir + test_tensor_list[idx+k]) for k in range(num_test_img - idx)]
            y__test = torch.cat(y__test, dim=0)
            y__test = Variable(y__test).cuda()

            y_test = generator.forward(x_test)
            y2_test = torch.round(y_test)

            test_loss = recon_loss_func(y_test,y__test)
            #loss_sum += loss.data[0]
            test_loss_sum += (test_loss.item() * batch_size)
            test_loss_rounded = recon_loss_func(y2_test,y__test)
            #loss_sum_rounded += loss_rounded.data[0]
            test_loss_sum_rounded += (test_loss_rounded.item() * batch_size)


    if i == 0:
        v_utils.save_image(y__test.cpu().data,"./result/res_img/label_image.png")
    v_utils.save_image(y2_test.cpu().data,"./result/res_img/gen_image_{:03d}.png".format(i))

    test_error_output = test_loss_sum / (num_test_img * len(fliprot_list))
    test_error_output2 = test_loss_sum_rounded / (num_test_img * len(fliprot_list))

    #tensorboard logging
    logger.scalar_summary('test_loss', test_error_output, i+1)
    logger.scalar_summary('test_loss_rounded', test_error_output2, i+1)

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

# initiate Generator
if args.network == "fusionnet":
	#generator = nn.DataParallel(FusionGenerator(9,1,64),device_ids=[i for i in range(args.num_gpu)]).cuda()
	generator = nn.DataParallel(FusionGenerator(9,1,64),device_ids=None).cuda()
	#generator = nn.DataParallel(FusionGenerator(8,1,64),device_ids=[i for i in range(args.num_gpu)])
	#generator = nn.DataParallel(FusionGenerator(9,1,4),device_ids=[i for i in range(args.num_gpu)])
elif args.network == "unet":
	#generator = nn.DataParallel(UnetGenerator(9,1,64),device_ids=[i for i in range(args.num_gpu)]).cuda()
        generator = nn.DataParallel(UnetGenerator(9,1,64),device_ids=None).cuda()


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
scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, step_size=100, gamma=0.1)


# training
file = open('./result/res_error/{}_{}_{}_{:03d}_loss'.format(args.network,args.batch_size,args.lr,args.epoch), 'w')
logger = Logger('./result/logs')

#tensor_train_input_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/train/"
tensor_train_input_dir = "/home/image/lv_challenge/data/dataset/dataset04_tensor/p/train/"
#tensor_train_output_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/train_output/"
tensor_train_output_dir = "/home/image/lv_challenge/data/dataset/dataset04_tensor/p/train_output/"
#tensor_test_input_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/test/"
tensor_test_input_dir = "/home/image/lv_challenge/data/dataset/dataset04_tensor/p/test/"
#tensor_test_output_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/test_output/"
tensor_test_output_dir = "/home/image/lv_challenge/data/dataset/dataset04_tensor/p/test_output/"
fliprot_list = ["h/","hv/","o/","rl/","rr/","v/"]
num_img = len(os.listdir(tensor_train_input_dir + "o/"))
num_batch = num_img // batch_size

for i in range(epoch):
    loss_sum = 0
    loss_sum_rounded = 0

    for fliprot in fliprot_list:
        fliprot_input_dir = tensor_train_input_dir + fliprot
        fliprot_output_dir = tensor_train_output_dir + fliprot
        tensor_list = os.listdir(fliprot_input_dir)
        tensor_list.sort()

        idx = 0
        for batch in range(num_batch):
            #print("idx : "+str(idx))
            x = [torch.load(fliprot_input_dir + tensor_list[idx+k]) for k in range(batch_size)]
            x = torch.cat(x, dim=0)
            x = Variable(x).cuda()
            y_ = [torch.load(fliprot_output_dir + tensor_list[idx+k]) for k in range(batch_size)]
            y_ = torch.cat(y_, dim=0)
            y_ = Variable(y_).cuda()

            """
            print("x shape : "+str(x.shape))
            print("y_ shape : "+str(y_.shape))
            """ 

            """
        for tensor in tensor_list:
            x = torch.load(fliprot_input_dir + tensor)
            x = Variable(x).cuda()
            y_ = torch.load(fliprot_output_dir + tensor)
            y_ = Variable(y_).cuda()
            """

            y = generator.forward(x)
            y2 = torch.round(y)
            gen_optimizer.zero_grad()

            loss = recon_loss_func(y,y_)
            #loss_sum += loss.data[0]
            loss_sum += (loss.item() * batch_size)
            loss_rounded = recon_loss_func(y2,y_)
            #loss_sum_rounded += loss_rounded.data[0]
            loss_sum_rounded += (loss_rounded.item() * batch_size)

            loss.backward()
            gen_optimizer.step()
            idx += batch_size

            """
            if j == 0:
                a[0][idx].imshow(x.data.cpu().numpy()[0][0].reshape(256,256),cmap='gray')
                a[0][idx].set_xticks(()); a[0][idx].set_yticks(())
                a[1][idx].imshow(y_.data.cpu().numpy().reshape(256,256),cmap='gray')
                a[1][idx].set_xticks(()); a[1][idx].set_yticks(())
        plt.show()

            #TensorBoard logging
            if j == 0 and idx == 0:
                #(1) Log the scalar values
                logger.scalar_summary('loss', loss.data[0], i+1)

                #(2) Log values and gradients of the parameters (histogram)
                for tag, value in generator.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), i+1)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), i+1)

                #(3) Log the images
                info = {
                    'train_images': y2.view(-1, 256, 256).data.cpu().numpy()
                }

                for tag, images in info.items():
                    logger.image_summary(tag, images, epoch+1)
            """

        if idx < num_img:
            #print("idx : "+str(idx))
            x = [torch.load(fliprot_input_dir + tensor_list[idx+k]) for k in range(num_img - idx)]
            x = torch.cat(x, dim=0)
            x = Variable(x).cuda()
            y_ = [torch.load(fliprot_output_dir + tensor_list[idx+k]) for k in range(num_img - idx)]
            y_ = torch.cat(y_, dim=0)
            y_ = Variable(y_).cuda()

            """
            print("x shape : "+str(x.shape))
            print("y_ shape : "+str(y_.shape))
            """
            
            y = generator.forward(x)
            y2 = torch.round(y)
            gen_optimizer.zero_grad()

            loss = recon_loss_func(y,y_)
            #loss_sum += loss.data[0]
            loss_sum += (loss.item() * (num_img-idx))
            loss_rounded = recon_loss_func(y2,y_)
            #loss_sum_rounded += loss_rounded.data[0]
            loss_sum_rounded += (loss_rounded.item() * (num_img-idx))

            loss.backward()
            gen_optimizer.step()

    print(i)
    tr_er = loss_sum / (num_img * len(fliprot_list))
    r_tr_er = loss_sum_rounded / (num_img * len(fliprot_list))
    tst_er, r_tst_er = cal_test_error(i)
    #print("len :",len(iter_output)*len(iter_arr))
    print("train error :",tr_er)
    print("rounded train error :",r_tr_er)
    print("test error :",tst_er)
    print("rounded test error :",r_tst_er)
    print("\n")

    #tensorboard logging
    logger.scalar_summary('loss', tr_er, i+1)
    
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
    torch.save(generator,'./result/trained/{}_{}.pkl'.format(args.network,i))
    scheduler.step()

file.write("train"+"\n"+str(train_error)+"\n"+"rounded train"+"\n"+str(r_train_error)+"\n"+"test"+"\n"+str(test_error)+"\n"+"rounded test"+"\n"+str(r_test_error))
