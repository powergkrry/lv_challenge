# Semantic Segmentation
# Code by GunhoChoi

from FusionNet import * 
from UNet import *
import numpy as np
import argparse
import folder2
from torch.utils.data.sampler import SequentialSampler
import os
from logger import Logger
import torchvision.transforms.functional as F
from skimage.exposure import rescale_intensity
import cv2
from scheduler_learning_rate import *
import time
from datetime import timedelta
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


def cal_test_error(i):

    num_test_img = len(os.listdir(tensor_test_output_dir + "o/"))
    num_test_batch = num_test_img // batch_size
    test_loss_sum = 0
    test_loss_sum_rounded = 0
    dice = 0
    f_measure = 0


    ##print("num_test_img : "+str(num_test_img))

    for fliprot in fliprot_list:
        fliprot_test_input_dir = tensor_test_input_dir + fliprot
        fliprot_test_output_dir = tensor_test_output_dir + fliprot
        test_tensor_list = os.listdir(fliprot_test_input_dir)
        test_tensor_list.sort()

        idx = 0
        for batch in range(num_test_batch):
            x_test = [torch.load(fliprot_test_input_dir + test_tensor_list[idx+k]) for k in range(batch_size)]

            """
            x_test_stack = []
            y__test_stack = []

            for k in range(batch_size)
                x_test_one_img = torch.load(fliprot_test_input_dir + test_tensor_list[idx+k])]
                y__test_one_img = torch.load(fliprot_test_output_dir + test_tensor_list[idx+k])

                x_test_one_img.cpu().data.numpy()

                x_test_stack += x_test_one_img
                y__test_stack += y__test_one_img
            """

            x_test = torch.cat(x_test, dim=0)
            x_test = Variable(x_test).cuda()
            y__test = [torch.load(fliprot_test_output_dir + test_tensor_list[idx+k]) for k in range(batch_size)]
            y__test = torch.cat(y__test, dim=0)
            y__test = Variable(y__test).cuda()

            y_test = generator.forward(x_test)


            #calculate dice and f_measure
            for k in range(batch_size):
                truth = y__test.cpu().data.numpy()[k][0]
                truth = truth.astype(np.bool)

                pred = y_test.cpu().data.numpy()[k][0]
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
                    #print("true positive is zero")
                    continue

                precision=float(tp)/(tp+fp)
                recall=float(tp)/(tp+fn)
    
                f_measure += 2 * ((precision*recall)/(precision+recall))

            #y2_test = (y_test >= threshold).float() * 1

            test_loss = recon_loss_func(y_test,y__test)
            test_loss_sum += (test_loss.item() * batch_size)
            #test_loss_rounded = recon_loss_func(y2_test,y__test)
            #test_loss_sum_rounded += (test_loss_rounded.item() * batch_size)

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
            #y2_test = (y_test >= threshold).float() * 1

            test_loss = recon_loss_func(y_test,y__test)
            test_loss_sum += (test_loss.item() * batch_size)
            #test_loss_rounded = recon_loss_func(y2_test,y__test)
            #test_loss_sum_rounded += (test_loss_rounded.item() * batch_size)

            #calculate dice and f_measure
            for k in range(num_test_img - idx):
                truth = y__test.cpu().data.numpy()[k][0]
                truth = truth.astype(np.bool)

                pred = y_test.cpu().data.numpy()[k][0]
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


    if i == 0:
        folder2.save_image(y__test.cpu().data,"./result/res_img/label_image.png")
    folder2.save_image(y_test.cpu().data,"./result/res_img/gen_image_{:03d}.png".format(i))

    test_error_output = test_loss_sum / (num_test_img * len(fliprot_list))
    #test_error_output2 = test_loss_sum_rounded / (num_test_img * len(fliprot_list))
    dice_output = dice / (num_test_img * len(fliprot_list))
    f_measure_output = f_measure / (num_test_img * len(fliprot_list))

    #tensorboard logging
    logger.scalar_summary('test_loss', test_error_output, i+1)
    #logger.scalar_summary('test_loss_rounded', test_error_output2, i+1)

    #print("test_len :",len(iter_output_test))
    return test_error_output, dice_output, f_measure_output#, test_error_output2


parser = argparse.ArgumentParser()
parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
parser.add_argument("--batch_size",type=int,default=4,help="batch size")
parser.add_argument("--optimizer",type=str,default="adam",help="choose between adam & sgd")
parser.add_argument("--lr_initial",type=float,default=0.001,help="initial learning rate")
parser.add_argument("--lr_final",type=float,default=0.001,help="final learning rate")
parser.add_argument("--lr_slope",type=int,default=10,help="learning rate slope")
parser.add_argument("--lamda",type=float,default=0,help="weight decay")
parser.add_argument("--epoch",type=int,default=101,help="epoch")
parser.add_argument("--num_gpu",type=int,default=2,help="number of gpus")
parser.add_argument("--number", type=int,default=999, help="th")
parser.add_argument("--dataset", type=int,default=999,help="number of dataset")
parser.add_argument("--input_layer", type=int,default=0,help="number of input layer")
args = parser.parse_args()


# hyperparameters
batch_size = args.batch_size
lr_initial = args.lr_initial
lr_final = args.lr_final
lr_slope = args.lr_slope
lamda = args.lamda
epoch = args.epoch
#threshold = Variable(torch.Tensor([args.threshold])).cuda()
img_size = 256
train_error = []
#r_train_error =[]
test_error = []
#r_test_error = []
dice_stack = []
f_measure_stack = []

# initiate Generator
torch.cuda.manual_seed(1)
if args.network == "fusionnet":
	#generator = nn.DataParallel(FusionGenerator(9,1,64),device_ids=[i for i in range(args.num_gpu)]).cuda()
	generator = nn.DataParallel(FusionGenerator(args.input_layer,1,32),device_ids=None).cuda()
	#generator = nn.DataParallel(FusionGenerator(8,1,64),device_ids=[i for i in range(args.num_gpu)])
	#generator = nn.DataParallel(FusionGenerator(9,1,4),device_ids=[i for i in range(args.num_gpu)])
elif args.network == "unet":
	#generator = nn.DataParallel(UnetGenerator(9,1,64),device_ids=[i for i in range(args.num_gpu)]).cuda()
        generator = nn.DataParallel(UnetGenerator(args.input_layer,1,32),device_ids=None).cuda()


"""
# load pretrained model
try:
    generator = torch.load('./result/trained/55_{}_{}.pkl'.format(args.network,'099'))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass
"""
print("\n--------model not restored--------\n")


# loss function & optimizer
recon_loss_func = nn.MSELoss()

if args.optimizer == "adam":
    optimizer = torch.optim.Adam(generator.parameters(), lr = lr_initial, weight_decay = lamda)
else:
    optimizer = torch.optim.SGD(generator.parameters(), lr = lr_initial, momentum = 0.9, nesterov = True)

##scheduler = scheduler_learning_rate_sigmoid(optimizer, lr_initial, lr_final, epoch, alpha = lr_slope)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=math.sqrt(10)/10.0)


# training
os.mkdir('./result/{:03d}'.format(args.number))
os.mkdir('./result/{:03d}/trained'.format(args.number))
os.mkdir('./result/{:03d}/result_img'.format(args.number))

file = open('./result/{:03d}/data'.format(args.number), 'w')
logger = Logger('./result/logs')

#tensor_train_input_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/train/"
tensor_train_input_dir = "/home/powergkrry/lv_challenge/data/dataset/dataset"+str(args.dataset)+"_tensor/i/train/"
#tensor_train_output_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/train_output/"
tensor_train_output_dir = "/home/powergkrry/lv_challenge/data/dataset/dataset"+str(args.dataset)+"_tensor/i/train_output/"
#tensor_test_input_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/test/"
tensor_test_input_dir = "/home/powergkrry/lv_challenge/data/dataset/dataset"+str(args.dataset)+"_tensor/i/test/"
#tensor_test_output_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/test_output/"
tensor_test_output_dir = "/home/powergkrry/lv_challenge/data/dataset/dataset"+str(args.dataset)+"_tensor/i/test_output/"
fliprot_list = ["o/", "rl1/", "rl2/", "rl3/", "ov/", "rl1v/", "rl2v/", "rl3v/"]
num_img = len(os.listdir(tensor_train_input_dir + "o/"))
num_batch = num_img // batch_size


time_start  = time.time()
for i in range(epoch):
    scheduler.step()

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
            #y2 = (y >= threshold).float() * 1
            optimizer.zero_grad()

            loss = recon_loss_func(y,y_)
            loss_sum += (loss.item() * batch_size)
            #loss_rounded = recon_loss_func(y2,y_)
            #loss_sum_rounded += (loss_rounded.item() * batch_size)

            loss.backward()
            optimizer.step()
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
$
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
            #y2 = (y >= threshold).float() * 1
            optimizer.zero_grad()

            loss = recon_loss_func(y,y_)
            loss_sum += (loss.item() * (num_img-idx))
            #loss_rounded = recon_loss_func(y2,y_)
            #loss_sum_rounded += (loss_rounded.item() * (num_img-idx))

            loss.backward()
            optimizer.step()

    print(i)
    tr_er = loss_sum / (num_img * len(fliprot_list))
    #r_tr_er = loss_sum_rounded / (num_img * len(fliprot_list))
    tst_er, dice_st, f_measure_st = cal_test_error(i)
    #print("len :",len(iter_output)*len(iter_arr))
    ##print("train error :",tr_er)
    #print("rounded train error :",r_tr_er)
    ##print("test error :",tst_er)
    #print("rounded test error :",r_tst_er)
    print("dice :",dice_st)
    ##print("f_measure :",f_measure_st)
    ##print("\n")

    #tensorboard logging
    logger.scalar_summary('loss', tr_er, i+1)
    
    train_error.append(tr_er)
    #r_train_error.append(r_tr_er)

    test_error.append(tst_er)
    #r_test_error.append(r_tst_er)

    dice_stack.append(dice_st)

    f_measure_stack.append(f_measure_st)

    torch.save(generator,'./result/{:03d}/trained/{}_{:03d}.pkl'.format(args.number,args.network,i))
    #scheduler.step()


file.write("train"+"\n"+str(train_error)+"\n"+"test"+"\n"+str(test_error)+"\n"+"dice"+"\n"+str(dice_stack)+"\n"+"f_measure"+"\n"+str(f_measure_stack))


#make test output img
num_img = len(os.listdir(tensor_test_input_dir + "o/"))
batch_size = 1
num_batch = num_img // batch_size
num_test_img = 256

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

        folder2.save_image(y.cpu().data,"./result/{:03d}/result_img/{}_{}_output.png".format(args.number,tensor_list[batch][0:3],fliprot[0:len(fliprot)-1]))

        idx += batch_size


#print the result
print("Number :", args.number)
print("Min_test_error :", min(test_error))
print("Min_test_error_index :", test_error.index(min(test_error)))
print("Max_Dice :", max(dice_stack))
print("Max_Dice_index :", dice_stack.index(max(dice_stack)))
print("Max_F_measure :", max(f_measure_stack))
print("Max_F_measure_index :", f_measure_stack.index(max(f_measure_stack)))
time_elapsed = time.time() - time_start
print("execution time :",str(timedelta(seconds=time_elapsed)))


#plot error curve
plt.plot(train_error, label = 'train')
plt.plot(test_error, label = 'test')
plt.legend()
plt.ylim(0, 0.02)
#plt.axhline(y=0.005, color='r')
plt.savefig('./result/{:03d}/error.png'.format(args.number))
plt.close()

#plot similarity curve
plt.plot(dice_stack, label = 'dice')
plt.plot(f_measure_stack, label = 'f_measure')
plt.legend()
plt.ylim(0, 1)
#plt.axhline(y=0.005, color='r')
plt.savefig('./result/{:03d}/similarity.png'.format(args.number))
plt.close()

