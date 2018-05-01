import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
import time
import folder2
import random


def img_to_tensorVariable(dir_path, name_list, num_img_from, num_img_to):
    img_list = []

    for i, name in enumerate(name_list):
        if i + 1 < num_img_from:
            continue

        img = Image.open(os.path.join(dir_path, name))
        img_list.append(list(img.getdata()))

        if i + 1 == num_img_to:
            img_arr = np.asfarray(img_list) / 255.
            break

    img_arr = Variable(torch.Tensor(img_arr).view(-1, 1, 256, 256)).cuda()
    return img_arr


def cal_test_error(test_file_num, BATCH_SIZE):
    test_input_iter = iter(test_loader_input)
    test_output_iter = iter(test_loader_output)
    loss = 0

    if len(test_input_iter) != len(test_output_iter):
        return "different number of picture"

    for i in range(len(test_input_iter)):
        data_input, _ = test_input_iter.next()
        data_output, _ = test_output_iter.next()
        data_input = data_input.type(torch.FloatTensor)
        data_output = data_output.type(torch.FloatTensor)

        x = Variable(data_input.cuda())
        y = Variable(data_output.cuda())

        test_denoise = autoencoder(x)
        loss += loss_func(test_denoise, y).data[0]

    test_error_output = loss / int(test_file_num / BATCH_SIZE)

    return test_error_output


# trans_comp = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0, ), (1, ))])
trans_comp = transforms.ToTensor()

torch.manual_seed(1)  # reproducible
torch.cuda.manual_seed_all(1)  # for multi gpu
# torch.backends.cudnn.enabled = False
np.random.seed(1)
random.seed(1)

EPOCH = 200
BATCH_SIZE = 16
LR1 = 0.001  # learning rate
LR2 = 0.0001
LR3 = 0.00001
N_TEST_IMG = 4
test_error = []

# change : dir_input, dir_test_input, namelist_input,
dir_input = '/home/powergkrry/lv_challenge/poriginal_onlyoriginal_train_png/original/'
dir_output = '/home/powergkrry/lv_challenge/pcontour_train_png/'
dir_test_input = '/home/powergkrry/lv_challenge/poriginal_onlyoriginal_test_png/original/'
dir_test_output = '/home/powergkrry/lv_challenge/pcontour_test_png/'
namelist_input = os.listdir('/home/powergkrry/lv_challenge/poriginal_onlyoriginal_train_png/original/folder/')
namelist_output = os.listdir('/home/powergkrry/lv_challenge/pcontour_train_png/folder/')
namelist_test = os.listdir('/home/powergkrry/lv_challenge/poriginal_test_png/original/folder')
namelist_input.sort()
namelist_output.sort()
namelist_test.sort()
test_file_num = len(namelist_test)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
#            nn.Conv2d(64, 64, 3, stride=1, padding=1),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

#            nn.Linear(256*128*87, 1024),
#            nn.BatchNorm2d(1024),
#            nn.ReLU()
	)

        self.decoder = nn.Sequential(
            nn.MaxUnpool2d(2, stride=2),
#            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ConvTranspose2d(64, 32, 3, padding=1),

            nn.MaxUnpool2d(2, stride=2),
            nn.ConvTranspose2d(32, 16, 3, padding=1),
            nn.ConvTranspose2d(16, 1, 3, padding=1),

#            nn.Sigmoid()
        )
        """
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.r1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.r2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.r3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.r4 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.r5 = nn.ReLU()

        self.conv6 = nn.Conv2d(256, 64, 1, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(64)
        self.r6 = nn.ReLU()
        self.conv7 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.r7 = nn.ReLU()
        self.conv8 = nn.Conv2d(64, 256, 1, stride=1, padding=0)
        self.bn8 = nn.BatchNorm2d(256)
        self.r8 = nn.ReLU()

        #        self.maxpool2 = nn.MaxPool2d(2, stride=2, return_indices=True)

        #        self.linear1 = nn.Linear(50*128*87, 80)
        #        self.bn4 = nn.BatchNorm2d(80)
        #        self.r4 = nn.ReLU()
        #        self.linear2 = nn.Linear(80, 50*128*87)
        #        self.r5 = nn.ReLU()

        #        self.maxunpool2 = nn.MaxUnpool2d(2, stride=2)

        self.deconv8 = nn.ConvTranspose2d(256, 64, 1, padding=0)
        self.bn9 = nn.BatchNorm2d(64)
        self.r9 = nn.ReLU()
        self.deconv7 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(64)
        self.r10 = nn.ReLU()
        self.deconv6 = nn.ConvTranspose2d(64, 256, 1, padding=0)
        self.bn11 = nn.BatchNorm2d(256)
        self.r11 = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(256, 128, 1, padding=0)
        self.bn12 = nn.BatchNorm2d(128)
        self.r12 = nn.ReLU()

        self.maxunpool2 = nn.MaxUnpool2d(2, stride=2)

        self.deconv4 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(128)
        self.r13 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.bn14 = nn.BatchNorm2d(64)
        self.r14 = nn.ReLU()

        self.maxunpool1 = nn.MaxUnpool2d(2, stride=2)

        self.deconv2 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.bn15 = nn.BatchNorm2d(64)
        self.r15 = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(64, 1, 3, padding=1)

    """
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    """

    def forward(self, x):
        out = self.r1(self.bn1(self.conv1(x)))
        out = self.r2(self.bn2(self.conv2(out)))

        size1 = out.size()
        out, indices1 = self.maxpool1(out)

        out = self.r3(self.bn3(self.conv3(out)))
        out = self.r4(self.bn4(self.conv4(out)))

        size2 = out.size()
        out, indices2 = self.maxpool2(out)

        out = self.r5(self.bn5(self.conv5(out)))
        out = self.r6(self.bn6(self.conv6(out)))
        out = self.r7(self.bn7(self.conv7(out)))
        out = self.r8(self.bn8(self.conv8(out)))

#        out = out.view(out.size()[0], -1)
#        out = self.r4(self.bn4(self.linear1(out)))
#        out = self.r5(self.linear2(out))
#        out = out.view(-1, 50, 128, 87)

        out = self.r9(self.bn9(self.deconv8(out)))
        out = self.r10(self.bn10(self.deconv7(out)))
        out = self.r11(self.bn11(self.deconv6(out)))
        out = self.r12(self.bn12(self.deconv5(out)))

        out = self.maxunpool2(out, indices2, size2)

        out = self.r13(self.bn13(self.deconv4(out)))
        out = self.r14(self.bn14(self.deconv3(out)))

        out = self.maxunpool1(out, indices1, size1)

        out = self.r15(self.bn15(self.deconv2(out)))
        out = self.deconv1(out)

        return(out)


print("generating autoencoder")
autoencoder = AutoEncoder()
autoencoder = torch.nn.DataParallel(autoencoder, device_ids=[0, 1])
autoencoder.cuda()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR1, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=14, gamma=0.1)

# autoencoder = torch.load('./pretrained/test_epoch_model.pth.tar')
# optimizer = torch.load('./pretrained/test_epoch_optimizer.pth.tar')
# do i need this?
##autoencoder.load_state_dict(torch.load('./pretrained/kang05_epoch12_model.pth.tar'))
##optimizer.load_state_dict(torch.load('./pretrained/kang05_epoch12_optimizer.pth.tar'))
"""
# original saved file with DataParallel
model_state_dict = torch.load('./pretrained/kang05_epoch1_model.pth.tar')
autoencoder.load_state_dict(model_state_dict)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in model_state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
autoencoder.load_state_dict(new_state_dict)
"""
# check = torch.load('./pretrained/kang04_epoch17.pth.tar')
# autoencoder.load_state_dict(check['model'])
# optimizer.load_state_dict(check['state'])
# autoencoder.eval()

loss_func = nn.MSELoss()

print("loading view_data")
view_data_in = img_to_tensorVariable(
    os.path.join(dir_input, 'folder/'), namelist_input, 1, N_TEST_IMG
)
view_data_out = img_to_tensorVariable(
    os.path.join(dir_output, 'folder/'), namelist_output, 1, N_TEST_IMG
)
print("loading complete")

# start_time = time.time()
print("loading train_loader")
folder_input = folder2.ImageFolder(
    root=dir_input,
    transform=trans_comp,
    loader=folder2.pil_loader
)
train_loader_input = torch.utils.data.DataLoader(
    folder_input, batch_size=BATCH_SIZE, shuffle=False)

folder_output = folder2.ImageFolder(
    root=dir_output,
    transform=trans_comp,
    loader=folder2.pil_loader
)
train_loader_output = torch.utils.data.DataLoader(
    folder_output, batch_size=BATCH_SIZE, shuffle=False)
print("loading complete")

print("loading test_loader")
folder_test_input = folder2.ImageFolder(
    root=dir_test_input,
    transform=trans_comp,
    loader=folder2.pil_loader
)
test_loader_input = torch.utils.data.DataLoader(
    folder_test_input, batch_size=8, shuffle=False)

folder_test_output = folder2.ImageFolder(
    root=dir_test_output,
    transform=trans_comp,
    loader=folder2.pil_loader
)
test_loader_output = torch.utils.data.DataLoader(
    folder_test_output, batch_size=8, shuffle=False)
print("loading complete")

# print("--- %s seconds ---" %(time.time() - start_time))

"""
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5,3))
#plt.ion()

for i in range(N_TEST_IMG):
    a[0][i].imshow(view_data_in.data.cpu().numpy()[i].reshape(512, 350), cmap='gray')
    a[0][i].set_xticks(()) 
    a[0][i].set_yticks(())
    a[1][i].imshow(view_data_out.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
    a[1][i].set_xticks(())
    a[1][i].set_yticks(())
plt.show()
"""
for epoch in range(EPOCH):
    train_input_iter = iter(train_loader_input)
    train_output_iter = iter(train_loader_output)

    if len(train_input_iter) != len(train_output_iter):
        print("different number of picture")
        break

    for step in range(len(train_input_iter)):
        data_input, _ = train_input_iter.next()
        data_output, _ = train_output_iter.next()
        data_input = data_input.type(torch.FloatTensor)
        data_output = data_output.type(torch.FloatTensor)

        x = Variable(data_input.cuda())
        y = Variable(data_output.cuda())
        decoded = autoencoder(x)  # delete encoded,

        loss = loss_func(decoded, y)
        #        print("Epoch :", epoch, "| step :",step,"| train loss: %0.6f" % loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            #            break
            print("Epoch :", epoch, "| step :", step, "| train loss: %0.6f" % loss.data[0])

        """
        if step%100 != 0:
            continue

        decoded_data = autoencoder(view_data_in)
        for i in range(N_TEST_IMG):
            a[2][i].clear()
            a[2][i].imshow(decoded_data.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
            a[2][i].set_xticks(())
            a[2][i].set_yticks(())
        plt.draw()
        plt.pause(0.05)
        """

    epoch_ = epoch + 0 + 1
    save_name_model = './pretrained/test_onlyori_epoch' + str(epoch_) + '_model' + '.pth.tar'
    save_name_optimizer = './pretrained/test_onlyori_epoch' + str(epoch_) + '_optimizer' + '.pth.tar'
    #요고torch.save(autoencoder.state_dict(), save_name_model)
    #    torch.save(autoencoder.module.state_dict(), save_name_model)
    #요고torch.save(optimizer.state_dict(), save_name_optimizer)
    #    torch.save(autoencoder, save_name_model)
    #    scheduler.step()
    print("model saved")

    test_error.append(cal_test_error(test_file_num, 8))
    print("test_error appended")
    print(test_error)

"""
# Plot Test Image
test_input_iter = iter(test_loader_input)
test_output_iter = iter(test_loader_output)

for i in range(len(test_input_iter)):
    data_input, _ = test_input_iter.next()
    data_output, _ = test_output_iter.next()
    data_input = data_input.type(torch.FloatTensor)
    data_output = data_output.type(torch.FloatTensor)

    x = Variable(data_input.cuda())
    y = Variable(data_output.cuda())

    test_denoise = autoencoder(x)

    f, a = plt.subplots(3, N_TEST_IMG, figsize=(5, 3))

    for i in range(N_TEST_IMG):
        a[0][i].imshow(x.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    for i in range(N_TEST_IMG):
        a[1][i].imshow(y.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())

    for i in range(N_TEST_IMG):
        a[2][i].imshow(test_denoise.data.cpu().numpy()[i].reshape(512,350), cmap='gray')
        a[2][i].set_xticks(())
        a[2][i].set_yticks(())
    plt.show()
"""
