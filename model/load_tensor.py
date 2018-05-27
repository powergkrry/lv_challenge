import torch
import os
import matplotlib.pyplot as plt

img_dir = "/home/yeonjee/lv_challenge/data/dataset_tensor/dataset/p/train_output/v/"
tensor_list = os.listdir(img_dir)
tensor_list.sort()

print(len(tensor_list))
print(tensor_list)
for i in range(len(tensor_list)):
    tens = torch.load(img_dir + tensor_list[i])
    plt.imshow(tens.numpy().reshape(256,256),cmap='gray')
    plt.show()
    """
    a1, _,_,_,_,_,_,_,_= torch.chunk(tens, chunks = 9, dim = 1)
    if i == 2:
        plt.imshow(a1.numpy().reshape(256,256),cmap = 'gray')
        plt.show()
    """

"""
img_num = len(os.listdir(img_dir+dir_list[0][0]+"/folder"))
for j in range(img_num):
    for idx in range(4):
        tens = torch.load("tensor_{:03}_{}".format(j,idx))
        print(tens)
"""
