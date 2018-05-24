import folder2
from torch.utils import data
import os
import torch
from torchvision import transforms

flip = ['_o','_h','_v','hv']
# input pipeline
img_dir = "/home/yeonjee/lv_challenge/data/dataset/dataset01/p/train/"
# 디렉토리를 4개로 분리시키는 건 어떨까?
dir_list = [[
    dir_name for dir_name in os.listdir(img_dir) if f == dir_name[-2:]
] for f in flip]
for lst in dir_list: lst.sort()

img_num = len(os.listdir(img_dir+dir_list[0][0]+"/folder"))
print(img_num)

img_data_arr = [[
    folder2.ImageFolder(
        root=img_dir+dirname,
        transform = transforms.ToTensor()
    ) for dirname in dir_l
] for dir_l in dir_list]

img_batch_arr = [[
    data.DataLoader(
        img_data,
        batch_size=1,num_workers=2
    ) for img_data in img_data_a
] for img_data_a in img_data_arr]

iter_arr = [[iter(batch) for batch in img_batch_a] for img_batch_a in img_batch_arr]

for j in range(img_num):
    tensor_list = [[next(it)[0] for it in iter_a] for iter_a in iter_arr]
    for idx in range(len(iter_arr)):
        all_diffusion = torch.cat(tensor_list[idx], dim=1)
        torch.save(all_diffusion, "tensor_save/tensor_{:03d}_{}".format(j,idx))
