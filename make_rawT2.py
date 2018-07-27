import folder2
from torch.utils import data
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# input pipeline
img_dir = "/home/yeonjee/lv_challenge/data/raw/ioriginal_png/"
out_dir = "/home/yeonjee/lv_challenge/data/rawT2/"

input_dir = os.listdir(img_dir)
input_dir.sort()
img_name_list = os.listdir(img_dir + "output_o/")
img_name_list.sort()


if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(out_dir+"ioriginal_png/"):
    os.makedirs(out_dir+"ioriginal_png/")
for dir_name in input_dir:
    os.makedirs(out_dir+"ioriginal_png/"+dir_name)


def return_mean_std(tensor):
    mean_t = tensor.mean()
    std_t = tensor.std()

    return transforms.Normalize([mean_t], [std_t])


#input
img_folder = folder2.ImageFolder(
    root = img_dir,
    transform = transforms.Compose([
        #transforms.CenterCrop(128),
        transforms.ToTensor(),
        #transforms.Lambda(lambda tensor: transforms.Normalize(str([x.mean()]) + ", " + str([x.std()]) for x in tensor)),
        transforms.Lambda(lambda tensor:
            save_t = tensor,
            mean_t = save_t.mean(),
            std_t = save_t.std(),
            transforms.Normalize([mean_t], [std_t])),
    ])
)

img_dataloader = data.DataLoader(
    img_folder,
    batch_size = 1,
    #num_workers = 2,
)

for i, (img_tensor, label) in enumerate(img_dataloader):
    out_dir_name = out_dir + "ioriginal_png/" + input_dir[label[0]] + "/" + img_name_list[i % len(img_name_list)][:3] + ".pt"
    torch.save(img_tensor, out_dir_name)
    print(out_dir_name)


"""
#output
img_folder = folder2.ImageFolder(
    root = img_dir,
    transform = transforms.Compose([
        #transforms.CenterCrop(128),
        transforms.ToTensor(),
    ])
)

img_dataloader = data.DataLoader(
    img_folder,
    batch_size = 1,
    #num_workers = 2,
)

for i, (img_tensor, label) in enumerate(img_dataloader):
    out_dir_name = out_dir + "ioriginal_png/" + input_dir[label[0]] + "/" + img_name_list[i % len(img_name_list)][:3] + ".pt"
    torch.save(img_tensor, out_dir_name)
    print(out_dir_name)
"""
