import folder2
from torch.utils import data
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# input pipeline
img_dir = "/home/yeonjee/lv_challenge/data/raw/ioriginal_png/"
out_dir = "/home/yeonjee/lv_challenge/data/rawT/"

input_dir = os.listdir(img_dir)
input_dir.sort()
#img_name_list = os.listdir(img_dir + "original_o/")
img_name_list = os.listdir(img_dir + "output_o/")
img_name_list.sort()


if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(out_dir+"ioriginal_png/"):
    os.makedirs(out_dir+"ioriginal_png/")
for dir_name in input_dir:
    os.makedirs(out_dir+"ioriginal_png/"+dir_name)


normalize_input = [0.5071, 0.0873, 0.2484, 0.3079, 0.4993, 0.0637, 0.2502, 0.3005, 0.2448, 0.2045, 0.2481, 0.3008, 0.248, 0.2901, 0.2329, 0.1562, 0.2479, 0.2719, 0.1964, 0.0985, 0.2473, 0.2438, 0.2481, 0.3109]
#normalize = transforms.Lambda(lambda j: print(normalize_input[j % len(img_name_list)]))
#normalize = transforms.Normalize([normalize_input[22]], [normalize_input[23]])
normalize = transforms.Normalize([0.5], [0.5])


"""
#input
img_folder = folder2.ImageFolder(
    root = img_dir,
    transform = transforms.Compose([
        transforms.CenterCrop(192),
        transforms.ToTensor(),
        normalize,
        #transforms.FiveCrop(64),
        #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        #transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
    ])
)

img_dataloader = data.DataLoader(
    img_folder,
    batch_size = 1,
    #num_workers = 2,
)

for i, (img_tensor, label) in enumerate(img_dataloader):
    out_dir_name = out_dir + "ioriginal_png/" + input_dir[label[0]] + "/" + img_name_list[i % len(img_name_list)][:3] + ".pt"
    torch.save(img_tensor.view(-1, 1, 192, 192), out_dir_name)
    print(out_dir_name)


"""
#output
img_folder = folder2.ImageFolder(
    root = img_dir,
    transform = transforms.Compose([
        transforms.CenterCrop(192),
        transforms.ToTensor(),
        #transforms.FiveCrop(64),
        #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])
)

img_dataloader = data.DataLoader(
    img_folder,
    batch_size = 1,
    #num_workers = 2,
)

for i, (img_tensor, label) in enumerate(img_dataloader):
    out_dir_name = out_dir + "ioriginal_png/" + input_dir[label[0]] + "/" + img_name_list[i % len(img_name_list)][:3] + ".pt"
    torch.save(img_tensor.view(-1, 1, 192, 192), out_dir_name)
    print(out_dir_name)

