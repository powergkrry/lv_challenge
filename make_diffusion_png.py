##making png with different diffusion levels
import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion
from medpy.io import load
from medpy.io import save
from PIL import Image
import matplotlib.pyplot as plt
import os

#directory where diffused images will be saved
check = ['/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu3_png', 
'/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu6_png', 
'/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu9_png', 
'/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu12_png']

#make directory
for check_ in check:
    if not os.path.exists(check_):
        os.makedirs(check_)

#directory where original images are saved
where = os.listdir("/hoem04/powergkrry/lv_challenge/data/poriginal_png/")

for location in where:
    path = "/hoem04/powergkrry/lv_challenge/data/poriginal_png/" + location
    img, image_header = load(path)

    img_filtered3 = anisotropic_diffusion(img, niter=3) #kappa = 20?
    img_filtered6 = anisotropic_diffusion(img, niter=6)
    img_filtered9 = anisotropic_diffusion(img, niter=9)
    img_filtered12 = anisotropic_diffusion(img, niter=12)

    #don't need to worry about flipping
    #plt.imshow(img_filtered3, cmap = 'gray')
    #plt.show()
    #plt.imshow(img_filtered6, cmap = 'gray')
    #plt.show()
    #plt.imshow(img_filtered9, cmap = 'gray')
    #plt.show()

    print(type(img_filtered3))

    N_TEST_IMG = 2 # code should be changed if the # is 1
    f, a = plt.subplots(3, N_TEST_IMG, figsize=(5,3))

    for i in range(N_TEST_IMG):
        a[0][i].imshow(img_filtered3, cmap = 'gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())
        a[1][i].imshow(img_filtered6, cmap = 'gray')
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())
        a[2][i].imshow(img_filtered9, cmap = 'gray')
        a[2][i].set_xticks(())
        a[2][i].set_yticks(())
    plt.show()
    break
"""
    img_filtered3 = img_filtered3.astype('uint8')
    img_filtered6 = img_filtered6.astype('uint8')
    img_filtered9 = img_filtered9.astype('uint8')
    img_filtered12 = img_filtered12.astype('uint8')

    path3 = '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu3_png/' + location[:17] + '-diffu3.png'
    path6 = '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu6_png/' + location[:17] + '-diffu6.png'
    path9 = '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu9_png/' + location[:17] + '-diffu9.png'
    path12 = '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu12_png/' + location[:17] + '-diffu12.png'

    save(img_filtered3, path3, image_header)
    save(img_filtered6, path6, image_header)
    save(img_filtered9, path9, image_header)
    save(img_filtered12, path12, image_header)
"""
