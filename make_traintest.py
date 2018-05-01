import os
import random
from PIL import Image

def save_image(directory, sample, subdirectory = None):
    if directory[0:8] == 'pcontour': #pcontour
        if directory[-9:-4] == 'train': #train
            files = os.listdir(os.path.join('/hoem04/powergkrry/lv_challenge/data/pcontour_png'))
            files.sort()

            i = 0
            for name in sample:
                if i < 32:
                    i+=1
                else:
                    img = Image.open(os.path.join('/hoem04/powergkrry/lv_challenge/data/pcontour_png', files[name]))
                    img.save(os.path.join(os.getcwd(), files[name]))
        else: #test
            files = os.listdir(os.path.join('/hoem04/powergkrry/lv_challenge/data/pcontour_png'))
            files.sort()

            i = 0
            for name in sample:
                if i < 32:
                    img = Image.open(os.path.join('/hoem04/powergkrry/lv_challenge/data/pcontour_png', files[name]))
                    img.save(os.path.join(os.getcwd(), files[name]))
                    i+=1
                else:
                    break
    else: #poriginal
        if subdirectory == None: #onlyoriginal
            if directory[-9:-4] == 'train':  # train
                files = os.listdir(os.path.join('/hoem04/powergkrry/lv_challenge/data/poriginal_png'))
                files.sort()

                i = 0
                for name in sample:
                    if i < 32:
                        i += 1
                    else:
                        img = Image.open(os.path.join('/hoem04/powergkrry/lv_challenge/data/poriginal_png', files[name]))
                        img.save(os.path.join(os.getcwd(), files[name]))
            else:  # test
                files = os.listdir(os.path.join('/hoem04/powergkrry/lv_challenge/data/poriginal_png'))
                files.sort()

                i = 0
                for name in sample:
                    if i < 32:
                        img = Image.open(os.path.join('/hoem04/powergkrry/lv_challenge/data/poriginal_png', files[name]))
                        img.save(os.path.join(os.getcwd(), files[name]))
                        i += 1
                    else:
                        break
        else: #full
            diffu_dir = '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_' + subdirectory + '_png'

            if directory[-9:-4] == 'train':  # train
                i = 0
                for name in sample:
                    if i < 32:
                        i += 1
                    else:
                        if subdirectory == 'original': #for original
                            files = os.listdir(os.path.join('/hoem04/powergkrry/lv_challenge/data/poriginal_png'))
                            files.sort()
                            img = Image.open(os.path.join('/hoem04/powergkrry/lv_challenge/data/poriginal_png', files[name]))
                            img.save(os.path.join(os.getcwd(), files[name]))
                        else: #for diffused
                            files = os.listdir(diffu_dir)
                            files.sort()
                            img = Image.open(os.path.join(diffu_dir, files[name]))
                            img.save(os.path.join(os.getcwd(), files[name]))
            else:  # test
                i = 0
                for name in sample:
                    if i < 32:
                        if subdirectory == 'original': #for original
                            files = os.listdir(os.path.join('/hoem04/powergkrry/lv_challenge/data/poriginal_png'))
                            files.sort()
                            img = Image.open(os.path.join('/hoem04/powergkrry/lv_challenge/data/poriginal_png', files[name]))
                            img.save(os.path.join(os.getcwd(), files[name]))
                        else: #for diffused
                            files = os.listdir(diffu_dir)
                            files.sort()
                            img = Image.open(os.path.join(diffu_dir, files[name]))
                            img.save(os.path.join(os.getcwd(), files[name]))
                        i += 1
                    else:
                        break



random.seed(1)
sample = random.sample([i for i in range(161)], 161) #161 : number of images


make_contour_directory = ['pcontour_train_png', 'pcontour_test_png']
make_original_directory = ['poriginal_onlyoriginal_train_png', 'poriginal_onlyoriginal_test_png', 'poriginal_full_train_png', 'poriginal_full_test_png']
make_original_subdirectory = ['original', 'diffu3', 'diffu6', 'diffu9', 'diffu12'] #change
origin_directory = ['/hoem04/powergkrry/lv_challenge/data/poriginal_png/',
             '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu3_png/',
             '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu6_png/',
             '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu9_png/',
             '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu12_png/',
'/hoem04/powergkrry/lv_challenge/data/pcontour_png'] #change


for contour in make_contour_directory:
    if not os.path.exists(os.path.join('/hoem04/powergkrry/lv_challenge/data/dataset', contour)):
        os.makedirs(os.path.join('/hoem04/powergkrry/lv_challenge/data/dataset', contour))
        os.makedirs(os.path.join('/hoem04/powergkrry/lv_challenge/data/dataset', contour, 'folder'))
        os.chdir(os.path.join('/hoem04/powergkrry/lv_challenge/data/dataset', contour, 'folder'))
        save_image(directory = contour, sample = sample)

for original in make_original_directory:
    if not os.path.exists(os.path.join('/hoem04/powergkrry/lv_challenge/data/dataset', original)):
        os.makedirs(os.path.join('/hoem04/powergkrry/lv_challenge/data/dataset', original))
        os.chdir(os.path.join('/hoem04/powergkrry/lv_challenge/data/dataset', original))
        point1 = os.getcwd()

        if original[10:14] != 'only':
            for sub in make_original_subdirectory:
                if not os.path.exists(os.path.join(os.getcwd(), sub)):
                    os.makedirs(os.path.join(os.getcwd(), sub))
                    os.chdir(os.path.join(os.getcwd(), sub))

                    os.makedirs(os.path.join(os.getcwd(), 'folder'))
                    os.chdir(os.path.join(os.getcwd(), 'folder'))
                    save_image(directory = original, subdirectory = sub, sample = sample)
                    os.chdir(point1)
        else:
            if not os.path.exists(os.path.join(os.getcwd(), 'original')):
                os.makedirs(os.path.join(os.getcwd(), 'original'))
                os.chdir(os.path.join(os.getcwd(), 'original'))

                os.makedirs(os.path.join(os.getcwd(), 'folder'))
                os.chdir(os.path.join(os.getcwd(), 'folder'))
                save_image(directory = original, sample = sample)
                os.chdir(point1)

"""        
origin_directory = ['/hoem04/powergkrry/lv_challenge/data/poriginal_png/',
             '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu3_png/',
             '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu6_png/',
             '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu9_png/',
             '/hoem04/powergkrry/lv_challenge/data/dataset/poriginal_diffu12_png/',
'/hoem04/powergkrry/lv_challenge/data/pcontour_png']
train_directory = ['/hoem04/powergkrry/lv_challenge/data/poriginal_train_png/original/',
                   '/hoem04/powergkrry/lv_challenge/data/poriginal_train_png/diffu2/',
                   '/hoem04/powergkrry/lv_challenge/data/poriginal_train_png/diffu4/',
                   '/hoem04/powergkrry/lv_challenge/data/poriginal_train_png/diffu6/',
                   '/hoem04/powergkrry/lv_challenge/data/poriginal_train_png/diffu8/', 
'/hoem04/powergkrry/lv_challenge/data/pcontour_train_png']
test_directory = ['/hoem04/powergkrry/lv_challenge/data/poriginal_test_png/original/',
                   '/hoem04/powergkrry/lv_challenge/data/poriginal_test_png/diffu2/',
                   '/hoem04/powergkrry/lv_challenge/data/poriginal_test_png/diffu4/',
                   '/hoem04/powergkrry/lv_challenge/data/poriginal_test_png/diffu6/',
                   '/hoem04/powergkrry/lv_challenge/data/poriginal_test_png/diffu8/', 
'/hoem04/powergkrry/lv_challenge/data/pcontour_test_png']


for origin, train, test in zip(origin_directory, train_directory, test_directory):
    files = os.listdir(origin)
    files.sort()

    random.seed(1)
    sample = random.sample([i for i in range(161)], 161) #161 : number of images

    i = 0
    for name in sample:
        if i < 32:
            img = Image.open(os.path.join(origin, files[name]))
            img.save(os.path.join(test, files[name]))
            i+=1
        else:
            img = Image.open(os.path.join(origin, files[name]))
            img.save(os.path.join(train, files[name]))
"""
