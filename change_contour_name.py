##changing name of contour file
#numpy package should be updated to execute code below
import os
import numpy as np

dcm = os.listdir("./")

match_contour = np.genfromtxt("/hoem04/powergkrry/lv_challenge/match_contour.txt", dtype = None, encoding = None)

which_dir = ['/hoem04/powergkrry/lv_challenge/Sunnybrook_Cardiac_MR_Database_ContoursPart3/TrainingDataContours',
             '/hoem04/powergkrry/lv_challenge/Sunnybrook_Cardiac_MR_Database_ContoursPart2/ValidationDataContours',
             '/hoem04/powergkrry/lv_challenge/Sunnybrook_Cardiac_MR_Database_ContoursPart1/OnlineDataContours']

for location in which_dir: #the location where contour files are saved
    os.chdir(location)
    dir_names = os.listdir("./")

    #check whether there is a folder or not
    dir_names_processed = [k for k in dir_names for l in match_contour if k == l[0]] #load directory names that should be considered

    for processed in dir_names_processed:
        move_path = location + '/' + processed + '/contours-manual/IRCCI-expert/'
        os.chdir(move_path)

        for matching in match_contour: #find image_number that will be attached to file name
            if (processed == matching[0]):
                image_number = matching[1]

        contour_name = os.listdir("./")

        for name in contour_name:
            read_contour = './' + name
            write_contour = '/hoem04/powergkrry/lv_challenge/data/' + image_number + name[7:]

            #read contour file for saving
            before_contour = np.genfromtxt(read_contour, dtype=None, encoding=None)

            #write contour file to new location
            f = open(write_contour, 'w')
            for l in range(0, len(before_contour)):
                txt = '%6.2f\t' % (before_contour[l][0]) + '%6.2f' % (before_contour[l][1]) + '\n'
                f.write(txt)
            f.close()
