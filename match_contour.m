%%Make icontour/ocontour images for matching contour .txt files and images

%change only line 4 to 12 or 17
which1 = 'challenge_validation';
which2 = 'validation';
which3 = 'ValidationDataContours';
mkdir ./match_contour/validation/SC-HF-I-7
folder_name1 = 'SC-HF-I-7';
folder_name2 = 'SC-HF-I-07';
image_number = '0211';
contour_number = '0160';
contour_part = 'Sunnybrook_Cardiac_MR_Database_ContoursPart2';

%two times of challenge_online at data1 path (, which1, '/')
%data1 : path+image_name of original image
%data2 : path+image_name of output image
data1 = {strcat('/hoem04/powergkrry/lv_challenge/', which1, '/', folder_name1, '/IM-', image_number, '-', contour_number)};
data2 = {strcat('/hoem04/powergkrry/lv_challenge/match_contour/', which2, '/', folder_name1, '/IM-', image_number, '-', contour_number, '_icont_1.png'),
         strcat('/hoem04/powergkrry/lv_challenge/match_contour/', which2, '/', folder_name1, '/IM-', image_number, '-', contour_number, '_icont.png'),
         strcat('/hoem04/powergkrry/lv_challenge/match_contour/', which2, '/', folder_name1, '/IM-', image_number, '-', contour_number, '_ocont_1.png'),
         strcat('/hoem04/powergkrry/lv_challenge/match_contour/', which2, '/', folder_name1, '/IM-', image_number, '-', contour_number, '_ocont.png')};

%path_coor_i : path+contour_name of in contour .txt file
%path_coor_o : path+contour_name of out contour .txt file
path_coor_i =  strcat('/hoem04/powergkrry/lv_challenge/', contour_part, '/', which3, '/', folder_name2, '/contours-manual/IRCCI-expert/IM-0001-', contour_number, '-icontour-manual.txt');
path_coor_o =  strcat('/hoem04/powergkrry/lv_challenge/', contour_part, '/', which3, '/', folder_name2, '/contours-manual/IRCCI-expert/IM-0001-', contour_number, '-ocontour-manual.txt');

%load icontour .txt
coordinates_i = fopen(path_coor_i, 'r');
coordinates_i_data = textscan(coordinates_i, '%f%f', 'delimiter', ' ');
x = coordinates_i_data{1};
y = coordinates_i_data{2};
fclose(coordinates_i);

%make icontour mask
mask_i = poly2mask(x, y, 256, 256); %icontour mask
mask_i_invert = ~mask_i; %mask excluding icontour mask
mask_i_cast = cast(mask_i, 'uint16'); %cast logical to uint16
mask_i_invert_cast = cast(mask_i_invert, 'uint16'); %cast logical to uint16

%load ocontour .txt
coordinates_o = fopen(path_coor_o, 'r');
coordinates_o_data = textscan(coordinates_o, '%f%f', 'delimiter', ' ');
x = coordinates_o_data{1};
y = coordinates_o_data{2};
fclose(coordinates_o);

%make ocontour mask
mask_o = poly2mask(x, y, 256, 256); %ocontour mask
mask_o_invert = ~mask_o; %mask excluding ocontour mask 
mask_o_cast = cast(mask_o, 'uint16'); %cast logical to uint16
mask_o_invert_cast = cast(mask_o_invert, 'uint16'); %cast logical to uint16 

%read dicom
one = data1(1);
two = data1(1);
original1 = dicomread(one{1});
original2 = dicomread(two{1});
original1_cast = cast(original1, 'uint16'); %cast int16 to uint16
original2_cast = cast(original2, 'uint16'); %cast int16 to uint16

contour1_1 = original1_cast.*mask_i_cast; %icontour image
contour1 = original1_cast.*mask_i_invert_cast; %image excluding icontour image
contour2_1 = original2_cast.*mask_o_cast; %ocontour image
contour2 = original2_cast.*mask_o_invert_cast; %image excluding ocontour image
contour1_1_cast = cast(contour1_1, 'double')/1000; %re-scaling to make the image visible
contour1_cast = cast(contour1, 'double')/1000; %re-scaling to make the image visible
contour2_1_cast = cast(contour2_1, 'double')/1000; %re-scaling to make the image visible
contour2_cast = cast(contour2, 'double')/1000; %re-scaling to make the image visible

%write image
one = data2(1);
two = data2(2);
three = data2(3);
four = data2(4);
imwrite(contour1_1_cast, one{1});
imwrite(contour1_cast, two{1});
imwrite(contour2_1_cast, three{1});
imwrite(contour2_cast, four{1});