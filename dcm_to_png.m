%%Convert dcm to png

%convert icontour dicom
dir_list = dir('/home/yeonjee/lv_challenge/data/icontour_txt'); %list of contour files needed to be converted
mkdir /home/yeonjee/lv_challenge/data/ioriginal_png

for i = 3:length(dir_list) %1 is ., 2 is .., 3 is content
    image_name_i = dir_list(i).name(1:12);
    path_dcm_i = strcat('/home/yeonjee/lv_challenge/data/original_dcm/', image_name_i, '.dcm'); %dcm path to load
    path_image_i = strcat('/home/yeonjee/lv_challenge/data/ioriginal_png/', num2str(i-2, '%0.3d'), '.png'); %png image path to save
    
    %load dicom
    dicom_i = dicomread(path_dcm_i);
    dicom_i_cast = cast(dicom_i, 'uint8');
    dicom_i_re = imadjust(dicom_i_cast); %re-scaling to make the image visible
    
    %make png
    imwrite(dicom_i_re, path_image_i);
end

%convert ocontour dicom
dir_list = dir('/home/yeonjee/lv_challenge/data/ocontour_txt'); %list of contour files needed to be converted
mkdir /home/yeonjee/lv_challenge/data/ooriginal_png

for i = 3:length(dir_list) %1 is ., 2 is .., 3 is content
    image_name_o = dir_list(i).name(1:12);
    path_dcm_o = strcat('/home/yeonjee/lv_challenge/data/original_dcm/', image_name_o, '.dcm'); %dcm path to load
    path_image_o = strcat('/home/yeonjee/lv_challenge/data/ooriginal_png/', num2str(i-2, '%0.3d'), '.png'); %png image path to save
    
    %load dicom
    dicom_o = dicomread(path_dcm_o);
    dicom_o_cast = cast(dicom_o, 'uint8');
    dicom_o_re = imadjust(dicom_o_cast); %re-scaling to make the image visible
    
    %make png
    imwrite(dicom_o_re, path_image_o);
end

%convert pcontour dicom
dir_list = dir('/home/yeonjee/lv_challenge/data/pcontour_txt'); %list of contour files needed to be converted
mkdir /home/yeonjee/lv_challenge/data/poriginal_png

check_multiple_pcontours = 0;
image_order = 1;
for i = 3:length(dir_list) %1 is ., 2 is .., 3 is content
    image_name_p = dir_list(i).name(1:12);
    path_dcm_p = strcat('/home/yeonjee/lv_challenge/data/original_dcm/', image_name_p, '.dcm'); %dcm path to load
    path_image_p = strcat('/home/yeonjee/lv_challenge/data/poriginal_png/', num2str(image_order, '%0.3d'), '.png'); %png image path to save

    if i ~= length(dir_list) % if i == length(dir_list), there will be an error at dir_list(i + 1)
        if dir_list(i).name(1:12) == dir_list(i + 1).name(1:12) %check whether there are multiple pcontours or not
            check_multiple_pcontours = 1; %There are multiple pcontours
            continue
        end
    end
    
    %load dicom
    dicom_p = dicomread(path_dcm_p);
    dicom_p_cast = cast(dicom_p, 'uint8');
    dicom_p_re = imadjust(dicom_p_cast); %re-scaling to make the image visible
    
    %make png
    imwrite(dicom_p_re, path_image_p);
    image_order = image_order + 1;
end
