%%Convert dcm to png

%convert icontour dicom
dir_list = dir('/hoem04/powergkrry/lv_challenge/data/icontour_txt'); %list of contour files needed to be converted
mkdir /hoem04/powergkrry/lv_challenge/data/ioriginal_png

for i = 3:length(dir_list) %1 is ., 2 is .., 3 is content
    image_name_i = dir_list(i).name(1:12);
    path_dcm_i = strcat('/hoem04/powergkrry/lv_challenge/data/original_dcm/', image_name_i, '.dcm'); %dcm path to load
    path_image_i = strcat('/hoem04/powergkrry/lv_challenge/data/ioriginal_png/', image_name_i, '-iori.png'); %png image path to save
    
    %load dicom
    dicom_i = dicomread(path_dcm_i);
    dicom_i_cast = cast(dicom_i, 'uint16');
    dicom_i_re = imadjust(dicom_i_cast); %re-scaling to make the image visible
    
    %make png
    imwrite(dicom_i_re, path_image_i);
end

%convert ocontour dicom
dir_list = dir('/hoem04/powergkrry/lv_challenge/data/ocontour_txt'); %list of contour files needed to be converted
mkdir /hoem04/powergkrry/lv_challenge/data/ooriginal_png

for i = 3:length(dir_list) %1 is ., 2 is .., 3 is content
    image_name_o = dir_list(i).name(1:12);
    path_dcm_o = strcat('/hoem04/powergkrry/lv_challenge/data/original_dcm/', image_name_o, '.dcm'); %dcm path to load
    path_image_o = strcat('/hoem04/powergkrry/lv_challenge/data/ooriginal_png/', image_name_o, '-oori.png'); %png image path to save
    
    %load dicom
    dicom_o = dicomread(path_dcm_o);
    dicom_o_cast = cast(dicom_o, 'uint16');
    dicom_o_re = imadjust(dicom_o_cast); %re-scaling to make the image visible
    
    %make png
    imwrite(dicom_o_re, path_image_o);
end

%convert pcontour dicom
dir_list = dir('/hoem04/powergkrry/lv_challenge/data/pcontour_txt'); %list of contour files needed to be converted
mkdir /hoem04/powergkrry/lv_challenge/data/poriginal_png

for i = 3:length(dir_list) %1 is ., 2 is .., 3 is content
    image_name_p = dir_list(i).name(1:12);
    path_dcm_p = strcat('/hoem04/powergkrry/lv_challenge/data/original_dcm/', image_name_p, '.dcm'); %dcm path to load
    path_image_p = strcat('/hoem04/powergkrry/lv_challenge/data/poriginal_png/', image_name_p, '-pori.png'); %png image path to save
    
    %load dicom
    dicom_p = dicomread(path_dcm_p);
    dicom_p_cast = cast(dicom_p, 'uint16');
    dicom_p_re = imadjust(dicom_p_cast); %re-scaling to make the image visible
    
    %make png
    imwrite(dicom_p_re, path_image_p);
end
