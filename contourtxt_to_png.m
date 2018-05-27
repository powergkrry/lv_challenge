%%Convert mask txt file to png

%convert icontour
dir_list = dir('/home/yeonjee/lv_challenge/data/icontour_txt');
mkdir /home/yeonjee/lv_challenge/data/icontour_png

image_order = 1;
for i = 3:length(dir_list) %1 is ., 2 is .., 3 is content
    path_coor_i = strcat('/home/yeonjee/lv_challenge/data/icontour_txt/', dir_list(i).name); %mask txt path to load
    image_name_i = dir_list(i).name(1:12);
    path_image_i = strcat('/home/yeonjee/lv_challenge/data/icontour_png/', num2str(image_order, '%0.3d'), '.png'); %png image path to save
    
    %load icontour .txt
    coordinates_i = fopen(path_coor_i, 'r');
    coordinates_i_data = textscan(coordinates_i, '%f%f', 'delimiter', '\t'); %not ' '
    x = coordinates_i_data{1};
    y = coordinates_i_data{2};
    fclose(coordinates_i);
    
    %make icontour mask
    mask_i = poly2mask(x, y, 256, 256); %icontour mask
    mask_i_cast = cast(mask_i, 'double'); %cast logical to double
    
    %save image
    imwrite(mask_i_cast, path_image_i);
    image_order = image_order + 1;
end

%convert ocontour
dir_list = dir('/home/yeonjee/lv_challenge/data/ocontour_txt');
mkdir /home/yeonjee/lv_challenge/data/ocontour_png

image_order = 1;
for i = 3:length(dir_list) %1 is ., 2 is .., 3 is content
    path_coor_o = strcat('/home/yeonjee/lv_challenge/data/ocontour_txt/', dir_list(i).name); %mask txt path to load
    image_name_o = dir_list(i).name(1:12);
    path_image_o = strcat('/home/yeonjee/lv_challenge/data/ocontour_png/', num2str(image_order, '%0.3d'), '.png'); %png image path to save
    
    %load ocontour .txt
    coordinates_o = fopen(path_coor_o, 'r');
    coordinates_o_data = textscan(coordinates_o, '%f%f', 'delimiter', '\t'); %not ' '
    x = coordinates_o_data{1};
    y = coordinates_o_data{2};
    fclose(coordinates_o);
    
    %make ocontour mask
    mask_o = poly2mask(x, y, 256, 256); %ocontour mask
    mask_o_cast = cast(mask_o, 'double'); %cast logical to double
    
    %save image
    imwrite(mask_o_cast, path_image_o);
    image_order = image_order + 1;
end

%convert pcontour
dir_list = dir('/home/yeonjee/lv_challenge/data/pcontour_txt');
mkdir /home/yeonjee/lv_challenge/data/pcontour_png

check_multiple_pcontours = 0;
image_order = 1;
for i = 3:length(dir_list) %1 is ., 2 is .., 3 is content
    path_coor_p = strcat('/home/yeonjee/lv_challenge/data/pcontour_txt/', dir_list(i).name); %mask txt path to load
    image_name_p = dir_list(i).name(1:12);
    path_image_p = strcat('/home/yeonjee/lv_challenge/data/pcontour_png/', num2str(image_order, '%0.3d'), '.png'); %png image path to save
    
    %load pcontour .txt
    coordinates_p = fopen(path_coor_p, 'r');
    coordinates_p_data = textscan(coordinates_p, '%f%f', 'delimiter', '\t'); %not ' '
    x = coordinates_p_data{1};
    y = coordinates_p_data{2};
    fclose(coordinates_p);
    
    %make pcontour mask
    if check_multiple_pcontours == 0
        mask_p = poly2mask(x, y, 256, 256); %pcontour mask
    end
    
    if i ~= length(dir_list) % if i == length(dir_list), there will be an error at dir_list(i + 1)
        if dir_list(i).name(1:12) == dir_list(i + 1).name(1:12) %check whether there are multiple pcontours or not
            if check_multiple_pcontours == 1 %if check_multible_pcontours == 1, mask should be overlapped with the previous mask.
                mask_p_multiple = poly2mask(x, y, 256, 256); %another pcontour mask
                mask_p = mask_p | mask_p_multiple; %overlay multiple pcontours
            end
            check_multiple_pcontours = 1; %There are multiple pcontours
            continue
        end
    end
    
    if check_multiple_pcontours == 1 %if check_multible_pcontours == 1, mask should be overlapped with the previous mask.

        mask_p_multiple = poly2mask(x, y, 256, 256); %another pcontour mask
        mask_p = mask_p | mask_p_multiple; %overlay multiple pcontours
    end
    
    mask_p_cast = cast(mask_p, 'double'); %cast logical to double

    %save image
    imwrite(mask_p_cast, path_image_p);
    check_multiple_pcontours = 0; %reset after saving
    image_order = image_order + 1;
end
