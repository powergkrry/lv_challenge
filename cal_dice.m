dir_list_out = "/home/yeonjee/lv_challenge/temp/";
%dir_list_truth = "/home/yeonjee/lv_challenge/data/dataset/dataset07_output/ioutput_TV_png/";
dir_list_truth = "/home/yeonjee/lv_challenge/data/dataset/dataset06/i/test_output/";
%flip = ["o/", "rl1/", "rl2/", "rl3/", "ov/", "rl1v/", "rl2v/", "rl3v/"];
flip = ["o/", "h/", "v/", "hv/", "rr/", "rl/"];

%dice
similarity = 0;
similarity_total = 0;
for i = 1:length(flip)
    dir_list_out_img = dir(strcat(dir_list_out, flip(i)));
    %dir_list_truth_img = dir(strcat(dir_list_truth, "output_", flip(i)));
    dir_list_truth_img = dir(strcat(dir_list_truth, flip(i), "output_", flip(i)));
    
    for j = 3:length(dir_list_out_img)
       out_img = imread(char(strcat(dir_list_out, flip(i), dir_list_out_img(j).name)));
       
       %otsu thresholding
       level = graythresh(out_img);
       out_img = imbinarize(out_img, level);
       
       %image padding for making 256*256 images
       %%out_img = padarray(out_img, [64 64], 0, 'both');
       
       %out_img = logical(out_img(:,:,1));
       
       %truth_img = imread(char(strcat(dir_list_truth, "output_", flip(i), dir_list_truth_img(j).name)));
       truth_img = imread(char(strcat(dir_list_truth, flip(i), "output_", flip(i), dir_list_truth_img(j).name)));
       
%        [p3, p4] = size(truth_img);
%        q1 = 127; % size of the crop box
%        i3_start = floor((p3-q1)/2); % or round instead of floor; using neither gives warning
%        i3_stop = i3_start + q1;
% 
%        i4_start = floor((p4-q1)/2);
%        i4_stop = i4_start + q1;
% 
%        truth_img = truth_img(i3_start:i3_stop, i4_start:i4_stop, :);
       
       truth_img = logical(truth_img);
       
       similarity = similarity + dice(out_img, truth_img);
    end
    
    similarity / (length(dir_list_out_img) - 2)
    similarity_total = similarity_total + similarity;
    similarity = 0;
end

similarity_total/(length(dir_list_out_img) - 2)/length(flip)


%fmeasure
similarity = 0;
similarity_total = 0;
for i = 1:length(flip)
    dir_list_out_img = dir(strcat(dir_list_out, flip(i)));
    dir_list_truth_img = dir(strcat(dir_list_truth, "output_", flip(i)));
    
    for j = 3:length(dir_list_out_img)
       out_img = imread(char(strcat(dir_list_out, flip(i), dir_list_out_img(j).name)));  
       
       %otsu thresholding
       level = graythresh(out_img);
       out_img = imbinarize(out_img, level);
       
       %image padding for making 256*256 images
       %%out_img = padarray(out_img, [64 64], 0, 'both');

       %out_img = logical(out_img(:,:,1));
       
       truth_img = imread(char(strcat(dir_list_truth, "output_", flip(i), dir_list_truth_img(j).name)));
       
%        [p3, p4] = size(truth_img);
%        q1 = 127; % size of the crop box
%        i3_start = floor((p3-q1)/2); % or round instead of floor; using neither gives warning
%        i3_stop = i3_start + q1;
% 
%        i4_start = floor((p4-q1)/2);
%        i4_stop = i4_start + q1;
% 
%        truth_img = truth_img(i3_start:i3_stop, i4_start:i4_stop, :);
       truth_img = logical(truth_img);
       
       save = computeFmeasure2(truth_img, out_img);
       similarity = similarity + save(6);
    end
    
    similarity / (length(dir_list_out_img) - 2)
    similarity_total = similarity_total + similarity;
    similarity = 0;
end

similarity_total/(length(dir_list_out_img) - 2)/length(flip)
