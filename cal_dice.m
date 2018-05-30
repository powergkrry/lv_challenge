dir_list_out = "/home/yeonjee/lv_challenge/temp/";
dir_list_truth = "/home/yeonjee/lv_challenge/data/dataset/dataset04/p/test_output/";
flip = ["h/","v/","hv/","o/","rr/","rl/"];

% %dice
% similarity = 0;
% similarity_total = 0;
% for i = 1:length(flip)
%     dir_list_out_img = dir(strcat(dir_list_out, flip(i)));
%     dir_list_truth_img = dir(strcat(dir_list_truth, flip(i), "output_", flip(i)));
%     
%     for j = 3:length(dir_list_out_img)
%        out_img = imread(char(strcat(dir_list_out, flip(i), dir_list_out_img(j).name)));
%        out_img = logical(out_img(:,:,1));
%        truth_img = imread(char(strcat(dir_list_truth, flip(i), "output_", flip(i), dir_list_truth_img(j).name)));
%        truth_img = logical(truth_img);
%        
%        similarity = similarity + dice(out_img, truth_img);
%     end
%     
%     similarity / (length(dir_list_out_img) - 2)
%     similarity_total = similarity_total + similarity;
%     similarity = 0;
% end
% 
% similarity_total/(length(dir_list_out_img) - 2)/length(flip)


%fmeasure
similarity = 0;
similarity_total = 0;
for i = 1:length(flip)
    dir_list_out_img = dir(strcat(dir_list_out, flip(i)));
    dir_list_truth_img = dir(strcat(dir_list_truth, flip(i), "output_", flip(i)));
    
    for j = 3:length(dir_list_out_img)
       out_img = imread(char(strcat(dir_list_out, flip(i), dir_list_out_img(j).name)));
       out_img = logical(out_img(:,:,1));
       truth_img = imread(char(strcat(dir_list_truth, flip(i), "output_", flip(i), dir_list_truth_img(j).name)));
       truth_img = logical(truth_img);
       
       similarity = similarity + computeFmeasure(out_img, truth_img);
    end
    
    similarity / (length(dir_list_out_img) - 2)
    similarity_total = similarity_total + similarity;
    similarity = 0;
end

similarity_total/(length(dir_list_out_img) - 2)/length(flip)