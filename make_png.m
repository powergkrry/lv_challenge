dirname_save1=["ioriginal_png"]; %, "ooriginal_A_png", "poriginal_A_png"];
dirname_save2=["/home/yeonjee/lv_challenge/data/raw/ioriginal_png/",...
    "/home/yeonjee/lv_challenge/data/raw/ooriginal_png/",...
    "/home/yeonjee/lv_challenge/data/raw/poriginal_png/"];
dirname_load1=["/home/yeonjee/lv_challenge/data/ioriginal_png/",...
    "/home/yeonjee/lv_challenge/data/ooriginal_png/",...
    "/home/yeonjee/lv_challenge/data/poriginal_png/"];
dirname_load2=["/home/yeonjee/lv_challenge/data/icontour_png/",...
    "/home/yeonjee/lv_challenge/data/ocontour_png/",...
    "/home/yeonjee/lv_challenge/data/pcontour_png/"];
flip=["o", "rl1", "rl2", "rl3", "ov", "rl1v", "rl2v", "rl3v"];


cd /home/yeonjee/lv_challenge/data/
mkdir raw
cd /home/yeonjee/lv_challenge/data/raw/
mkdir(dirname_save1(1))


% for original input
for i=1:length(dirname_save1)
    cd(dirname_save2(i))
    
    original_names = dir(dirname_load1(i));
    
    for j=3:length(original_names)
        img = imread(char(strcat(dirname_load1(i), original_names(j).name)));
        img = im2double(img);
        
        for m=1:length(flip)
            mkdir_name = strcat("original", "_", flip(m));
            
            if j == 3
                mkdir(mkdir_name)
            end
            
            cd(strcat(dirname_save2(i), mkdir_name))
            
            dst = imflip(img, flip(m));
            dst = whiten(dst);
            
            dst = im2uint8(dst);
            imwrite(dst, strcat('./', num2str(j-2, '%0.3d'), '.png'));
                    
            cd ..
        end
    end
end
% 
% 
% % output
% for i=1:length(dirname_save1)
%     cd(dirname_save2(i))
%     
%     original_names = dir(dirname_load2(i));
%     
%     for j=3:length(original_names)
%         img = imread(char(strcat(dirname_load2(i), original_names(j).name)));
%         img = im2double(img);
%         
%         for m=1:length(flip)
%             mkdir_name = strcat("output", "_", flip(m));
%             
%             if j == 3
%                 mkdir(mkdir_name)
%             end
%             
%             cd(strcat(dirname_save2(i), mkdir_name))
%             
%             dst = imflip(img, flip(m));
%             
%             dst = im2uint8(dst);
%             imwrite(dst, strcat('./', num2str(j-2, '%0.3d'), '.png'));
%                     
%             cd ..
%         end
%     end
% end
% 
% 
% empty=[0];
% G_sigma=[16, 32, 64];
% 
% % for input G
% for i=1:length(dirname_save1)
%     cd(dirname_save2(i))
%     
%     original_names = dir(dirname_load1(i));
%     
%     for j=3:length(original_names)
%         img = imread(char(strcat(dirname_load1(i), original_names(j).name)));
%         img = im2double(img);
%         
%         for k=1:length(empty)
%             for l=1:length(G_sigma)
%                 for m=1:length(flip)
%                     mkdir_name = strcat(num2str(G_sigma(l)), "_", flip(m));
%                     
%                     if j == 3
%                         mkdir(mkdir_name)
%                     end
%                     
%                     cd(strcat(dirname_save2(i), mkdir_name))
%                     
%                     dst = imgaussfilt(img, G_sigma(l));
%                     dst = imflip(dst, flip(m));
%                     
%                     dst = im2uint8(dst);
%                     imwrite(dst, strcat('./', num2str(j-2, '%0.3d'), '.png'));
%                     
%                     cd ..
%                 end
%             end
%         end
%     end
% end
% 
% 
% empty=[0];
% mu=[0.02, 0.1];
% N = 256;
% 
% % for input TV
% for i=1:length(dirname_save1)
%     cd(dirname_save2(i))
%     
%     original_names = dir(dirname_load1(i));
%     
%     for j=3:length(original_names)
%         img = imread(char(strcat(dirname_load1(i), original_names(j).name)));
%         img = im2double(img);
%         
%         for k=1:length(empty)
%             for l=1:length(mu)
%                 for m=1:length(flip)
%                     mkdir_name1 = strcat(num2str(mu(l)), "_", flip(m));
%                     mkdir_name2 = strcat(num2str(mu(l)), "_", "noise", "_", flip(m));
%                     
%                     if j == 3
%                         mkdir(mkdir_name1)
%                         mkdir(mkdir_name2)
%                     end
% 
%                     % TV                    
%                     cd(strcat(dirname_save2(i), mkdir_name1))
%                     
%                     dst = SB_ATV(img, mu(l));
%                     dst = reshape(dst,N,N);
% 
%                     dst_f = rescale(dst);
%                     dst_f = imflip(dst_f, flip(m));
%                     dst_f = im2uint8(dst_f);
% 
%                     imwrite(dst_f, strcat('./', num2str(j-2, '%0.3d'), '.png'));
%                     
%                     cd ..
% 
%                     % TV noise
%                     cd(strcat(dirname_save2(i), mkdir_name2))
% 
%                     noise = img - dst;
%                     noise = imflip(noise, flip(m));
% 
%                     noise = rescale(noise);
%                     noise = im2uint8(noise);
%                     imwrite(noise, strcat('./', num2str(j-2, '%0.3d'), '.png'));
% 
%                     cd ..
%                 end
%             end
%         end
%     end
% end
