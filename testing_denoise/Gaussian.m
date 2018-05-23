% delete(gcp)
% parpool(4);

dirname = './testing_image/';
savedirname = './result/find_interval/';
MyFolderInfo = dir(dirname);

%parfor j=3:length(MyFolderInfo)
for j=3:length(MyFolderInfo)
    %var=[2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5];
    %var=[1,3,5,7,9,11,13,17,33,65,129];
    %var=["spatial","frequency","spatial","frequency","spatial","frequency","spatial","frequency","spatial","frequency","spatial"];

    sigma = [2^-5,2^-3,2^-2,2^0,2^2,2^4];
    filt = ["spatial","frequency"];
    
    showimg=zeros(256,256,12);
    img = imread(strcat(dirname, MyFolderInfo(j).name));
    img = im2double(img);
    showimg(:,:,1)=img;
    
%     for i = 1:11
    for i = 1:12
        %dst = imgaussfilt(img,var(i));
        %dst = imgaussfilt(img,0.5,'FilterSize',var(i));
        %dst = imgaussfilt(img,0.5,'FilterSize',3,'FilterDomain',var(i));
        dst = imgaussfilt(img,sigma(floor((i+1)/2)),'FilterDomain',filt(mod(i,2)+1));
        
        showimg(:,:,i+1)=dst;
    end
    
    figure;
    title('');
    for i = 1:12
        subplot(3,4,i), imshow(showimg(:,:,i));
        
%         if i==1
%            title('original');
%         else
%            title(num2str(var(i-1)));
%         end
        title(strcat(num2str(sigma(floor((i+1)/2))),',',num2str(filt(mod(i,2)+1))));
    end
    tightfig;
    saveas(gcf,strcat(savedirname,MyFolderInfo(j).name,'_Gaussian_Sigma_FilterDomain.png'));
end