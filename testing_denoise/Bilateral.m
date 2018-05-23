delete(gcp)
parpool(4);

dirname = './testing_image/';
savedirname = './result/find_interval/';
MyFolderInfo = dir(dirname);

parfor j=3:length(MyFolderInfo)
    var=[10^-8,10^-7,10^-6,10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2];
    %var=[2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5];
    %var=[1,3,5,7,9,11,13,17,33,65,129]
    showimg=zeros(256,256,12);
    img = imread(strcat(dirname, MyFolderInfo(j).name));
    img = im2double(img);
    showimg(:,:,1)=img;
    
    for i = 1:11
        dst = imbilatfilt(img,var(i));
        %dst = imbilatfilt(img,0.01,var(i));
        %dst = imbilatfilt(img,0.01,1,'NeighborhoodSize',var(i));
        showimg(:,:,i+1)=dst;
    end
    
    figure;
    title('degreeOfSmoothing');
    for i = 1:12
        subplot(3,4,i), imshow(showimg(:,:,i));
        
        if i==1
           title('original');
        else
           title(num2str(var(i-1)));
        end
    end
    tightfig;
    saveas(gcf,strcat(savedirname,MyFolderInfo(j).name,'_Bilateral_degreeOfSmoothing.png'));
end