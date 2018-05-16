delete(gcp)
parpool(4);

dirname = './testing_image/';
savedirname = './result/find_interval/';
MyFolderInfo = dir(dirname);

parfor j=3:length(MyFolderInfo)
    %var=[1,3,5,7,9,11,13,17,33,65,129]
    %var=[1,3,5,7,9,11,13,17,33,65,129]
    var=[2^1,2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10,2^11]

    showimg=zeros(256,256,12);
    img = imread(strcat(dirname, MyFolderInfo(j).name));
    img = im2double(img);
    showimg(:,:,1)=img;
    
    for i = 1:11
        %dst = NLmeansfilter(img,var(i),1,1);
        %dst = NLmeansfilter(img,1,var(i),1);
        dst = NLmeansfilter(img,1,1,var(i));

        showimg(:,:,i+1)=dst;
        j
        i
    end
    
    figure;
    title('h');
    for i = 1:12
        subplot(3,4,i), imshow(showimg(:,:,i));
        
        if i==1
           title('original');
        else
           title(num2str(var(i-1)));
        end
    end
    tightfig;
    saveas(gcf,strcat(savedirname,MyFolderInfo(j).name,'_NLMeans_h.png'));
end
