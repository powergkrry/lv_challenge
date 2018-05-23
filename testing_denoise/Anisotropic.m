% delete(gcp)
% parpool(4);

dirname = './testing_image/';
savedirname = './result/find_interval/';
MyFolderInfo = dir(dirname);

% parfor j=3:length(MyFolderInfo)
for j=3:length(MyFolderInfo)
    %var=[10^-2,10^-1,10^0,10^1,10^2,10^3,10^4,10^5,10^6,10^7,10^8];
    %var=[2^1,2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10,2^11];
    %var=["maximal","minimal","maximal","minimal","maximal","minimal","maximal","minimal","maximal","minimal","maximal"];
    %var=["exponential","quadratic","exponential","quadratic","exponential","quadratic","exponential","quadratic","exponential","quadratic","exponential"];
    var=[10^-2,2*10^-2,3*10^-2,4*10^-2,5*10^-2,6*10^-2,7*10^-2,8*10^-2,9*10^-2,10^-1,10^0];
    %var=[10^-1,2*10^-1,3*10^-1,4*10^-1,5*10^-1,6*10^-1,7*10^-1,8*10^-1,9*10^-1,10^0,10^1];
    %var=[2,4,6,8,10,12,14,16,18,20,22]; 
   
%     grad = [10^-2,10^-1,10^0,10^1,10^2,10^3];   %var for GradientThreshold
%     con = ["maximal","minimal"];                %var for Connectivity
    
%     grad = [10^-2,10^-1,10^0,10^1,10^2,10^3];   %var for GradientThreshold
%     con = ["exponential","quadratic"];          %var for ConductionMethod

%     niter = [2^1,2^2,2^3,2^4,2^5,2^6];          %var for NumberOfIterations
%     con = ["maximal","minimal"];                %var for Connectivity

%     niter = [2^1,2^2,2^3,2^4,2^5,2^6];          %var for NumberOfIterations
%     con = ["exponential","quadratic"];          %var for ConductionMethod

    showimg=zeros(256,256,12);
    img = imread(strcat(dirname, MyFolderInfo(j).name));
    img = im2double(img);
    showimg(:,:,1)=img;
    
    for i = 1:11
%     for i = 1:12
        %dst = imdiffusefilt(img,'GradientThreshold',var(i));
        %dst = imdiffusefilt(img,'GradientThreshold',0.1,'NumberOfIterations',var(i));
        %dst = imdiffusefilt(img,'GradientThreshold',0.1,'NumberOfIterations',5,'Connectivity',var(i));
        %dst = imdiffusefilt(img,'GradientThreshold',0.1,'NumberOfIterations',5,'Connectivity','maximal','ConductionMethod',var(i));
        %dst = imdiffusefilt(img,'GradientThreshold',grad(floor((i+1)/2)),'Connectivity',con(mod(i,2)+1));
        %dst = imdiffusefilt(img,'GradientThreshold',grad(floor((i+1)/2)),'ConductionMethod',con(mod(i,2)+1));
        %dst = imdiffusefilt(img,'GradientThreshold',0.1,'NumberOfIterations',niter(floor((i+1)/2)),'Connectivity',con(mod(i,2)+1));
        %dst = imdiffusefilt(img,'GradientThreshold',0.1,'NumberOfIterations',niter(floor((i+1)/2)),'ConductionMethod',con(mod(i,2)+1));
        dst = imdiffusefilt(img,'GradientThreshold',var(i),'NumberOfIterations',32);
        
        showimg(:,:,i+1)=dst;
    end
    
    figure;
    title('');
    for i = 1:12
        subplot(3,4,i), imshow(showimg(:,:,i));
        
        if i==1
           title('original');
        else
           title(strcat(num2str(var(i-1))));
        end

%         title(strcat(num2str(niter(floor((i+1)/2))),',',num2str(con(mod(i,2)+1))));
    end
     tightfig;
     saveas(gcf,strcat(savedirname,MyFolderInfo(j).name,'_Anisotropic_GradientThreshold_NumberOfIteration_32.png'));
end
