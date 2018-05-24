dirname_save1=["poriginal_A_png", "ooriginal_A_png", "ioriginal_A_png"];
dirname_save2=["/hoem04/powergkrry/lv_challenge/data/Anisotropic/poriginal_A_png/",...
    "/hoem04/powergkrry/lv_challenge/data/Anisotropic/ooriginal_A_png/",...
    "/hoem04/powergkrry/lv_challenge/data/Anisotropic/ioriginal_A_png/"];
dirname_load=["/hoem04/powergkrry/lv_challenge/data/poriginal_png/",...
    "/hoem04/powergkrry/lv_challenge/data/ooriginal_png/",...
    "/hoem04/powergkrry/lv_challenge/data/ioriginal_png/"];
diffusion_gradient=[0.01, 0.05, 0.1, 1];
diffusion_conduction=["exponential","quadratic"];
flip=["o", "h", "v", "hv"];

cd /hoem04/powergkrry/lv_challenge/data/
mkdir Anisotropic

for i=1:length(dirname_save1)
    cd /hoem04/powergkrry/lv_challenge/data/Anisotropic/
    mkdir(dirname_save1(i))
    cd(dirname_save2(i))
    
    original_names = dir(dirname_load(i));
    
    for j=3:length(original_names)
        img = imread(char(strcat(dirname_load(i), original_names(j).name)));
        img = im2double(img);
        
        for k=1:length(diffusion_gradient)
            for l=1:length(diffusion_conduction)
                for m=1:length(flip)
                    mkdir_name = strcat(num2str(diffusion_gradient(k)), "_", diffusion_conduction(l), "_", flip(m));
                    if j == 3
                        mkdir(mkdir_name)
                    end         
                    cd(strcat(dirname_save2(i), mkdir_name))
                    
                    dst = imdiffusefilt(img,'GradientThreshold',diffusion_gradient(k),'NumberOfIterations',8, 'ConductionMethod',diffusion_conduction(l));
                    dst = imflip(dst, flip(m));
                    
                    dst = im2uint8(dst);
                    imwrite(dst, strcat('./', int2str(j-2), '.png'));
                    
                    cd ..
                end
            end
        end
    end
end