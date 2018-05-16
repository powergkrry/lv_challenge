function flipped = imflip(image, direction)

if size(image,3) > 1
    grayimage = rgb2gray(image);
else
    grayimage = image;
end

if direction == 'o'
    flipped = grayimage;    
elseif direction == 'h'
    flipped = flip(grayimage,2); %horizontal
elseif direction == 'v'
    flipped = flip(grayimage,1); %vertical
elseif direction == "hv"
    flipped = flip(flip(grayimage,2),1); %horizontal and vertical
else
    disp('Fatal Error : argument direction must be integer between 1~3');
end

end
