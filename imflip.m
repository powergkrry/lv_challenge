% function flipped = imflip(image, direction)
% 
% if size(image,3) > 1
%     grayimage = rgb2gray(image);
% else
%     grayimage = image;
% end
% 
% if direction == 'o'
%     flipped = grayimage;    
% elseif direction == 'h'
%     flipped = flip(grayimage,2); %horizontal
% elseif direction == 'v'
%     flipped = flip(grayimage,1); %vertical
% elseif direction == "hv"
%     flipped = flip(flip(grayimage,2),1); %horizontal and vertical
% elseif direction == "rr" %rotate right
%     flipped = imrotate(grayimage,-90);
% elseif direction == "rl" %rotate left
%     flipped = imrotate(grayimage,90);
% else
%     disp('Fatal Error : argument direction must be integer between 1~3');
% end
% 
% end


function flipped = imflip(image, direction)

if size(image,3) > 1
    grayimage = rgb2gray(image);
else
    grayimage = image;
end

if direction == 'o'
    flipped = grayimage;
elseif direction == 'rl1'
    flipped = imrotate(grayimage,90);
elseif direction == 'rl2'
    flipped = imrotate(grayimage,180);
elseif direction == 'rl3'
    flipped = imrotate(grayimage,270);
elseif direction == 'ov'
    flipped = flip(grayimage,1); %vertical
elseif direction == 'rl1v'
    flipped = flip(imrotate(grayimage,90),1);
elseif direction == 'rl2v'
    flipped = flip(imrotate(grayimage,180),1);
elseif direction == 'rl3v'
    flipped = flip(imrotate(grayimage,270),1);
else
    disp('Fatal Error : argument direction must be integer between 1~3');
end

end