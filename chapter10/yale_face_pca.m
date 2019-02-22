clear,clc
% 
folder = 'yalefaces\';
impaths = dir(fullfile(folder,'*.bmp'));
for i=1:length(impaths)
    im = imread( fullfile(folder, impaths(i).name) );
    coeff = pca(double(im));
    % 
    pca_info = double(im) * (coeff*coeff');
    imshow(pca_info,[])
end
