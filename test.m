%% This file demonstrates the Split Bregman method for Total Variation denoising
%
%   SB_ATV.m  Split Bregman Anisotropic Total Variation Denoising
%   SB_ITV.m  Split Bregman Isotropic Total Variation Denoising
%
% Benjamin Trémoulhéac
% University College London
% b.tremoulheac@cs.ucl.ac.uk
% April 2012

clc; clear all;
close all;

N = 256; n = N^2;
f = im2double(imread('100','png'));
g = f(:) + 0.09*max(f(:))*randn(n,1);

mu = 0.02;

g_denoise_atv = SB_ATV(f,mu);

fprintf('ATV Rel.Err = %g\n',norm(g_denoise_atv(:) - f(:)) / norm(f(:)));
fprintf('ITV Rel.Err = %g\n',norm(g_denoise_atv(:) - f(:)) / norm(f(:)));

a=reshape(g_denoise_atv,N,N);
%a(:,128)

b=f-reshape(g_denoise_atv,N,N);
%b(:,128)

c=im2uint8(b);
c(:,128)

%g_denoise_atv = im2uint8(reshape(g_denoise_atv,N,N));
%imwrite(g_denoise_atv, './100_0.02.png');

% figure; colormap gray;
% subplot(221); imagesc(f); axis image; title('Original');
% subplot(222); imagesc(reshape(g,N,N)); axis image; title('Noisy');
% subplot(223); imagesc(reshape(g_denoise_atv,N,N)); axis image; ;
% title('Anisotropic TV denoising');
% subplot(224); imagesc(reshape(g_denoise_atv,N,N)); axis image; 
% title('Isotropic TV denoising');
