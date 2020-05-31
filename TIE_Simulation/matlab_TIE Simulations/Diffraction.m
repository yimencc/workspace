% TIE
% Basic


%% Start
clear all;
close all;
clc;

%% Input Constants
lambda = 632.8*10^(-9);  % Wavelength
PixelNum = 256;  % Pixel Number
PixelSize = 6*10^(-6);  % Pixel Size
DeltaDistance = 0.5*10^(-3);  % Defocusing Distance
RI = 1;  % Refractive Index

%% Other Constants
k = 2*pi/lambda;  % Wave number
Freqency = 1/PixelSize;  % Frequency
Fxvector = linspace(-Freqency/2,Freqency/2,PixelNum);
Fyvector = linspace(-Freqency/2,Freqency/2,PixelNum);
[FxMat, FyMat] = meshgrid(Fxvector,Fyvector);

%% Imread Figures
I_seo = double(rgb2gray(imread('Seo.jpg')));
I_ishikawa = double(rgb2gray(imread('Ishikawa.jpg')));
I_ishikawa = I_ishikawa(11:266,100:355);
I_seo = I_seo(11:266,31:286);
I_ishikawa = I_ishikawa/max(max(I_ishikawa));
I_seo = I_seo/max(max(I_seo));

%% Propagtaion
Amplitude = (1+I_ishikawa/5).^0.5;
% Amplitude = ones(PixelNum,PixelNum);

Incident_Amplitude = ones(PixelNum,PixelNum);
Dx = linspace(-3,3,PixelNum);
Dy = linspace(-2,2,PixelNum);
[Dxx, Dyy] = meshgrid(Dx,Dy);
Incident_Phase = 0*(Dxx.^2+Dyy.^2);
% Incident_Phase = 0;

Phase = I_seo*1.5;
d1 = DeltaDistance;
d2 = -DeltaDistance;
Wavefront = Amplitude.*exp(1i*(Phase+Incident_Phase));

Hz1 = exp((1i*2*pi*d1*RI/lambda)*(1-(lambda.*FxMat/RI).^2-(lambda.*FyMat/RI).^2).^0.5);
Hz2 = exp((1i*2*pi*d2*RI/lambda)*(1-(lambda.*FxMat/RI).^2-(lambda.*FyMat/RI).^2).^0.5);

Wavefront1 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Wavefront))).*Hz1)));
Wavefront2 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Wavefront))).*Hz2)));

I_focus = Wavefront.*conj(Wavefront);
I_plus = Wavefront1.*conj(Wavefront1);
I_minus = Wavefront2.*conj(Wavefront2);






