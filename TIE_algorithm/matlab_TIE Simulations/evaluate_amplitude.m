%% Start
clear all;
close all;
clc;

lambda = 632.8*10^(-9);  % Wavelength
PixelNum = 256;  % Pixel Number
PixelSize = 6*10^(-6);  % Pixel Size
DeltaDistance = 0.5*10^(-3);  % Defocusing Distance
RI = 1;  % Refractive Index

k = 2*pi/lambda;  % Wave number
Freqency = 1/PixelSize;  % Frequency
Fxvector = linspace(-Freqency/2,Freqency/2,PixelNum);
Fyvector = linspace(-Freqency/2,Freqency/2,PixelNum);
[FxMat, FyMat] = meshgrid(Fxvector,Fyvector);

I_seo = double(rgb2gray(imread('Seo.jpg')));
I_ishikawa = double(rgb2gray(imread('Ishikawa.jpg')));
I_ishikawa = I_ishikawa(11:266,100:355);
I_seo = I_seo(11:266,31:286);
I_ishikawa = I_ishikawa-min(min(I_ishikawa));
I_ishikawa = I_ishikawa/max(max(I_ishikawa));
I_seo = I_seo-min(min(I_seo));
I_seo = I_seo/max(max(I_seo));

d1 = DeltaDistance;
d2 = -DeltaDistance;
Hz1 = exp((1i*2*pi*d1*RI/lambda)*(1-(lambda.*FxMat/RI).^2-(lambda.*FyMat/RI).^2).^0.5);
Hz2 = exp((1i*2*pi*d2*RI/lambda)*(1-(lambda.*FxMat/RI).^2-(lambda.*FyMat/RI).^2).^0.5);

% Incident_Amplitude = ones(PixelNum,PixelNum);
% Dx = linspace(-3,3,PixelNum);
% Dy = linspace(-2,2,PixelNum);
% [Dxx, Dyy] = meshgrid(Dx,Dy);
% Incident_Phase = 0*(Dxx.^2+Dyy.^2);
% Incident_Phase = 0;
coef = zeros(9,1);
Phase_imgs = zeros(9, 256, 256);

for n = 1:9
min_val = 0.1*n;
Amplitude = I_ishikawa*(1-min_val)+min_val;
Phase = I_seo*1.5;
% Wavefront = Amplitude.*exp(1i*(Phase+Incident_Phase));
Wavefront = Amplitude.*exp(1i*Phase);

Wavefront1 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Wavefront))).*Hz1)));
Wavefront2 = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(Wavefront))).*Hz2)));

I_focus = Wavefront.*conj(Wavefront);
I_plus = Wavefront1.*conj(Wavefront1);
I_minus = Wavefront2.*conj(Wavefront2);

%% Other Constants
epsilon = 0;
k = 2*pi/lambda;  % Wave number
Freqency = 1/PixelSize;  % Frequency
Fxvector = 2*pi*linspace(-Freqency/2,Freqency/2,PixelNum);
Fyvector = 2*pi*linspace(-Freqency/2,Freqency/2,PixelNum);
[FxMat, FyMat] = meshgrid(Fxvector,Fyvector);
FMatSqure = 1./(FxMat.^2+FyMat.^2+epsilon);

%% STIE
% Derivative = k*(I_plus-I_minus)/(2*DeltaDistance);
% Part = fftshift(ifft2(ifftshift(Derivative))).*FMatSqure;
% Phase = real(fftshift(ifft2(ifftshift(Part))))./I_focus;

%% PTIE
Derivative = k*(I_plus-I_minus)/(2*DeltaDistance);
Part1X = FxMat.*FMatSqure.*fftshift(fft2(ifftshift(Derivative)));
Part1Y = FyMat.*FMatSqure.*fftshift(fft2(ifftshift(Derivative)));
Part2X = fftshift(ifft2(ifftshift(Part1X)))./I_focus;
Part2Y = fftshift(ifft2(ifftshift(Part1Y)))./I_focus;
Part3X = fftshift(fft2(ifftshift(Part2X))).*FxMat;
Part3Y = fftshift(fft2(ifftshift(Part2Y))).*FyMat;
Part4 = Part3X+Part3Y;
PhaseX = real(fftshift(ifft2(ifftshift(Part4.*FMatSqure))));
Phase1 = PhaseX-min(min(PhaseX));

mue_x = mean(mean(Phase1));
mue_y = mean(mean(Phase));
cov = mean(mean((Phase1-mue_x).*(Phase-mue_y)));
sigma_x = (mean(mean((Phase1-mue_x).^2))).^0.5;
sigma_y = (mean(mean((Phase-mue_y).^2))).^0.5;
rou_xy = cov/sigma_x/sigma_y;
coef(n) = rou_xy;
Phase_imgs(n, :, :) = Phase1;
end
