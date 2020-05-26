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
epsilon = 0;

%% Other Constants
k = 2*pi/lambda;  % Wave number
Freqency = 1/PixelSize;  % Frequency
Fxvector = 2*pi*linspace(-Freqency/2,Freqency/2,PixelNum);
Fyvector = 2*pi*linspace(-Freqency/2,Freqency/2,PixelNum);
[FxMat, FyMat] = meshgrid(Fxvector,Fyvector);
FMatSqure = 1./(FxMat.^2+FyMat.^2+epsilon);

%% Imread Figures
% I_minus = double(rgb2gray(imread('I_minus.tif')));
% I_focus = double(rgb2gray(imread('I_focus.tif')));
% I_plus = double(rgb2gray(imread('I_plus.tif')));
% I_minus = I2;
% I_focus = I1;
% I_plus = I3;
% I_plus1 = ones(PixelNum,PixelNum);
% I_minus1 = ones(PixelNum,PixelNum);
% I_focus1 = ones(PixelNum,PixelNum);
% I_plus1(5:256,5:256) = I_plus(1:252,1:252);
% I_minus1(1:252,1:252) = I_minus(5:256,5:256);
% I_plus = I_plus1;
% I_minus = I_minus1;
% I_focus = I_focus1;

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

PhaseA = Phase1-Incident_Phase;

