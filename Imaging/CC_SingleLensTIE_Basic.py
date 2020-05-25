#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author:Chao Chen
Time:July 6,2019
"""

from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
from numpy import fft


# Import Image
# I have just added a single line, did you find it?
# it is amazing!
# yes I find it.

def show_im(im_array):
    plt.imshow(im_array, cmap='gray')
    plt.colorbar()        # 将numpy数组转会为matplotlib.Image对象，并设置colomap为'gray'
    plt.show()            # 显示所有matplotlib.Image对象图片


class ReadImage:
    def __init__(self, a, b):
        self.im_a = np.array(Image.open(a).convert('L'))
        self.im_b = np.array(Image.open(b).convert('L'))

    def convert(self):
        im_amplitude = (self.im_a/self.im_a.max()+4)/5
        im_phase = self.im_b/self.im_b.max()
        return im_amplitude, im_phase

    def size(self):
        am_size = shape(self.im_a)
        ph_size = shape(self.im_b)
        return am_size, ph_size

# Set Object's Wavefront


def get_object_wavefront(pixel_num, im_array_a, im_array_b):
    index1 = int((pixel_num - shape(im_array_a)[0]) / 2)
    index2 = int((pixel_num + shape(im_array_a)[0]) / 2)
    index3 = int((pixel_num - shape(im_array_a)[1]) / 2)
    index4 = int((pixel_num + shape(im_array_a)[1]) / 2)
    obj_amplitude = np.ones((pixel_num, pixel_num))
    obj_phase = np.zeros((pixel_num, pixel_num))
    obj_amplitude[index1:index2, index3:index4] = im_array_a
    obj_phase[index1:index2, index3:index4] = im_array_b
    return obj_amplitude*np.exp(1j*obj_phase)


def ccfft2(a):
    return fft.fftshift(fft.fft2(fft.ifftshift(a)))


def ccifft2(a):
    return fft.fftshift(fft.ifft2(fft.ifftshift(a)))


class Trans:
    def __init__(self, wavefront, lambda1, pixel_size):
        self.wavefront = wavefront
        self.lambda1 = lambda1
        self.pixel_size = pixel_size
        self.pixel_num = shape(wavefront)[0]

    def spacial_transfer(self, d):
        fx = linspace(-1/self.pixel_size/2, 1/self.pixel_size/2, self.pixel_num)
        fy = linspace(-1/self.pixel_size/2, 1/self.pixel_size/2, self.pixel_num)
        [fx_mat, fy_mat] = meshgrid(fx, fy)
        hz = np.exp(1j*2*np.pi/self.lambda1*d*np.sqrt(1-(self.lambda1*fx_mat)**2-(self.lambda1*fy_mat)**2))
        self.wavefront = ccifft2(ccfft2(self.wavefront)*hz)
        return self

    def lens_transfer(self, focus):
        total_size = self.pixel_num*self.pixel_size
        dx = linspace(-total_size/2, total_size/2, self.pixel_num)
        dy = linspace(-total_size/2, total_size/2, self.pixel_num)
        [dx_mat, dy_mat] = meshgrid(dx, dy)
        t = np.exp(-1j*2*np.pi/self.lambda1*(dx_mat**2+dy_mat**2)/(2*focus))
        self.wavefront = self.wavefront*t
        return self

    def get_amplitude(self):
        return np.abs(self.wavefront)

    def get_phase(self):
        return np.angle(self.wavefront)


def tie(a0, a1, d21, a2, d22, epsilon, wavelength, pixel_size, pixel_num):
    eps = 1e-6
    if isinstance(pixel_num, tuple) and len(pixel_num) == 2:
        pn1 = pixel_num[0]  # pixel_num 1
        pn2 = pixel_num[1]  # pixel_num 2
    else:
        pn1, pn2 = pixel_num, pixel_num
    sh1, sh2 = a0.shape
    index1 = int((sh1-pn1)/2)
    index2 = int((sh1+pn1)/2)
    index3 = int((sh2-pn2)/2)
    index4 = int((sh2+pn2)/2)
    a0 = a0[index1:index2, index3:index4]
    a1 = a1[index1:index2, index3:index4]
    a2 = a2[index1:index2, index3:index4]

    k = 2 * np.pi / wavelength
    fx = 2*np.pi*linspace(-1/pixel_size/2, 1/pixel_size/2, pn1)
    fy = 2*np.pi*linspace(-1/pixel_size/2, 1/pixel_size/2, pn2)
    [fx_mat, fy_mat] = meshgrid(fx, fy)

    temp = k * (a2 - a1) / (d22 - d21)
    freq_square = 1/(fx_mat**2+fy_mat**2+epsilon)
    temp1 = ccfft2(temp)*freq_square
    temp_x = ccifft2(temp1*fx_mat)/(a0+eps)
    temp_x = ccfft2(temp_x)*fx_mat
    temp_y = ccifft2(temp1*fy_mat)/(a0+eps)
    temp_y = ccfft2(temp_y)*fy_mat
    p_xy = ccifft2((temp_x+temp_y)*freq_square)
    return p_xy


def single_lens_tie(lambda1, pixel_num, pixel_size, d1, ff, d2, delta_z, epsilon,
                    im_name1='./source/aaaaa.jpg', im_name2='./source/bbbbb.jpg'):
    im = ReadImage(im_name1, im_name2)
    am_pic, ph_pic = im.convert()                                               # 读取的两幅图片
    wavefront0 = get_object_wavefront(pixel_num, am_pic, ph_pic)                # 获得初始波前
    out00, out01 = np.abs(wavefront0), np.angle(wavefront0)                     # 初始波前振幅、相位
    tr10 = Trans(wavefront0, lambda1, pixel_size)
    wavefront10 = tr10.spacial_transfer(d1).lens_transfer(ff).wavefront         # 传递至透镜后平面复振幅
    out1 = np.abs(wavefront10*conjugate(wavefront10))                           # 传递至透镜后平面强度分布
    wavefront20 = Trans(wavefront10, lambda1, pixel_size).spacial_transfer(d2).wavefront
    out20 = np.abs(wavefront20*conjugate(wavefront20))                          # 在焦强度
    wavefront21 = Trans(wavefront10, lambda1, pixel_size).spacial_transfer(d2-delta_z).wavefront
    out21 = np.abs(wavefront21*conjugate(wavefront21))                          # 欠焦强度
    wavefront22 = Trans(wavefront10, lambda1, pixel_size).spacial_transfer(d2+delta_z).wavefront
    out22 = np.abs(wavefront22*conjugate(wavefront22))                          # 离焦强度
    p_xy = tie(out20, out21, d2-delta_z, out22, d2+delta_z, epsilon, lambda1, pixel_size, im.size()[0])  # 恢复相位
    return out00, out01, out1, out20, out21, out22, np.real(p_xy)


def main(ff):
    lambda1 = 632.8e-9
    pixel_size = 7.4e-6
    pixel_num = 1024
    epsilon = 1e+8
    df = 2*ff
    df1 = df-0.002
    df2 = df+0.002
    im = ReadImage('aaaaa.jpg', 'bbbbb.jpg')
    im_wavefront = get_object_wavefront(pixel_num, im.convert()[0], im.convert()[1])
    tr = Trans(im_wavefront, lambda1, pixel_size)
    tr.spacial_transfer(df).lens_transfer(ff).spacial_transfer(df)
    wavefront0 = tr.wavefront
    a0 = np.real(wavefront0*np.conjugate(wavefront0))
    tr1 = Trans(im_wavefront, lambda1, pixel_size)
    tr1.spacial_transfer(df).lens_transfer(ff).spacial_transfer(df1)
    wavefront1 = tr1.wavefront
    a1 = np.real(wavefront1*np.conjugate(wavefront1))
    tr2 = Trans(im_wavefront, lambda1, pixel_size)
    tr2.spacial_transfer(df).lens_transfer(ff).spacial_transfer(df2)
    wavefront2 = tr2.wavefront
    a2 = np.real(wavefront2*np.conjugate(wavefront2))
    p_xy = tie(a0, a1, df1, a2, df2, epsilon, lambda1, pixel_size, im.size()[0])
    out = np.real(p_xy)
    figure()
    subplot(221)
    plt.imshow(a0, cmap='gray')
    plt.colorbar()
    subplot(222)
    plt.imshow(np.angle(wavefront0), cmap='gray')
    plt.colorbar()
    subplot(223)
    plt.imshow(a0, cmap='gray')
    plt.colorbar()
    subplot(224)
    plt.imshow(out, cmap='gray')
    plt.colorbar()
    show()


if __name__ == '__main__':
    # main(0.125)
    phase_rt = single_lens_tie(632.8e-9, 1024, 7.4e-6, 0.25, 0.125, 0.25, 0.002, 10**8)
    plt.figure(figsize=[12, 5.5])
    plt.subplot(241)
    plt.imshow(phase_rt[0], cmap='gray')
    # plt.colorbar()
    plt.subplot(242)
    plt.imshow(phase_rt[1], cmap='gray')
    # plt.colorbar()
    plt.subplot(243)
    plt.imshow(phase_rt[2], cmap='gray')
    # plt.colorbar()
    plt.subplot(244)
    plt.imshow(phase_rt[3], cmap='gray')
    # plt.colorbar()
    plt.subplot(245)
    plt.imshow(phase_rt[4], cmap='gray')
    # plt.colorbar()
    plt.subplot(246)
    plt.imshow(phase_rt[5], cmap='gray')
    # plt.colorbar()
    plt.subplot(247)
    plt.imshow(phase_rt[6], cmap='gray')
    # plt.colorbar()
    plt.tight_layout(pad=0.4, h_pad=0.2, w_pad=0.2)
    plt.show()
