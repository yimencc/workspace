#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author:Chao Chen
Time:July 6,2019
"""

from pylab import *
import matplotlib.pyplot as plt
from skimage import io as sio
from skimage.transform import resize
import os
import sys
sys.path.append("D:\\Workspace\\project\\CCCode")
from imaging_process import ft2, ift2, Wavefront, tie_solution, img_val_norm


def single_lens_tie(wavelength, pixel_num, pixel_size,
                    d1, ff, d2, delta_d, epsilon):
    # import images
    imgs_path = "D:\\Workspace\\datasets\\open_image_val_standard"
    imgs_name_list = os.listdir(imgs_path)[0:2]
    imgs_fpath_list = [os.path.join(imgs_path, img_name) for img_name in imgs_name_list]
    amp_img = img_val_norm(resize(sio.imread(imgs_fpath_list[0]), (pixel_num, pixel_num)), 0.8, 1)
    pha_img = img_val_norm(resize(sio.imread(imgs_fpath_list[1]), (pixel_num, pixel_num)), 0.2, 1.5)

    wf_obj = Wavefront.from_bioimage(amp_img, pha_img, wavelength, pixel_size)
    wf_focus = wf_obj.spatial_transfer(d1).lens_transfer(ff).spatial_transfer(d2).wavefront
    i_focus, i_minus, i_plus = Wavefront.multi_focus_img(wf_focus, delta_d, wavelength, pixel_size)

    p_xy = tie_solution(i_focus, i_minus, i_plus, delta_d, wavelength, pixel_size, epsilon)
    return i_focus, i_minus, i_plus, p_xy


def main():
    # main(0.125)
    WAVELENGTH = 500e-9
    PIXEL_SIZE = 5e-6
    DELTA_D = 1e-3
    obj_len_d = 0.08
    f = obj_len_d/2
    i_focus, i_minus, i_plus, p_xy = single_lens_tie(WAVELENGTH, 512, PIXEL_SIZE,
                                                     obj_len_d, f, obj_len_d, DELTA_D, 1e9)

    plt.figure(figsize=[12, 8])
    plt.subplot(221)
    plt.imshow(i_focus, cmap="gray")
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(i_minus, cmap="gray")
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(i_plus, cmap="gray")
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(p_xy, cmap="gray")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
