#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author:Chao Chen
Time:July 6,2019
"""

import numpy as np
import os
import sys
import skimage.io as sio
from skimage.transform import resize
import matplotlib.pyplot as plt
from scipy import io as spio

sys.path.append("D:\\Workspace\\git_proj\\CCCode")
from imaging_process import Check, Wavefront, img_val_norm
from cc_math import mse_compute


def iterative_based_tie(img_list, z_list, wavelength, pixel_size, n_iter=10):
    z_len = len(img_list)
    cen_idx = int((len(img_list)-1)//2)
    principle_img = img_list[cen_idx]

    estimate_phase = np.ones(principle_img.shape)
    estimate_wf = np.sqrt(principle_img)*np.exp(1j*estimate_phase)

    # iteration
    estimate_updated = estimate_wf
    Check.wavefront(estimate_updated, name="the first")
    idx_list_part1 = list(range(z_len))[cen_idx:]
    idx_list_part2 = list(range(z_len))[:z_len-1]
    idx_list_part2.sort(reverse=True)
    idx_list_part3 = list(range(z_len))[1:cen_idx+1]
    idx_list = idx_list_part1 + idx_list_part2 + idx_list_part3
    print(idx_list)

    es_list = []
    for n in range(n_iter):
        print("Iteration -- ", n)
        for i in range(len(idx_list)-1):
            delta_z = z_list[idx_list[i+1]] - z_list[idx_list[i]]
            estimate_propagated = Wavefront.forward_propagate(estimate_updated, wavelength, pixel_size, delta_z)
            estimate_updated = np.sqrt(img_list[idx_list[i+1]])*np.exp(1j*np.angle(estimate_propagated))
            # print("delta_z: {}".format(delta_z))
            # Check.wavefront(estimate_updated, name=str(z_list[idx_list[i+1]]))
        es_list.append(estimate_updated)
    return es_list


def post_code1():
    WAVELENGTH = 632.8e-9
    PIXEL_SIZE = 5e-6

    imgs_path = "D:\\Workspace\\datasets\\open_image_val_standard"
    img_fullname_list = [os.path.join(imgs_path, name) for name in os.listdir(imgs_path)[:2]]
    [amp_img, pha_img] = [resize(sio.imread(fullname), (512, 512)) for fullname in img_fullname_list]
    amp_img = img_val_norm(amp_img, 0.8, 1.)
    pha_img = img_val_norm(pha_img, 0.2, 1.5)

    wf_obj = Wavefront.from_bioimage(amp_img, pha_img, WAVELENGTH, PIXEL_SIZE)
    wf_list = []
    for i, d in enumerate([-5e-3, -2e-3, 2e-3, 5e-3]):
        wf_list.append(Wavefront.forward_propagate(wf_obj.wavefront, WAVELENGTH, PIXEL_SIZE, d))
    intensity_list = [abs(wf*wf.conj()) for wf in wf_list]
    intensity_list.insert(2, abs(wf_obj.wavefront*wf_obj.wavefront.conj()))

    # plt.figure(figsize=[17, 8])
    # for i, img in enumerate(intensity_list):
    #     plt.subplot(2, 3, i+1)
    #     plt.imshow(img, cmap="gray")
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.tight_layout()
    # plt.show()
    e_u = iterative_based_tie(intensity_list, [-5e-3, -2e-3, 2e-3, 5e-3], WAVELENGTH, PIXEL_SIZE, 500)
    mse = []
    for estimate in e_u:
        estimate_phase = np.angle(estimate)
        mse.append(mse_compute(pha_img-pha_img.min(), estimate_phase-estimate_phase.min()))
    plt.plot(mse)
    plt.show()
    Check.wavefront(e_u[-1])


def main():
    multi_focus_imgs = spio.loadmat("..\\source\\l.mat")
    l3_mat = spio.loadmat("..\\source\\r1.mat")
    l4_mat = spio.loadmat("..\\source\\r2.mat")
    l2 = multi_focus_imgs["Il"]
    l1 = multi_focus_imgs["Ir"]
    r1 = l3_mat["Il"]
    r2 = l4_mat["Il"]

    img_list = [l2, l1, r1, r2]
    z_list = [24e-6, 12e-6, -12e-6, -24e-6]
    Check.multi_img(l2=l2, l1=l1, r1=r1, r2=r2)
    e_u = iterative_based_tie(img_list, z_list, 532e-9, 5e-7, 200)
    Check.wavefront(e_u[-1], name="the last ")
    out = Wavefront.forward_propagate(e_u[-1], 532e-9, 5e-7, -12e-6)
    Check.wavefront(out)


if __name__ == '__main__':
    main()
