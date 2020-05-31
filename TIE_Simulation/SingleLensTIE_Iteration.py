#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author:Chao Chen
Time:July 6,2019
"""

import SingleLensTIE_Basic as Cs
import numpy as np
import PIL.Image as Image
import time
import matplotlib.pyplot as plt


Lambda = 632.8e-9
k = 2*np.pi/Lambda
f = np.arange(40, 160, 20)*1e-3  # length:6
PixelSize = np.arange(3.8e-6, 8.2e-6, 0.4e-6)  # length:11
PixelNum = np.arange(700, 1500, 100)  # length:8
epsilon = np.arange(1e8, 9e8, 2e8)  # length:4
# n = 0
# tick0 = time.time()
# ticks = tick0
init_im = Cs.ReadImage('aaaaa.jpg', 'bbbbb.jpg').convert()
# init_pic = (np.angle(Cs.get_object_wavefront(600, init_im[0], init_im[1]))+0.25)/1.25
# print(n, ticks-tick0)
# loss_a = {}
'''
best parameter:'PicNum=1200, PicSize=7.8um, f=40mm, eps=3e8'
'''

# for ff in f:
#     for pic_size in PixelSize:
#         for pic_num in PixelNum:
#             for eps in epsilon:
set_im = Cs.single_lens_tie(632.8e-9, 1200, 7.8e-6, 2*0.4, 0.4, 2*0.4, 0.002, 3e8)[6]
set_im = np.rot90(set_im, 2)
# set_im = set_im-np.min(set_im)
# set_im = set_im/np.max(set_im)
# set_im = (set_im+0.25)/1.25
# loss_pic = (init_pic-set_im)**2
# loss_r = loss_pic/(init_pic**2)
# loss = np.sum(loss_r)
# para = 'PicNum=%d, PicSize=%.1fum, f=%.fmm, eps=%.fe8' % (1200, 7.8e-6/1e-6, 4/1e-3, 3e8/1e8)
# loss_a[para] = loss
# n = n+1
# temp_t = ticks
# ticks = time.time()
# str_1 = para+', spend time: %.2f S,' % (ticks-temp_t)
# str_2 = 'total time: %.1fS, ' % (ticks-tick0) + 'loss: %.f' % loss
# print(n, str_1, str_2)
plt.imshow(set_im, cmap='gray')
plt.colorbar()
plt.show()

