import numpy as np
from numpy import fft


def pearson_correlation_coefficient_comput(img_1, img_2):
    mue_x = np.mean(img_1)
    mue_y = np.mean(img_2)
    cov = np.mean((img_1-mue_x)*(img_2-mue_y))
    sigma_x = np.sqrt(np.mean((img_1-mue_x)**2))
    sigma_y = np.sqrt(np.mean((img_2-mue_y)**2))
    rou_xy = cov/sigma_x/sigma_y
    return rou_xy


def mse_compute(img_1, img_2):
    return np.mean((img_1 - img_2) ** 2)


def img_scale(img, img_min=0, img_max=1):
    # equal to normalization
    prop = (img_max - img_min) / img_max
    img = img - np.min(img)
    img = img / np.max(img) * prop
    img = (img + (1 - prop)) * img_max
    return img


def MSEs(imgs_1, imgs_2):
    mse_list = []
    for i, j in zip(imgs_1, imgs_2):
        mse = mse_compute(i, j)
        mse_list.append(mse)
    return mse_list

def PCCs(imgs_1, imgs_2):
    pcc_list = []
    for i, j in zip(imgs_1, imgs_2):
        pcc = pearson_correlation_coefficient_comput(i, j)
        pcc_list.append(pcc)
    return pcc_list


def cc_fft2(x):
    return fft.fftshift(fft.fft2(fft.ifftshift(x)))


def cc_ifft2(x):
    return fft.fftshift(fft.ifft2(fft.ifftshift(x)))


if __name__ == '__main__':
    a = np.random.normal(loc=1.3, scale=1.5, size=(20, 128, 128))
    b = a + 3 + np.random.normal(loc=0, scale=0.3, size=(20, 128, 128))
    pccs_a_b = PCCs(a, b)
    print(len(pccs_a_b), np.mean(pccs_a_b))