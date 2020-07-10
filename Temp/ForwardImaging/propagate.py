import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import json


def ft2(x):
    return fftshift(fft2(ifftshift(x)))


def ift2(x):
    return fftshift(ifft2(ifftshift(x)))


def wavefront_show(wf, spectrum=0, wf_name=None):
    plt.figure(figsize=[12, 5])
    if spectrum:
        img_list = [np.log(abs(wf)+1), np.angle(wf), "{} amplitude".format(wf_name), "{} phase".format(wf_name)]
    else:
        img_list = [abs(wf), np.angle(wf), "{} amplitude".format(wf_name), "{} phase".format(wf_name)]
    for i in range(2):
        plt.subplot(1, 2, i+1)
        h = plt.imshow(img_list[i], cmap="gray")
        if wf_name is not None:
            plt.title(img_list[i+2])
        plt.colorbar(h)
    plt.tight_layout()
    plt.show()


def cartToNa(point_list_cart, z_offset=8):
    """functions for calcuating the NA of an LED on the quasi-dome based on it's index for the quasi-dome illuminator
    converts a list of cartesian points to numerical aperture (NA)
    笛卡尔坐标系 => 数值孔径坐标

    Args:
        point_list_cart: List of (x,y,z) positions relative to the sample (origin)
        z_offset : Optional, offset of LED array in z, mm

    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (Na_x, NA_y)
    """
    yz = np.sqrt(point_list_cart[:, 1] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)
    xz = np.sqrt(point_list_cart[:, 0] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)

    result = np.zeros((np.size(point_list_cart, 0), 2))
    result[:, 0] = np.sin(np.arctan(point_list_cart[:, 0] / yz))
    result[:, 1] = np.sin(np.arctan(point_list_cart[:, 1] / xz))
    return result


def loadLedPositonsFromJson(file_name, z_offset=8):
    """Function which loads LED positions from a json file
    Args:
        file_name: Location of file to load
        z_offset : Optional, offset of LED array in z, mm
        # micro : 'TE300B' or 'TE300A'
    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (x, y, z) in mm
    """
    json_data = open(file_name).read()
    data = json.loads(json_data)

    source_list_cart = np.zeros((len(data['led_list']), 3))
    x = [d['x'] for d in data['led_list']]
    y = [d['y'] for d in data['led_list']]
    z = [d['z'] for d in data['led_list']]

    source_list_cart[:, 0] = x
    source_list_cart[:, 1] = y
    source_list_cart[:, 2] = z

    source_list_na = cartToNa(source_list_cart, z_offset=z_offset)

    return source_list_na, source_list_cart


def get_led_na(led_index):
    source_list_na, source_list_cart = loadLedPositonsFromJson("../data/quasi_dome_design.json")
    angle = np.arcsin(np.sqrt(source_list_na[:, 0]**2 + source_list_na[:, 1]**2))
    return np.sin(angle[led_index - 1])


def get_led_nas(led_index):
    # return na_x, na_y for led_index
    source_list_na, source_list_cart = loadLedPositonsFromJson("../data/quasi_dome_design.json")
    return source_list_na[led_index - 1]
