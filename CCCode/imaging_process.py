import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Im
import os
import math
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from skimage import io as sio
from skimage.transform import resize


def ft2(img):
    return fftshift(fft2(ifftshift(img)))


def ift2(img):
    return fftshift(ifft2(ifftshift(img)))


def img_val_norm(img, min_val=0., max_val=1.):
    img -= img.min()
    img /= img.max()
    if min_val != 0. or max_val != 1.:
        img *= max_val-min_val
        img += min_val
    return img


class ImageProcess:

    def __init__(self, init_img, convert_mode=True):
        if isinstance(init_img, str):
            self.image = Im.open(init_img)
        else:
            self.image = init_img
        if convert_mode:
            self.image = self.image.convert("L")
        if isinstance(self.image, np.ndarray):
            if len(self.image.shape) != 2:
                raise Exception("bad image shape:", self.image.shape)
            self.height, self.width = np.shape(self.image)
            self.type_ = "numpy.array"
        if isinstance(self.image, Im.Image):
            self.width, self.height = self.image.size
            self.type_ = "Image.Image"

    def image_crop(self, rotate=False, crop_rate=1.0, crop_box=None):
        # print("pre_w: ", self.width,
        #       "pre_h: ", self.height)
        # crop_box shape: [left, up, right, down]
        if rotate:
            self.image = self.image.transpose(Im.ROTATE_90)
            self.width, self.height = self.image.size
            # print("aft_w: ", self.width,
            #       "aft_h: ", self.height)
        if crop_box is None:
            if self.width >= self.height:
                left = math.floor((self.width-int(crop_rate*self.height))/2)
                right = math.floor((self.width+int(crop_rate*self.height))/2)
                up = math.floor((self.height-int(crop_rate*self.height))/2)
                down = math.floor((self.height+int(crop_rate*self.height))/2)
                crop_box = [left, up, right, down]
                # print(crop_box)
            else:
                left = math.floor((self.width-int(crop_rate*self.width))/2)
                right = math.floor((self.width+int(crop_rate*self.width))/2)
                up = math.floor((self.height-int(crop_rate*self.width))/2)
                down = math.floor((self.height+int(crop_rate*self.width))/2)
                crop_box = [left, up, right, down]
                # print(crop_box)

        if crop_box[2] > self.width and crop_box[3] > self.height:
            raise Exception("wrong crop box shape, cut out of range")
        if self.type_ == "numpy.array":
            self.image = self.image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            return self
        if self.type_ == "Image.Image":
            self.image = self.image.crop(box=crop_box)
            return self

    def image_scale(self, scale_size):
        if self.type_ == "numpy.array":
            self.image = Im.fromarray(self.image)
        if not isinstance(scale_size, tuple) or len(scale_size) != 2:
            raise Exception("bad scale size")
        self.image.thumbnail(scale_size)
        return self


class Check:

    def __init__(self, name=None):
        if name is not None:
            self.name = name

    def multi_img(self, plot_title=None,
                  show_extremum=False,
                  ticks_trun_off=True,
                  colorbar_ticks_num=None,
                  img_list=None,
                  name_list=None,
                  **pic):
        if img_list is not None:
            len_img = len(img_list)
        else:
            len_img = len(pic)
        if 0 < len_img <= 2:
            row_num = 1
            colum_num = 2
            fig_size = [8, 4]
        elif 2 < len_img <= 4:
            row_num = 2
            colum_num = 2
            fig_size = [8, 5.5]
        elif 4 < len_img <= 8:
            row_num = 2
            colum_num = 4
            fig_size = [12, 5.5]
        else:
            raise Exception("Invalid Picture Number!", len_img)
        if img_list is not None:
            n = 0
            plt.figure(figsize=fig_size)
            for _ in range(len_img):
                n += 1
                plt.subplot(row_num, colum_num, n)
                if name_list is not None:
                    plt.title(name_list[_])
                plt.imshow(img_list[_], cmap="gray")
                if ticks_trun_off:
                    plt.xticks([])
                    plt.yticks([])
                cbar = plt.colorbar()
                if colorbar_ticks_num is not None:
                    ticks_val_list = [np.min(img_list[_])+(np.max(img_list[_])
                                                           - np.min(img_list[_]))/(colorbar_ticks_num-1)*i
                                      for i in range(colorbar_ticks_num)]
                    cbar.set_ticks(ticks_val_list)
            plt.suptitle(plot_title)
            plt.tight_layout()
            plt.show()
            return
        name_list = pic.keys()
        if show_extremum:
            for _ in name_list:
                print(_, " min:", np.min(pic[_]), _, "max:", np.max(pic[_]))
        n = 0
        plt.figure(figsize=fig_size)
        for _ in name_list:
            n += 1
            plt.subplot(row_num, colum_num, n)
            plt.title(_)
            plt.imshow(pic[_], cmap="gray")
            plt.colorbar()
        plt.suptitle(plot_title)
        plt.tight_layout()
        plt.show()

    def wavefront(self, wave_front, name=None, extremun=True):
        amp = np.abs(wave_front)
        pha = np.angle(wave_front)
        if extremun:
            if name is not None:
                print(name, "amp min:", np.min(amp), name, "amp max:", np.max(amp))
                print(name, "pha min:", np.min(pha), name, "pha max:", np.max(pha))
            else:
                print("amp min:", np.min(amp), "amp max:", np.max(amp))
                print("pha min:", np.min(pha), "pha max:", np.max(pha))
        plt.figure(figsize=[12, 5.5])
        plt.subplot(121)
        plt.imshow(amp, cmap="gray")
        plt.colorbar()
        if name is not None:
            plt.title(name + " amplitude")
        plt.subplot(122)
        plt.imshow(pha, cmap="gray")
        plt.colorbar()
        if name is not None:
            plt.title(name + " phase")
        plt.tight_layout(pad=0.5, w_pad=0.4, h_pad=0.4)
        plt.show()

    def directly_propagate(self, input_list, name=None, extremun=True):
        if extremun:
            print("picture 0 min:", np.min(input_list[0]),
                  "picture 0 max:", np.max(input_list[0]))
            print("picture 1 min:", np.min(input_list[1]),
                  "picture 1 max:", np.max(input_list[1]))
            print("picture 2 min:", np.min(input_list[2]),
                  "picture 2 max:", np.max(input_list[2]))
        plt.figure(figsize=[12, 4])
        plt.subplot(131)
        plt.imshow(input_list[0], cmap="gray")
        plt.colorbar()
        if name is not None:
            plt.title(name + " infocus")
        plt.subplot(132)
        plt.imshow(input_list[1], cmap="gray")
        plt.colorbar()
        if name is not None:
            plt.title(name + " unfocus")
        plt.subplot(133)
        plt.imshow(input_list[2], cmap="gray")
        plt.colorbar()
        if name is not None:
            plt.title(name + " defocus")
        plt.tight_layout()
        plt.show()


class Wavefront:
    """
    toolkit class
    """
    def __init__(self, wavefront, wavelength, pixel_size):
        self.wavefront = wavefront
        self.wavelength = wavelength
        self.k = 2*np.pi/wavelength
        self.p_s = pixel_size  # pixel_size
        self.p_n = wavefront.shape[0:1] if wavefront.shape[0] == wavefront.shape[1] else wavefront.shape

    def spatial_transfer(self, d):
        fx = 2*np.pi*np.linspace(-1/self.p_s/2, 1/self.p_s/2, self.p_n[-1])
        fy = 2*np.pi*np.linspace(-1/self.p_s/2, 1/self.p_s/2, self.p_n[0])
        [fx_mat, fy_mat] = np.meshgrid(fx, fy)
        hz = np.exp(1j*self.k*d * np.sqrt(1-(fx_mat/self.k)**2-(fy_mat/self.k)**2))
        self.wavefront = ift2(ft2(self.wavefront) * hz)
        return self

    def lens_transfer(self, focus):
        total_s = (self.p_n[-1]*self.p_s, self.p_n[0]*self.p_s)
        dx = np.linspace(-total_s[-1]/2, total_s[-1]/2, self.p_n[-1])
        dy = np.linspace(-total_s[0]/2, total_s[0]/2, self.p_n[0])
        [dx_mat, dy_mat] = np.meshgrid(dx, dy)
        t = np.exp(-1j*self.k * (dx_mat**2+dy_mat**2)/focus/2)
        self.wavefront = self.wavefront*t
        return self

    def get_amplitude(self):
        return np.abs(self.wavefront)

    def get_phase(self):
        return np.angle(self.wavefront)

    @classmethod
    def from_bioimage(cls, amplitude, phase, wavelength, pixel_size):
        return cls(amplitude*np.exp(1j*phase), wavelength, pixel_size)

    @staticmethod
    def forward_propagate(wavefront, wavelength, pixel_size, d):
        wf_obj = Wavefront(wavefront, wavelength, pixel_size)
        return wf_obj.spatial_transfer(d).wavefront

    @staticmethod
    def multi_focus_img(wavefront, delta_d, wavelength, pixel_size):
        wavefront_minus = Wavefront.forward_propagate(wavefront, wavelength, pixel_size, -delta_d)
        wavefront_plus = Wavefront.forward_propagate(wavefront, wavelength, pixel_size, delta_d)
        i_focus = abs(wavefront*wavefront.conj())
        i_minus = abs(wavefront_minus*wavefront_minus.conj())
        i_plus = abs(wavefront_plus*wavefront_plus.conj())
        return i_focus, i_minus, i_plus


def tie_solution(i_focus, i_minus, i_plus, delta_d, wavelength, pixel_size, epsilon):
    # set constants
    assert i_focus.shape == i_minus.shape == i_plus.shape
    p_n = i_focus.shape
    fx = 2*np.pi*np.linspace(-1/pixel_size/2, 1/pixel_size/2, p_n[-1])
    fy = 2*np.pi*np.linspace(-1/pixel_size/2, 1/pixel_size/2, p_n[0])
    [fx_mat, fy_mat] = np.meshgrid(fx, fy)
    k = 2 * np.pi / wavelength

    # TIE solution
    derivative = k * (i_plus - i_minus) / (2 * delta_d)
    fmat_square = 1/(fx_mat**2+fy_mat**2+epsilon)
    temp1 = ft2(derivative)*fmat_square
    temp_x = ift2(temp1*fx_mat) / i_focus
    temp_x = ft2(temp_x)*fx_mat
    temp_y = ift2(temp1*fy_mat) / i_focus
    temp_y = ft2(temp_y)*fy_mat

    return ift2((temp_x+temp_y)*fmat_square).real


if __name__ == '__main__':
    # import images
    imgs_path = "D:\\Workspace\\datasets\\open_image_val_standard"
    imgs_name_list = os.listdir(imgs_path)[0:2]
    imgs_fpath_list = [os.path.join(imgs_path, img_name) for img_name in imgs_name_list]
    amp_img = img_val_norm(resize(sio.imread(imgs_fpath_list[0]), (512, 512)), 0.9, 1)
    pha_img = img_val_norm(resize(sio.imread(imgs_fpath_list[1]), (512, 512)), 0.5, 1.5)

    # Imaging
    EPSILON = 1e6
    DELTA_D = 1e-3
    WAVELENGTH = 500e-9
    PIXEL_SIZE = 5e-6
    i_focus, i_minus, i_plus = Wavefront.multi_focus_img(Wavefront.get_wf(amp_img, pha_img),
                                                         delta_d=DELTA_D,
                                                         wavelength=WAVELENGTH,
                                                         pixel_size=PIXEL_SIZE)
    p_xy = tie_solution(i_focus, i_minus, i_plus,
                        delta_d=DELTA_D,
                        wavelength=WAVELENGTH,
                        pixel_size=PIXEL_SIZE,
                        epsilon=EPSILON)

    plt.figure(figsize=[17, 5])
    plt.subplot(131)
    plt.imshow(pha_img, cmap="gray")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow((i_minus+i_plus)/2, cmap="gray")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(p_xy, cmap="gray")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
