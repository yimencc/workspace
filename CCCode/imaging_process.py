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

    @staticmethod
    def multi_img(plot_title=None,
                  ticks_trun_off=True,
                  colorbar_ticks_num=None,
                  **pic):
        """
        :param plot_title:
        :param ticks_trun_off:
        :param colorbar_ticks_num:
        :param pic:
        :return:

        Example:
            Check.multi_img(img_1=img_a, img_2=img_b)
        """

        len_img = len(pic)

        if 0 < len_img <= 2:
            row_num = 1
            colum_num = 2
            fig_size = [12, 5]
        elif 2 < len_img <= 4:
            row_num = 2
            colum_num = 2
            fig_size = [12, 9]
        elif 4 < len_img <= 8:
            row_num = 2
            colum_num = 4
            fig_size = [17, 8]
        else:
            raise Exception("Invalid Picture Number!", len_img)

        name_list = pic.keys()
        fig = plt.figure(figsize=fig_size)
        fig.suptitle(plot_title, fontsize=12)
        for n, _ in enumerate(name_list):
            plt.subplot(row_num, colum_num, n+1)
            plt.title(_)
            plt.imshow(pic[_], cmap="gray")
            if ticks_trun_off:
                plt.xticks([])
                plt.yticks([])
            plt.colorbar()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def wavefront(wave_front, name=None, extremun=False):
        amp, pha = [np.abs(wave_front), np.angle(wave_front)]
        if extremun:
            if name is not None:
                print(name, "amp min:", np.min(amp), name, "amp max:", np.max(amp))
                print(name, "pha min:", np.min(pha), name, "pha max:", np.max(pha))
            else:
                print("amp min:", np.min(amp), "amp max:", np.max(amp))
                print("pha min:", np.min(pha), "pha max:", np.max(pha))
        plt.figure(figsize=[12, 5])
        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.imshow([amp, pha][i], cmap="gray")
            plt.colorbar()
            if name is not None:
                plt.title(name + [" amplitude", " phase"][i])
        plt.tight_layout(pad=0.5, w_pad=0.4, h_pad=0.4)
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
        self.p_n = wavefront.shape

    def __getattribute__(self, item):
        if item == "wavefront_ft":
            return ft2(object.__getattribute__(self, "wavefront"))
        else:
            return object.__getattribute__(self, item)

    def __getattr__(self, item):
        if item == "amplitude":
            return abs(self.wavefront)
        elif item == "phase":
            return np.angle(self.wavefront)

    @classmethod
    def from_bioimage(cls, amplitude, phase, wavelength, pixel_size):
        return cls(amplitude*np.exp(1j*phase), wavelength, pixel_size)
    
    def _coordinate_fre(self, img_shape):
        fx = 2*np.pi*np.linspace(-1/self.p_s/2, 1/self.p_s/2, int(img_shape[-1]))
        fy = 2*np.pi*np.linspace(-1/self.p_s/2, 1/self.p_s/2, int(img_shape[0]))
        return np.meshgrid(fx, fy)  # [fx_mat, fy_mat]

    def _coordinate_spatial(self, img_shape):
        total_size = [self.p_s*shape for shape in img_shape]
        dx = np.linspace(-total_size[-1]/2, total_size[-1]/2, int(img_shape[-1]))
        dy = np.linspace(-total_size[0]/2, total_size[0]/2, int(img_shape[0]))
        return np.meshgrid(dx, dy)  # [dx_mat, dy_mat]

    def spatial_transfer(self, d):
        [fx_mat, fy_mat] = self._coordinate_fre(self.wavefront.shape)
        hz = np.exp(1j*self.k*d * np.sqrt(1-(fx_mat/self.k)**2-(fy_mat/self.k)**2))
        self.wavefront = ift2(ft2(self.wavefront) * hz)
        return self

    def lens_transmit(self, focus):
        lens_shape = self.wavefront.shape
        [dx_mat, dy_mat] = self._coordinate_spatial(lens_shape)
        t = np.exp(-1j*self.k*(dx_mat**2+dy_mat**2)/2/focus)
        self.wavefront = self.wavefront*t
        return self

    def lens_transfer(self, d1, focus, d2, **kwargs):
        """
        :param d1: distance between object wavefront and lens.
        :param focus: lens focus.
        :param kwargs: "pupil_plane" is used for checking lens.
        :param d2: distance between lens and imaging plane
        :param kwargs: {"obj_na": None, "working_distance": None}
        :return: self
        """
        # determine the lens size according pixel size and distance between object and lens
        k_max = np.pi/self.p_s
        cos_theta_max = k_max/self.k
        tan_theta = np.tan(np.pi-np.arccos(cos_theta_max))
        lens_pixel_num = np.array([pixel_num-np.min(self.p_n)+tan_theta*d1//self.p_s
                                  for pixel_num in self.p_n])
        [dx_mat, dy_mat] = self._coordinate_spatial(lens_pixel_num)
        t = np.exp(-1j*self.k * (dx_mat**2+dy_mat**2)/focus/2)

        # redefine the object with new coordinate
        target_pl = np.zeros(t.shape).astype(np.complex)
        target_c_x = target_pl.shape[-1]//2
        target_c_y = target_pl.shape[0]//2
        obj_w_half = self.p_n[-1]//2
        obj_h_half = self.p_n[0]//2
        idx_x1 = int(target_c_x - obj_w_half)
        idx_x2 = int(target_c_x - obj_w_half + self.p_n[-1])
        idx_y1 = int(target_c_y - obj_h_half)
        idx_y2 = int(target_c_y - obj_h_half + self.p_n[0])
        target_pl[idx_y1:idx_y2, idx_x1:idx_x2] = self.wavefront

        # transfer function
        [fx_mat, fy_mat] = self._coordinate_fre(lens_pixel_num)
        hz1 = np.exp(1j*self.k*d1*np.sqrt(1-(fx_mat/self.k)**2-(fy_mat/self.k)**2))
        hz2 = np.exp(1j*self.k*d2*np.sqrt(1-(fx_mat/self.k)**2-(fy_mat/self.k)**2))

        # propagate
        pre_lens_ft = ft2(target_pl) * hz1
        post_lens_ft = ft2(ift2(pre_lens_ft) * t)
        imaging_ft = post_lens_ft * hz2
        self.wavefront = ift2(imaging_ft)[idx_y1:idx_y2, idx_x1:idx_x2]
        return self

    @staticmethod
    def forward_propagate(wavefront, wavelength, pixel_size, d):
        wf_obj = Wavefront(wavefront, wavelength, pixel_size)
        return wf_obj.spatial_transfer(d).wavefront

    @staticmethod
    def lens_propagate(wavefront, d1, focus, d2, wavelength, pixel_size):
        wf_obj = Wavefront(wavefront, wavelength, pixel_size)
        return wf_obj.lens_transfer(d1=d1, focus=focus, d2=d2).wavefront

    @staticmethod
    def multi_focus_img(wavefront, delta_d, wavelength, pixel_size):
        wavefront_minus = Wavefront.forward_propagate(wavefront, wavelength, pixel_size, -delta_d).wavefront
        wavefront_plus = Wavefront.forward_propagate(wavefront, wavelength, pixel_size, delta_d).wavefront
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


def main():
    # import images
    imgs_path = "D:\\Workspace\\datasets\\open_image_val_standard"
    imgs_name_list = os.listdir(imgs_path)[0:2]
    imgs_fpath_list = [os.path.join(imgs_path, img_name) for img_name in imgs_name_list]
    amp_img = img_val_norm(resize(sio.imread(imgs_fpath_list[0]), (512, 512)), 0.9, 1)
    pha_img = img_val_norm(resize(sio.imread(imgs_fpath_list[1]), (512, 512)), 0.5, 1.5)

    wf_obj = Wavefront.from_bioimage(amp_img, pha_img, 6e-9, 5e-6)
    print(wf_obj.k)
    print(wf_obj.amplitude.shape)


if __name__ == '__main__':
    main()
