import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import PIL.Image as Im
import PIL
import os
import math
from numpy.fft import fft2, fftshift, ifft2, ifftshift


def cc_fft2(img):
    return fftshift(fft2(ifftshift(img)))


def cc_ifft2(img):
    return fftshift(ifft2(ifftshift(img)))


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

    def to_numpy(self, min_val=0.0, max_val=1.0):
        image = np.asarray(self.image)
        image = img_scale(image, min_val, max_val)
        return image


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


class WavefrontPropagate:
    """
    toolkit class
    """

    def __init__(self, pixel_num, pixel_size, lambda_):
        self.wavefront = None
        self.pixel_size = pixel_size
        self.pixel_num = pixel_num
        self.lambda_ = lambda_
        self.total_size = self.pixel_num * self.pixel_size
        fx = np.linspace(-1/self.pixel_size/2, 1/self.pixel_size/2, self.pixel_num)
        self.fx_mat, self.fy_mat = np.meshgrid(fx, fx)
        dx = np.linspace(-self.total_size/2, self.total_size/2, self.pixel_num)
        self.dx_mat, self.dy_mat = np.meshgrid(dx, dx)

    def get_obj_wavefront(self, amp, pha,
                          pixel_num=0, output_wavefront=False):
        if not output_wavefront:
            init_size = amp.shape[0]
            idx1 = int((self.pixel_num-init_size)/2)
            idx2 = int((self.pixel_num+init_size)/2)
            amplitude_arr = np.zeros((self.pixel_num, self.pixel_num))
            phase_arr = np.zeros((self.pixel_num, self.pixel_num))
            amplitude_arr[idx1:idx2, idx1:idx2] = amp
            phase_arr[idx1:idx2, idx1:idx2] = pha
            self.wavefront = amplitude_arr*np.exp(1j*phase_arr)
            return self
        else:
            init_size = amp.shape[0]
            idx1 = int((pixel_num-init_size)/2)
            idx2 = int((pixel_num+init_size)/2)
            amplitude_arr = np.zeros((pixel_num, pixel_num))
            phase_arr = np.zeros((pixel_num, pixel_num))
            amplitude_arr[idx1:idx2, idx1:idx2] = amp
            phase_arr[idx1:idx2, idx1:idx2] = pha
            return amplitude_arr*np.exp(1j*phase_arr)

    def spatial_transfer(self, d, second_version=False):
        hz = np.exp(1j*2*np.pi/self.lambda_*d*np.sqrt(1-(self.lambda_*self.fx_mat)**2
                                                      - (self.lambda_*self.fy_mat)**2))
        if not second_version:
            self.wavefront = cc_ifft2(cc_fft2(self.wavefront)*hz)
            return self
        else:
            return cc_ifft2(cc_fft2(self.wavefront) * hz)

    def lens_transfer(self, focus):
        t = np.exp(-1j*np.pi/self.lambda_/focus*(self.dx_mat**2+self.dy_mat**2))
        self.wavefront = self.wavefront*t
        return self

    def directly_propagate(self,
                           delta_,
                           crop_size,
                           output_wavefront=False,
                           is_crop=True,
                           save_to_mat=False,
                           mat_save_fn=None):
        wavefront_mins = self.spatial_transfer(-delta_, True)
        wavefront_plus = self.spatial_transfer(delta_, True)
        out0 = np.abs(self.wavefront*np.conjugate(self.wavefront))
        out1 = np.abs(wavefront_mins*np.conjugate(wavefront_mins))
        out2 = np.abs(wavefront_plus*np.conjugate(wavefront_plus))
        if save_to_mat:
            if mat_save_fn is None:
                raise Exception("File name couldn't be None:", mat_save_fn)
            else:
                scipy.io.savemat(mat_save_fn,
                                 mdict={"infocus_wavefront": self.wavefront,
                                        "unfocus_wavefront": wavefront_mins,
                                        "defocus_wavefront": wavefront_plus})
        if not output_wavefront:
            if is_crop:
                idx1 = int((self.pixel_num-crop_size)/2)
                idx2 = int((self.pixel_num+crop_size)/2)
                out0_c = out0[idx1:idx2, idx1:idx2]
                out1_c = out1[idx1:idx2, idx1:idx2]
                out2_c = out2[idx1:idx2, idx1:idx2]
                return [out0_c, out1_c, out2_c]
            else:
                return [out0, out1, out2]
        else:
            return self.wavefront, wavefront_mins, wavefront_plus, [out0, out1, out2]


def tie_algorithm(input0, input1, input2, img_size,
                  delta, pixel_size, lambda_, epsilon, normalization=False):
    # set constants
    fx = 2*np.pi*np.linspace(-1/pixel_size/2, 1/pixel_size/2, img_size)
    [fx_mat, fy_mat] = np.meshgrid(fx, fx)
    k = 2*np.pi/lambda_

    # TIE solution
    derivative = k*(input2-input1)/(2*delta)
    fmat_square = 1/(fx_mat**2+fy_mat**2+epsilon)
    temp1 = cc_fft2(derivative)*fmat_square
    temp_x = cc_ifft2(temp1*fx_mat)/input0
    temp_x = cc_fft2(temp_x)*fx_mat
    temp_y = cc_ifft2(temp1*fy_mat)/input0
    temp_y = cc_fft2(temp_y)*fy_mat
    p_xy = cc_ifft2((temp_x+temp_y)*fmat_square)
    p_xy = np.real(p_xy)
    if normalization:
        p_xy = p_xy - (np.max(p_xy)-np.min(p_xy))/2
        p_xy = p_xy / np.max(p_xy)
    return p_xy


if __name__ == '__main__':
    file_path = "C:\\Users\\chenchao\\.keras\\datasets\\HAM10000_images"
    file_list = os.listdir(file_path)[:100]
    img_obj = ImageProcess(os.path.join(file_path, file_list[0]))
    img_obj.image_crop()
    plt.imshow(np.asarray(img_obj.image), cmap="gray")
    plt.show()
