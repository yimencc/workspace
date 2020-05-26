import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from ForwardImaging.propagate import ft2, ift2, loadLedPositonsFromJson, wavefront_show, get_led_na


def forward_propagate(wavefront, d, wavelength, pixel_num, pixel_size):
    # Propagate algorithm suitable for single image processing
    k = 2 * np.pi / wavelength
    kx = 2*np.pi*np.linspace(-1/pixel_size/2, 1/pixel_size/2, pixel_num)
    ky = 2*np.pi*np.linspace(-1/pixel_size/2, 1/pixel_size/2, pixel_num)
    [kx_mat, ky_mat] = np.meshgrid(kx, ky)
    if np.sqrt(2*(2*np.pi/pixel_size/2)**2) >= k:
        cut_frequency_limit = np.array(((kx_mat/k)**2 + (ky_mat/k)**2 < 1))  # bool array
        cos_term = (kx_mat/k)**2+(ky_mat/k)**2
        cos_term[~cut_frequency_limit] = 0
        h = np.exp(1j*k*d*np.sqrt(1-cos_term))
        h[~cut_frequency_limit] = 0
    else:
        h = np.exp(1j*k*d*np.sqrt(1-(kx_mat/k)**2-(ky_mat/k)**2))
    spectrum_0 = ft2(wavefront)
    wf_1 = ift2(spectrum_0*h)
    return wf_1


class ValSpectrum:
    # Is it possible to achieve the purpose of simulating multi-light source lighting by
    # propagating the LED light source located in the lighting plane to the object plane?

    def __init__(self, **kwargs):
        self.p = {"amp_img": "a.jpg", "pha_img": "b.jpg",
                  "pixel_size": 7.2e-6, "wavelength": 632.8e-9,
                  "pixel_num": 512, "magnification": 20, "obj_na": 0.5}
        for key in self.p:
            if key in kwargs:
                self.p[key] = kwargs[key]

        self.k = 2 * np.pi / self.p["wavelength"]
        amp_img = resize(io.imread(self.p["amp_img"]),
                         (self.p["pixel_num"], self.p["pixel_num"]))
        pha_img = resize(io.imread(self.p["amp_img"]),
                         (self.p["pixel_num"], self.p["pixel_num"]))
        self.amp_img = (amp_img - amp_img.min()) / (amp_img - amp_img.min()).max()
        self.pha_img = np.pi * (pha_img - pha_img.min()) / (pha_img - pha_img.min()).max()
        self.wf_init = self.amp_img * np.exp(1j * self.pha_img)

        self.x_mat, self.y_mat = self._spatial_coord()
        self.delta_k, (self.kx_mat, self.ky_mat) = self._frequency_coord()
        self._ctf = (self.kx_mat**2 + self.ky_mat**2) < (self.p["obj_na"]*self.k)**2

    def _spatial_coord(self):
        self.dx = self.p["pixel_size"] / self.p["magnification"]
        self.dy = self.p["pixel_size"] / self.p["magnification"]
        x = np.linspace(-self.p["pixel_num"]*self.dx/2, self.p["pixel_num"]*self.dx/2, self.p["pixel_num"])
        y = -np.linspace(-self.p["pixel_num"]*self.dy/2, self.p["pixel_num"]*self.dy/2, self.p["pixel_num"])
        return np.meshgrid(x, y)

    def _frequency_coord(self):
        # return delta_k, (kx_mat, ky_mat)
        kx = 2*np.pi*np.linspace(-1/self.dx/2, 1/self.dx/2, self.p["pixel_num"])
        ky = 2*np.pi*np.linspace(-1/self.dx/2, 1/self.dx/2, self.p["pixel_num"])
        return 2*np.pi/self.dx/self.p["pixel_num"], np.meshgrid(kx, ky)

    def tilt_illu_formation(self, x_na, y_na):
        # algorithm is performs  bright field microscopy when (x_na**2 + y_na**2) < obj_na**2
        kx_na, ky_na = self.k * x_na, self.k * y_na
        factor_tilt = np.exp(1j*(self.x_mat*kx_na + self.y_mat*ky_na))

        print("for the tilt illu formation:\nx_shift: ", kx_na / self.delta_k,
              "y_shift: ", ky_na / self.delta_k)

        self.ft_init = ft2(factor_tilt * self.wf_init)
        self.ft_init[~self._ctf] = 0+0j

        # compute the center position of tilt illuminated spectrum
        (p_arr_y, p_arr_x) = np.where(abs(self.ft_init) == np.max(abs(self.ft_init)))
        print("tilt illuminated spectrum center position: ({}, {})".format(p_arr_y[0], p_arr_x[0]))

        # compute the real shift value of spectrum
        print("the real shift value of spectrum is: ({}, {})".format(p_arr_y[0]-256, p_arr_x[0]-256))
        return self.ft_init

    def spectrum_offset_formation(self, x_na, y_na):
        self.ft_init = ft2(self.wf_init)

        kx_na, ky_na = self.k * x_na, self.k * y_na
        kx_offset = kx_na / self.delta_k
        ky_offset = ky_na / self.delta_k
        print("the na mod", abs(kx_offset) % 1, abs(ky_offset) % 1)
        if 0.4 < (abs(kx_offset) % 1) < 0.5 or 0.4 < (abs(ky_offset) % 1) < 0.5:
            print("Waning!!")
            # raise Exception("NA shift value might be unavailable")
        kxc = self.ft_init.shape[-1] // 2 - np.round(kx_offset)
        kyc = self.ft_init.shape[-2] // 2 + np.round(ky_offset)
        print("\nfor the spectrum offset formation: \nx_shift: {}, y_shift: {}".format(kx_offset, ky_offset))
        print("kx center: {}, ky center: {}".format(kxc, kyc))

        core_w = int(self.k * 0.45 // self.delta_k)
        self.ft_shifted = np.zeros((512, 512)).astype(np.complex)
        spectrum_core = self.ft_init[int(kyc-core_w):int(kyc+core_w), int(kxc-core_w):int(kxc+core_w)]

        self.ft_shifted[int(256-core_w):int(256+core_w), int(256-core_w):int(256+core_w)] = spectrum_core
        self.ft_shifted[~self._ctf] = 0+0j

        # compute the real center of spectrum
        (p_arr_y, p_arr_x) = np.where(abs(self.ft_shifted) == np.max(abs(self.ft_shifted)))
        print("frequency shift spectrum center position: ({}, {})".format(p_arr_y[0], p_arr_x[0]))
        print("the real shift value of spectrum is: ({}, {})".format(p_arr_y[0]-256, p_arr_x[0]-256))
        return self.ft_shifted


class Illuminate:

    def __init__(self, **kwargs):
        # the default led pattern is quasi-dome, led index is 119th
        self.para = {"wavelength": 632.8e-9,
                     "pixel_size": 6.328e-6*2,
                     "magnification": 20,
                     "obj_na": 0.5,
                     "led_pattern": {"name": "quasi-dome",
                                     "idx": 119}
                     }

        for key in self.para.keys():
            if key in kwargs.keys():
                self.para[key] = kwargs[key]

        self.k = np.pi / self.para["wavelength"]
        self.d = self.para["pixel_size"] / self.para["magnification"]

        led_pattern = self.para["led_pattern"]
        if led_pattern["name"] is "quasi-dome":
            source_list_na, source_list_cart = loadLedPositonsFromJson("../data/quasi_dome_design.json")
            self.illu_nas = source_list_na[led_pattern["idx"] - 1]  # (n_led, 2): [led idx, (na_x, na_y)]
        elif led_pattern["name"] is "led_array":
            self.illu_nas = None
            Exception("led array pattern is not built")
        else:
            self.illu_nas = None
            Exception("unexpected illuminate pattern")

        try:
            illu_na = get_led_na(led_pattern["idx"])
            imaging_na_capacity = 2*np.pi/self.d/self.k/2
            if imaging_na_capacity >= 1:
                print("The 'Image NA Capacity' is large then 1: {}".format(imaging_na_capacity))
            assert (illu_na + self.para["obj_na"]) <= imaging_na_capacity
        except AssertionError:
            raise Exception("\nThe 'Imaging NA Capacity' is lower than expect."
                            "\nIt could be handled with decrease 'pixel_size', increase 'magnification' or"
                            " decrease 'obj_na'.")

    def _spatial_coord(self, w, h):
        x = np.linspace(-w * self.d / 2, w * self.d / 2, w)
        y = -np.linspace(-h * self.d / 2, h * self.d / 2, h)
        return np.meshgrid(x, y)

    def _frequency_coord(self, w, h):
        # return delta_k, (kx_mat, ky_mat)
        kx = 2*np.pi*np.linspace(-1 / self.d / 2, 1 / self.d / 2, w)
        ky = 2*np.pi*np.linspace(-1 / self.d / 2, 1 / self.d / 2, h)
        return 2*np.pi/self.d/w, 2*np.pi/self.d/w, np.meshgrid(kx, ky)

    def _illu_cut_ctf(self, kx_mat, ky_mat, obj_na):
        return (kx_mat**2 + ky_mat**2) < (self.k * obj_na) ** 2

    def wf_illuminated(self, wf_init):
        h, w = wf_init.shape
        delta_kx, delta_ky, (kx_mat, ky_mat) = self._frequency_coord(w, h)
        _cut_ctf = self._illu_cut_ctf(kx_mat, ky_mat, obj_na=self.para["obj_na"])

        ft_init = ft2(wf_init)
        kx_offset = self.k * self.illu_nas[0] / delta_kx
        ky_offset = self.k * self.illu_nas[1] / delta_ky
        kxc = ft_init.shape[-1] // 2 - np.round(kx_offset)
        kyc = ft_init.shape[-2] // 2 + np.round(ky_offset)
        core_w = int(self.k * self.para["obj_na"] // delta_kx)
        core_h = int(self.k * self.para["obj_na"] // delta_ky)

        ft_shifted = np.zeros((h, w)).astype(np.complex)
        spectrum_core = ft_init[int(kyc-core_h):int(kyc+core_h), int(kxc-core_w):int(kxc+core_w)]
        ft_shifted[int(h//2-core_h):int(h//2+core_h), int(w//2-core_w):int(w//2+core_w)] = spectrum_core
        ft_shifted[~_cut_ctf] = 0+0j

        return ft_shifted


def main():
    # import image
    pixel_num = 512
    pixel_size = 5e-6
    mag = 10
    interval_d = 2e-6
    wavelength = 500e-9
    amp_img = resize(io.imread("../data/a.jpg"), (pixel_num, pixel_num))
    pha_img = resize(io.imread("../data/a.jpg"), (pixel_num, pixel_num))
    amp_img = (amp_img - amp_img.min()) / (amp_img - amp_img.min()).max()
    pha_img = np.pi * (pha_img - pha_img.min()) / (pha_img - pha_img.min()).max()

    # config illumination
    illu = Illuminate(magnification=mag, pixel_size=pixel_size, obj_na=0.5)

    # wavefront propagate
    wf_init = amp_img * np.exp(1j * pha_img)
    ft_shifted = illu.wf_illuminated(wf_init)

    fig, ax = plt.subplots(figsize=[8, 8])
    for i in range(-30, 30):
        ax.cla()
        wf_propagated = forward_propagate(ift2(ft_shifted), i*interval_d,
                                          wavelength, pixel_num, pixel_size/mag)
        ax.imshow(abs(wf_propagated), cmap="gray")
        ax.set_title("distance: {} um".format(i*interval_d//1e-6))
        plt.pause(0.5)


if __name__ == '__main__':
    main()
