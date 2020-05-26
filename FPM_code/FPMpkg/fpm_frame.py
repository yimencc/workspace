import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from FPMpkg.fpm_config import IndexSequence, IlluminatePlane, IterationCore
# import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io
# import os


class FPMRecover:

    def __init__(self, amplitude, phase, wavelength=632.8e-9, sampling_size=2.75e-6,
                 sampling_image_size=64, na=0.08, z_aber=0.0):
        self.wavelength = wavelength
        self.sampling_size = sampling_size
        self.NA = na
        self.m_s, self.n_s = sampling_image_size, sampling_image_size
        self.k0 = 2 * np.pi / wavelength
        self.cutoff_frequency = self.NA * self.k0
        self.illuminate_plane()
        self._object_plane(amplitude, phase)
        self.pupil_plane(z_aber)
        self.image_sequences()

    def illuminate_plane(self):
        illu_obj = IlluminatePlane(self.k0)
        self.kx_illu, self.ky_illu = illu_obj.kx_illu, illu_obj.ky_illu

    def _object_plane(self, amplitude, phase):
        # object size == image size: m * pixel_size == m_s * sampling_size
        self.object = amplitude * np.exp(1j * phase)
        [self.m, self.n] = self.object.shape  # object pixel number
        self.pixel_size = self.m_s * self.sampling_size / self.m  # object plane pixel size
        self.dkx = 2 * np.pi / (self.pixel_size * self.n)
        self.dky = 2 * np.pi / (self.pixel_size * self.m)
        # notice: maximum resolvabel wavelength: pixel_size * pixel_number
        #         minimum resolvabel frequence: 2 * pi / pixel_size * pixel_number
        kmax_s = np.pi / self.sampling_size
        [self.kx_s, self.ky_s] = np.meshgrid(np.linspace(-kmax_s, kmax_s, self.n_s),
                                             -np.linspace(-kmax_s, kmax_s, self.m_s))
        self.object_recover = np.ones((self.m, self.n))
        self.object_recover_FT = fftshift(fft2(self.object_recover))

    def pupil_plane(self, z):
        self.CTF = self.kx_s ** 2 + self.ky_s ** 2 < self.cutoff_frequency ** 2
        kzm = np.sqrt(self.k0**2-self.kx_s**2-self.ky_s**2)
        # phase delay for first item and amplitude attenuation for second item
        # in this instance the amplitude attenuation is zero in all position
        self.pupil = np.exp(1j*z*np.real(kzm))*np.exp(-abs(z)*abs(np.imag(kzm)))

    def image_sequences(self):
        object_ft = fftshift(fft2(self.object))
        width_x, width_y = round(self.n_s / 2), round(self.m_s / 2)
        self.image_seqs = np.zeros((len(self.kx_illu), 64, 64))
        for i in range(len(self.kx_illu)):
            kxc = round(self.n / 2 + self.kx_illu[i] / self.dkx)
            kyc = round(self.m / 2 + self.ky_illu[i] / self.dky)
            kxl, kxh = int(kxc - width_x), int(kxc + width_x)
            kyl, kyh = int(kyc - width_y), int(kyc + width_y)
            image_seqs_ft = (self.m_s/self.m)**2*object_ft[kyl:kyh, kxl:kxh]*self.CTF*self.pupil
            self.image_seqs[i] = abs(ifft2(ifftshift(image_seqs_ft))) ** 2

    def recover_process(self, n_loop=5, index_mode="array",
                        iter_mode="initial", output_list=False):
        width_x, width_y = int(self.n_s/2), int(self.m_s/2)
        self.P_e = np.ones((self.m_s, self.n_s))
        object_list = []
        index_obj = IndexSequence(index_mode)
        iter_core = IterationCore(self.m, self.m_s, self.object_recover_FT,
                                  self.CTF, self.pupil, self.P_e, iter_mode)
        for loop in range(n_loop):
            for idx in index_obj.sequences:
                kxc = round(self.n/2 + self.kx_illu[idx]/self.dkx)
                kyc = round(self.m/2 + self.ky_illu[idx]/self.dky)
                kxl, kxh = int(kxc - width_x), int(kxc + width_x)
                kyl, kyh = int(kyc - width_y), int(kyc + width_y)
                iter_core(kxl, kxh, kyl, kyh, self.image_seqs[idx])
            self.object_recover, self.P_e = iter_core.object_output()
            if output_list:
                object_list.append((self.object_recover, self.P_e))
        return object_list if output_list else (self.object_recover, self.P_e)


if __name__ == '__main__':
    base_path = "D:\\Postgraduate\\Paper\\FPM\\code\\python_code"
    cameraman = resize(io.imread(base_path+"\\cameraman.tif", as_gray=True),
                       (256, 256))
    westdoor = resize(io.imread(base_path+"\\westdoor.png", as_gray=True),
                      (256, 256))
    # ck.multi_img(amp=cameraman, pha=westdoor) !! Unavailable
    fpm_init = FPMRecover(amplitude=cameraman, phase=np.pi*westdoor)
    _object_list = fpm_init.recover_process(50, output_list=True)
    object_recover = _object_list[49][0]
    # ck.multi_img(amp=abs(object_recover), pha=np.angle(object_recover))  !! Unavailable
