import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift


class IlluminatePlane:

    def __init__(self, k0, mode="array", **kwargs):
        self.k0 = k0
        if mode in ["array"]:
            arg_dict = {"led_array_size": 15, "led_gap": 4e-3, "led_height": 90e-3}
            for key in arg_dict.keys():
                arg_dict.update(kwargs.fromkeys(key)) if key in kwargs else None
            self.array_illuminate(**arg_dict)

    def array_illuminate(self, **args_dict):
        arg_list = ["led_array_size", "led_gap", "led_height"]
        led_array_size, led_gap, led_height = [args_dict.get(arg) for arg in arg_list]
        x_location, y_location = np.zeros((2, led_array_size ** 2))
        for i in range(led_array_size):
            x_location[i*led_array_size:(i+1)*led_array_size] = (np.arange(led_array_size)
                                                                 - (led_array_size-1)/2)*led_gap
            y_location[i*led_array_size:(i+1)*led_array_size] = ((led_array_size-1)/2-i)*led_gap
        kx_relative = -np.sin(np.arctan(x_location/led_height))
        ky_relative = -np.sin(np.arctan(y_location/led_height))
        self.kx_illu = kx_relative * self.k0
        self.ky_illu = ky_relative * self.k0


class IterationCore:

    def __init__(self, m, m_s, object_recover_ft,
                 ctf, pupil, pupil_e, mode="initial"):
        if mode in ["initial", "known_abr", "unknown_abr"]:
            self.mode = mode
        else:
            raise Exception("bad mode")
        self.m = m
        self.m_s = m_s
        self.object_recover_FT = object_recover_ft
        self.object_recover = ifft2(ifftshift(object_recover_ft))
        self.CTF = ctf
        self.pupil = pupil
        self.P_e = pupil_e

    def initial_algorithm(self, kxl, kxh, kyl, kyh, low_image):
        # generate estimate of low-pass filtered image in Fourier domain and spatial domain
        estimate_ft_m = (self.m_s/self.m)**2*self.object_recover_FT[kyl:kyh, kxl:kxh]*self.CTF
        estimate_m = ifft2(ifftshift(estimate_ft_m))

        # update the estimate of low-pass filtered image using acquired intensity image
        estimate_m = (self.m/self.m_s)**2*np.sqrt(low_image)*np.exp(1j*np.angle(estimate_m))

        # update the high resolution estimate
        estimate_ft_m = fftshift(fft2(estimate_m)) * self.CTF
        self.object_recover_FT[kyl:kyh, kxl:kxh] = (1-self.CTF)*self.object_recover_FT[
            kyl:kyh, kxl:kxh] + estimate_ft_m

    def known_aberration_algorithm(self, kxl, kxh, kyl, kyh, low_image):
        # generate estimate of low-pass filtered image in Fourier domain and spatial domain
        estimate_ft_m = (self.m_s/self.m)**2*self.object_recover_FT[kyl:kyh,
                                                                    kxl:kxh]*self.CTF*self.pupil
        estimate_m = ifft2(ifftshift(estimate_ft_m))

        # update the estimate of low-pass filtered image using acquired intensity image
        estimate_m = (self.m/self.m_s)**2*np.sqrt(low_image)*np.exp(1j*np.angle(estimate_m))

        # update the high resolution estimate
        estimate_ft_m = fftshift(fft2(estimate_m)) * self.CTF / self.pupil
        self.object_recover_FT[kyl:kyh, kxl:kxh] = (1-self.CTF)*self.object_recover_FT[
            kyl:kyh, kxl:kxh] + estimate_ft_m

    def unknown_aberration_algorithm(self, kxl, kxh, kyl, kyh, low_image):
        # generate estimate of low-pass filtered image in Fourier domain and spatial domain
        estimate_ft_m_1 = self.object_recover_FT[kyl:kyh, kxl:kxh] * self.CTF * self.P_e
        estimate_m = ifft2(ifftshift(estimate_ft_m_1))

        # update the estimate of low-pass filtered image using acquired intensity image
        estimate_m = (self.m/self.m_s)**2 * np.sqrt(low_image) * np.exp(1j*np.angle(estimate_m))

        # update the high resolution estimate
        estimate_ft_m_2 = fftshift(fft2(estimate_m))

        self.object_recover_FT[kyl:kyh, kxl:kxh] = self.object_recover_FT[
            kyl:kyh, kxl:kxh] + np.conj(self.CTF*self.P_e)*(
            estimate_ft_m_2-estimate_ft_m_1)/np.max(abs(self.CTF*self.P_e)**2)

        self.P_e = self.P_e + np.conj(self.object_recover_FT[kyl:kyh, kxl:kxh])*(
            estimate_ft_m_2-estimate_ft_m_1)/np.max(abs(self.object_recover_FT[kyl:kyh, kxl:kxh])**2)

    def __call__(self, kxl, kxh, kyl, kyh, low_image, *args, **kwargs):
        if self.mode is "initial":
            self.initial_algorithm(kxl, kxh, kyl, kyh, low_image)
        elif self.mode is "known_abr":
            self.known_aberration_algorithm(kxl, kxh, kyl, kyh, low_image)
        elif self.mode is "unknown_abr":
            self.unknown_aberration_algorithm(kxl, kxh, kyl, kyh, low_image)

    def object_output(self):
        self.object_recover = ifft2(ifftshift(self.object_recover_FT))
        return self.object_recover, self.P_e


class IndexSequence:

    def __init__(self, mode="array"):
        self.array_index_sequences(15)

    def array_index_sequences(self, led_array_size):
        # generate the recover sequence of acquired images
        led_array_size = int(led_array_size) if not isinstance(led_array_size, int) else led_array_size
        n = int(led_array_size - 1) / 2
        self.sequences = np.zeros((led_array_size * led_array_size), dtype=np.int64)
        self.sequences[0] = n * led_array_size + n
        temp = self.sequences[0]
        dx, dy, step_x, step_y, direction, counter = 1, 15, 1, 1, 1, 0
        for i in range(1, led_array_size * led_array_size):
            counter += 1
            if direction == 1:
                self.sequences[i] = temp + dx
                temp = self.sequences[i]
                if counter == abs(step_x):
                    counter, direction, dx, step_x = 0, -direction, -dx, -step_x
                    step_x += 1 if step_x > 0 else -1
            else:
                self.sequences[i] = temp + dy
                temp = self.sequences[i]
                if counter == abs(step_y):
                    counter, direction, dy, step_y = 0, -direction, -dy, -step_y
                    step_y += 1 if step_y > 0 else -1


if __name__ == '__main__':
    pass
