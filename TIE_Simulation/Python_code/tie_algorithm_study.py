import numpy as np
from PIL import Image as Im
from CCCode.imaging_process import Check, WavefrontPropagate, tie_algorithm
from CCCode.cc_math import img_scale, pearson_correlation_coefficient


def import_study_images(init_size):
    images_dir = "data/"
    amp_img = Im.open(images_dir+"pic_amp.jpg")
    pha_img = Im.open(images_dir+"pic_pha.jpg")
    amp_img.thumbnail((init_size, init_size))
    pha_img.thumbnail((init_size, init_size))
    amp_img = np.asarray(amp_img)
    pha_img = np.asarray(pha_img)
    return amp_img, pha_img


def optimize_tie(init_size, a_min, a_max, p_min, p_max,
                 target_size, pixel_size, delta):
    amp_img, pha_img = import_study_images(init_size)
    pic_size = amp_img.shape[0]

    # images scaling
    amp_img = img_scale(amp_img, a_min, a_max)
    pha_img = img_scale(pha_img, p_min, p_max)

    # wavefront generating
    wf_obj = WavefrontPropagate(target_size, pixel_size, 632.8e-9)
    wf_obj.get_obj_wavefront(amp_img, pha_img, 0, 0)

    # directly propagating
    multi_focus_imgs = wf_obj.directly_propagate(delta, pic_size)

    # cropping
    # idx1 = int((target_size-pic_size)/2)
    # idx2 = int((target_size+pic_size)/2)
    in_focus = multi_focus_imgs[0]  # [idx1:idx2, idx1:idx2]
    under_focus = multi_focus_imgs[1]  # [idx1:idx2, idx1:idx2]
    over_focus = multi_focus_imgs[2]  # [idx1:idx2, idx1:idx2]
    input_size = in_focus.shape[0]

    # tie solve
    p_xy = tie_algorithm(in_focus, under_focus, over_focus, input_size,
                         delta, pixel_size, 632.8e-9, 0)
    return pha_img, [in_focus, under_focus, over_focus], p_xy


def evaluating_process():
    pha, multi_focus_imgs, pxy = optimize_tie(256, 1.1, 1.18, 0.0, 1.5,
                                              256, 13e-6, 0.0005)
    diff = pha-np.min(pha)-(pxy-np.min(pxy))
    ck = Check()
    ck.directly_propagate(multi_focus_imgs, name="py_")
    ck.multi_img(ground_truth=pha, recover=pxy, diff=diff)
    coef = pearson_correlation_coefficient(pha, pxy)
    print(coef)


if __name__ == '__main__':
    evaluating_process()
