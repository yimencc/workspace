from CCCode.imaging_process import Check, WavefrontPropagate, tie_algorithm
from scipy import io
import numpy as np
from CCCode.cc_math import pearson_correlation_coefficient
from PIL import Image as Im


def import_multi_focus_data():
    fpath = "TIE Simulations/multi_focus_img.mat"
    load_data = io.loadmat(fpath)
    in_focus = load_data["I_focus"]
    under_focus = load_data["I_minus"]
    over_focus = load_data["I_plus"]
    phase_mr = load_data["Phase1"]
    return (in_focus, under_focus, over_focus), phase_mr


def import_amp_pha_data():
    fpath = "TIE Simulations/wavefront_img.mat"
    load_data = io.loadmat(fpath)
    amp_arr = load_data["Amplitude"]
    pha_arr = load_data["Phase"]
    return amp_arr, pha_arr


def py_predict():
    ck = Check()
    amp, pha = import_amp_pha_data()
    wf_obj = WavefrontPropagate(256, 6.0e-6, 632.8e-9)
    wf_obj.get_obj_wavefront(amp, pha)
    py_output = wf_obj.directly_propagate(0.0005, 256)
    print(py_output[0].shape)
    ck.directly_propagate(py_output)
    py_pha_xy = tie_algorithm(py_output[0], py_output[1], py_output[2],
                              0.0005, 6.0e-6, 632.8e-9, 0, 256)
    ck.multi_img(phase_recover=py_pha_xy)
    return py_output, py_pha_xy


def process_a_b_pic():
    amp_img = Im.open("data/pic_amp.jpg")
    pha_img = Im.open("data/pic_pha.jpg")
    amp_img.convert("L")
    pha_img.convert("L")
    amp_img.thumbnail((512, 512))
    pha_img.thumbnail((512, 512))
    amp_img.save("data/aaa.jpg")
    pha_img.save("data/bbb.jpg")


def evaluate_processing():
    _, pha = import_amp_pha_data()
    ml_multi_focus_img, ml_recover = import_multi_focus_data()
    py_multi_focus_img, py_recover = py_predict()
    diff_multi_focus_img = [a - b for a, b in zip(list(ml_multi_focus_img),
                                                  list(py_multi_focus_img))]
    ck = Check()
    ck.directly_propagate(ml_multi_focus_img, name="ml_")
    ck.directly_propagate(py_multi_focus_img, name="py_")
    ck.directly_propagate(diff_multi_focus_img)
    diff_pxy = ml_recover - np.min(ml_recover) - (py_recover - np.min(py_recover))
    ck.multi_img(ml_recover=ml_recover,
                 py_recover=py_recover,
                 phase_diff=diff_pxy)
    np_coef_ml = pearson_correlation_coefficient(pha, ml_recover)
    np_coef_py = pearson_correlation_coefficient(pha, py_recover)
    print(np_coef_ml, np_coef_py)


if __name__ == '__main__':
    evaluate_processing()
