import scipy.io
from CCCode.imaging_process import Check
import numpy as np
import matplotlib.pyplot as plt


# test_arr1 = np.random.normal(size=(5, 5))
# test_arr2 = np.random.normal(size=(6, 6))
#
# scipy.io.savemat("data\\data_exchange\\test_mat",
#                  mdict={"test_arr1": test_arr1,
#                         "test_arr2": test_arr2})
# fname = "data\\data_exchange\\test_mat.mat"
# load_data = scipy.io.loadmat(fname)
# test_arr = load_data["test_arr"]
# print(type(test_arr))
# print(test_arr.shape)


def code():
    amp_influence_dir = "TIE Simulations/amp_value_influence.mat"
    amp_influence = scipy.io.loadmat(amp_influence_dir)
    coef = amp_influence["coef"]
    coef = np.reshape(coef, (9,))
    phase_imgs = amp_influence["Phase_imgs"]
    ck = Check()
    imgs = [phase_imgs[i] for i in range(1, 9)]
    ck.multi_img(img_list=imgs,
                 name_list=["amplitude: ({:.2f}, 1)".format(0.1*i)
                            for i in range(2, 10)],
                 colorbar_ticks_num=3,
                 interval_list=[[256+i//10 for i in range(8)],
                                [np.max(imgs[i])-np.min(imgs[i]) for i in range(8)]])
    x = np.linspace(0.2, 0.9, 8)
    coef_short = [0.625, 0.786, 0.884, 0.939, 0.969, 0.983, 0.989, 0.990]
    plt.plot(x, coef[1:])
    plt.ylabel("correlation coefficient")
    plt.xlabel("minimum value of amplitude")
    for a, b, n in zip(x, coef_short, range(8)):
        plt.text(a, b, "{:.3f}".format(coef[n+1]))
    plt.show()


if __name__ == '__main__':
    code()
