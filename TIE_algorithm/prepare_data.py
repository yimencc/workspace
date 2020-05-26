import os
import numpy as np
from CCCode.imaging_process import ImageProcess_v2, Check, WavefrontPropagate, tie_algorithm
from CCCode.cc_math import pearson_correlation_coefficient_comput
import pickle


def process_one_pic(amp_fp, pha_fp):
    amp_obj = ImageProcess_v2(amp_fp)
    pha_obj = ImageProcess_v2(pha_fp)
    amp = amp_obj.img_crop(crop_rate=1.0).img_resize((512, 512)).add_hole_mask().img_normalize(1.0, 1.08)
    pha = pha_obj.img_crop(crop_rate=1.0).img_resize((512, 512)).add_hole_mask().img_normalize(0.0, 1.5)
    wave_obj = WavefrontPropagate(512, 7.4e-6, 632.8e-9)
    multi_focus_img = wave_obj.get_obj_wavefront(amp, pha).directly_propagate(delta_=0.0015, crop_size=512)
    p_xy = tie_algorithm(multi_focus_img[0], multi_focus_img[1], multi_focus_img[2],
                         512, 0.0015, 7.4e-6, 632.8e-9, 0)
    return pha, multi_focus_img, p_xy


def one_pic_processing_check():
    start_idx = np.random.randint(0, 41000)
    file_list = os.listdir(file_path)[start_idx:start_idx+8]
    coef = []
    for i in range(4):
        amp_fp = os.path.join(file_path, file_list[2*i])
        pha_fp = os.path.join(file_path, file_list[2*i+1])
        pha, multi_focus_img, p_xy = process_one_pic(amp_fp, pha_fp)
        coef.append(pearson_correlation_coefficient_comput(pha, p_xy))
        ck.directly_propagate(multi_focus_img, extremun=False)
        ck.multi_img(ground_truth=pha, predict=p_xy)
    print(np.mean(coef))


def prepare_data(process_number=1000,
                 save_number=1000,
                 shuffle_list=True,
                 utilize_threshold=True,
                 threshold=0.97):
    file_list = os.listdir(file_path)
    if shuffle_list:
        np.random.shuffle(file_list)
    amp_file_list = file_list
    pha_file_list = file_list[len(file_list)//2:] + file_list[:len(file_list)//2]
    coef_list, current_cor_num, save_file_idx = [], 0, 0
    arr_size = process_number if process_number <= save_number else save_number
    data_arr = np.zeros((arr_size, 512, 512, 4))
    print("Processing:")
    for i in range(process_number):
        amp_fp = os.path.join(file_path, amp_file_list[i])
        pha_fp = os.path.join(file_path, pha_file_list[i])
        pha, multi_focus_img, p_xy = process_one_pic(amp_fp, pha_fp)
        coef = pearson_correlation_coefficient_comput(pha, p_xy)
        coef_list.append(coef)
        save_img = True if coef >= threshold else False if utilize_threshold else True
        if save_img:
            data_arr[current_cor_num, :, :, 0] = multi_focus_img[0]
            data_arr[current_cor_num, :, :, 1] = multi_focus_img[1]
            data_arr[current_cor_num, :, :, 2] = multi_focus_img[2]
            data_arr[current_cor_num, :, :, 3] = pha
            current_cor_num += 1
        if current_cor_num != 0 and current_cor_num % save_number == 0:
            current_cor_num = 0
            save_file_name = "multi_focus_img_" + str(save_file_idx)
            save_file_idx += 1
            print("\r", "saving file -- ", save_file_idx, end="", flush=True)
            with open(os.path.join(save_path, save_file_name), "wb") as wf:
                pickle.dump(data_arr, wf)
        if (i + 1) % 10 == 0:
            print("\r", "processed number:", i+1,
                  "   right number:", current_cor_num,
                  "   saved file number: ", save_file_idx,
                  end="", flush=True)
    print("\n\naverage number of coef:", np.mean(coef_list))
    print("total right number:", save_number*save_file_idx+current_cor_num,
          "  use rate: {}%".format(100*(save_number*save_file_idx+current_cor_num)/process_number))


def data_test_version2():
    f_n = save_path+"\\multi_focus_img_0"
    with open(f_n, "rb") as rf:
        multi_focus_imgs = pickle.load(rf)
    idx_list = [i for i in range(100)]
    np.random.shuffle(idx_list)
    coef_list = []
    for i in range(10):
        idx = idx_list[i]
        out0 = multi_focus_imgs[idx, :, :, 0]
        out1 = multi_focus_imgs[idx, :, :, 1]
        out2 = multi_focus_imgs[idx, :, :, 2]
        pha = multi_focus_imgs[idx, :, :, 3]
        p_xy = tie_algorithm(out0, out1, out2,
                             512, 0.0015, 7.4e-6, 632.8e-9, 0)
        ck.multi_img(ground_truth=pha, predict=p_xy)
        coef = pearson_correlation_coefficient_comput(pha, p_xy)
        coef_list.append(coef)
        if i % 10 == 0:
            print(i)
    print("average number of coef: ", np.mean(coef_list))


def main():
    # prepare_data(process_number=1000,
    #              save_number=100, threshold=0.97)
    # one_pic_processing_check()
    data_test_version2()


if __name__ == '__main__':
    ck = Check()
    file_path = "D:/workspace_D/datasets/open_image_val"
    save_path = "D:/workspace_D/datasets/open_image_val_sparse"
    main()
