import numpy as np
import os

'''To avoid the problem of label mismatch due to the missed detection of new organs'''

def sort_by_number(file_name):
    Initial = int(os.path.splitext(file_name)[0].split("_")[1])
    return Initial

def find_start(files):
    start_index = [0]
    for start_id in range(1, len(files)):
        if files[start_id].split('_')[3] != files[start_id - 1].split('_')[3] or files[start_id].split('_')[4] != files[start_id - 1].split('_')[4]:
            start_index.append(start_id)
    return start_index


standard_mode = False        # set to True if ther pretrained model is 3D-NOD; otherwise, set to False
if standard_mode:
    home_path = os.path.dirname(os.getcwd())
    gt_folder = os.path.join(home_path, "norm_GT_fps")
    res_folder = os.path.join(home_path, "output(3D-NOD)", "predict_result")
    save_folder = os.path.join(home_path, "output(3D-NOD)", "final_result")
    old_folder = os.path.join(home_path, "instance_old_organ(3D-NOD)")
    new_folder = os.path.join(home_path, "instance_new_organ(3D-NOD)")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    gt_ini_files = os.listdir(gt_folder)
    gt_files = []
    for file_name in gt_ini_files:
        if file_name.endswith(".txt"):
            gt_files.append(file_name)

    gt_files = sorted(gt_files, key=sort_by_number)
    file_numbers = len(gt_files)
    start = find_start(gt_files)
    dict = {}
    multi_dict = {}

    for i in range(file_numbers):
        if i in start:
            dict = {}
            multi_dict = {}
        ini_data = np.loadtxt(os.path.join(old_folder, gt_files[i]))
        print(gt_files[i])

        if len(ini_data) == 2048:
            res_data = np.loadtxt(os.path.join(res_folder, gt_files[i]))
            gt_data = np.loadtxt(os.path.join(gt_folder, gt_files[i]))
            unique_gt = np.unique(gt_data[:, 3])
            unique_res = np.unique(res_data[:, 3])
            if max(unique_res) in unique_gt:
                for key in multi_dict:
                    if key in dict:
                        del dict[key]
        else:
            gt_data = np.loadtxt(os.path.join(gt_folder, gt_files[i]))
            res_data = np.loadtxt(os.path.join(res_folder, gt_files[i]))

            new_points_data = np.loadtxt(os.path.join(new_folder, gt_files[i]))
            points_num = len(new_points_data)
            res_array = res_data[:points_num, :]
            res_unique_lab = np.unique(res_array[:, 3])

            gt_array = np.zeros((points_num, 4))
            for j in range(res_array.shape[0]):
                coord = res_array[:, :3][j]
                label = res_array[:, 3][j]
                match_indices = np.all(np.isclose(gt_data[:, :3], coord, atol=1e-6), axis=1)
                gt_new_data = gt_data[match_indices, :]
                gt_array[j, :] = gt_new_data

            for res_lab in res_unique_lab:
                res_lab_indice = np.where(res_array[:, 3]==res_lab)[0]
                res_lab_array = res_array[res_lab_indice, :]

                gt_lab_array = gt_array[res_lab_indice, :]
                gt_unique_lab, counts = np.unique(gt_lab_array[:, 3], return_counts=True)
                most_common_label = gt_unique_lab[np.argmax(counts)]

                unique_gt = np.unique(gt_data[:, 3])
                unique_res = np.unique(res_data[:, 3])
                if len(multi_dict) != 0:
                    unique_res = np.delete(unique_res, np.where(unique_res == list(multi_dict.keys())[0]))
                if res_lab == max(unique_gt):
                    replacement_dict = {res_lab: most_common_label}
                    replacement_dict = {k: v for k, v in replacement_dict.items() if k != v}
                    dict.update(replacement_dict)
                    for key in multi_dict:
                        if key in dict:
                            del dict[key]
                elif res_lab == max(unique_gt) + len(multi_dict):
                    replacement_dict = {res_lab: most_common_label}
                    replacement_dict = {k: v for k, v in replacement_dict.items() if k != v}
                    dict.update(replacement_dict)
                else:
                    if most_common_label in np.setdiff1d(unique_res, res_unique_lab):
                        replacement_dict = {res_lab: most_common_label - 15.0}
                        replacement_dict = {k: v for k, v in replacement_dict.items() if k != v}
                        multi_dict.update(replacement_dict)
                        dict.update(replacement_dict)
                    else:
                        replacement_dict = {res_lab: most_common_label}
                        replacement_dict = {k: v for k, v in replacement_dict.items() if k != v}
                        dict.update(replacement_dict)


        if len(dict) != 0:
            ini_labels = res_data[:, 3].copy()
            for key, value in dict.items():
                res_data[ini_labels == key, 3] = value

        save_path = os.path.join(save_folder, gt_files[i])
        np.savetxt(save_path, res_data, delimiter=" ", fmt="%f %f %f %d")