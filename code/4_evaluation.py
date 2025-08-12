import os
import numpy as np
import time


def log_string(out_str):
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)


def sort_by_number(file_name):
    Initial = int(file_name.split('_')[1])
    return Initial


def find_start(files):
    start_index = [0]
    for start_id in range(1, len(files)):
        if files[start_id].split('_')[3] != files[start_id - 1].split('_')[3] or files[start_id].split('_')[4] != files[start_id - 1].split('_')[4]:
            start_index.append(start_id)
    return start_index


if __name__ == "__main__":
    t1 = time.time()
    plant_list = ['maize', 'sorghum', 'tobacco', 'tomato']
    standard_mode = False                                                  # set to True if ther pretrained model is 3D-NOD; otherwise, set to False
    base = os.path.dirname(os.getcwd())

    if standard_mode:
        output_path = os.path.join(base, "output(3D-NOD)")
        predict_result = os.path.join(output_path, "final_result")
    else:
        output_path = os.path.join(base, "output(by_human)")
        predict_result = os.path.join(output_path, "predict_result")

    for pid in range(len(plant_list)):
        col_path = os.path.join(output_path, "iou_col", plant_list[pid])
        res_path = os.path.join(predict_result, plant_list[pid])
        gt_path = os.path.join(base, "norm_GT_fps", plant_list[pid])

        total_instance = 0
        seq_total_instance = 0
        STTA = True
        true_matchII = 0
        seq_true_matchII = 0
        single_true_matchII = 0

        log_file = open(os.path.join(output_path, plant_list[pid] + "_evaluation.txt"), "w")

        files = sorted(os.listdir(col_path), key=sort_by_number)
        num_files = len(files)
        start = find_start(files)

        for i in range(num_files):
            if i not in start:
                cur_gt = np.loadtxt(os.path.join(gt_path, files[i]))
                gt_class = int(np.max(cur_gt[:, -1])) + 1
                total_instance = total_instance + len(np.unique(cur_gt[:, -1]))
                seq_total_instance = seq_total_instance + len(np.unique(cur_gt[:, -1]))
                if STTA:
                    cur_res = np.loadtxt(os.path.join(res_path, files[i]))
                    res_class =len(np.unique(cur_res[:, -1]))
                    cur_col = np.loadtxt(os.path.join(col_path, files[i])).reshape(-1, 1)
                    print(cur_col)
                    for ins in range(gt_class):
                        if ins in cur_res[:, -1] and cur_col[ins, 0] == ins:
                            single_true_matchII = single_true_matchII + 1
                            true_matchII = true_matchII + 1
                            seq_true_matchII = seq_true_matchII + 1
                    log_string(files[i] + " : %d / %d %d" % (single_true_matchII, gt_class, res_class))
                    single_true_matchII = 0

            if i < num_files - 1:
                if files[i].split("_")[4] != files[i + 1].split("_")[4] or files[i].split('_')[3] != files[i + 1].split('_')[3]:
                    log_string("Matching results for the entire sequence:")
                    log_string("total_instance=%d" % seq_total_instance)
                    log_string("true_matchII=%d" % seq_true_matchII)
                    seq_total_instance = 0
                    seq_true_matchII = 0
                    log_string("------------------------")
            elif i == num_files - 1:
                log_string("Matching results for the entire sequence:")
                log_string("total_instance=%d" % seq_total_instance)
                log_string("true_matchII=%d" % seq_true_matchII)
                seq_total_instance = 0
                seq_true_matchII = 0
                log_string("-------End-------")

        log_string("total_instance=%d" % total_instance)
        acc2 = float(true_matchII / total_instance)
        log_string("true_matchII=%d" % true_matchII)
        log_string("AccuracyII =%f" % acc2)
    t2 = time.time()
    print(t2 - t1)
