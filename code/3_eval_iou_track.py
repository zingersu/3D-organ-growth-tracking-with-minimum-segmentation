import os
import numpy as np
import pandas as pd
import shutil

def sort_by_number(file_name):
    Initial = int(os.path.splitext(file_name)[0].split("_")[1])
    return Initial


standard_mode = False                      # set to True if ther pretrained model is 3D-NOD; otherwise, set to False
home_path = os.path.dirname(os.getcwd())
if standard_mode:
    output_path = os.path.join(home_path, "output(3D-NOD)")
    predict_result = os.path.join(output_path, "final_result")
else:
    output_path = os.path.join(home_path, "output(by_human)")
    predict_result = os.path.join(output_path, "predict_result")
files = os.listdir(predict_result)

plant_list = ['maize', 'sorghum', 'tobacco', 'tomato']
for pid in range(len(plant_list)):
    res_path = os.path.join(predict_result, plant_list[pid])
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    for file_name in files:
        if file_name.endswith(".txt") == True and plant_list[pid] in file_name:
            prediction = os.path.join(predict_result, file_name)
            shutil.copy(prediction, res_path)


for pid in range(len(plant_list)):
    res_path = os.path.join(predict_result, plant_list[pid])
    gt_path = os.path.join(home_path, "norm_GT_fps", plant_list[pid])
    iou = os.path.join(output_path, "iou")
    col = os.path.join(output_path, "iou_col")
    if not os.path.exists(iou):
        os.mkdir(iou)
    if not os.path.exists(col):
        os.mkdir(col)

    iou_path = os.path.join(iou, plant_list[pid])
    col_path = os.path.join(col, plant_list[pid])
    if not os.path.exists(iou_path):
        os.mkdir(iou_path)
    if not os.path.exists(col_path):
        os.mkdir(col_path)

    files = sorted(os.listdir(gt_path), key=sort_by_number)
    n = len(files)
    for i in range(n):
        gt = np.loadtxt(os.path.join(gt_path, files[i]))
        res = np.loadtxt(os.path.join(res_path, files[i]))
        gt_label = np.unique(gt[:, -1])
        res_label = np.unique(res[:, -1])
        gt_class = len(gt_label)
        res_class = len(res_label)

        matrix = np.zeros((max(gt_class, res_class), max(gt_class, res_class)))

        for class_gt in range(max(gt_class, res_class)):
            for class_res in range(max(gt_class, res_class)):
                gt_idx = np.where(gt[:, -1] == class_gt)[0]
                res_idx = np.where(res[:, -1] == class_res)[0]
                gt_curclass = gt[gt_idx]
                res_curclass = res[res_idx]

                if gt_curclass.shape[0] == 0 or res_curclass.shape[0] == 0:
                    matrix[class_gt, class_res] = 0
                else:
                    gt_curclass = gt_curclass[:, :3]
                    res_curclass = res_curclass[:, :3]
                    gt_df = pd.DataFrame(gt_curclass, columns=['x', 'y', 'z'])
                    res_df = pd.DataFrame(res_curclass, columns=['x', 'y', 'z'])

                    intersection_df = pd.merge(gt_df, res_df, on=['x', 'y', 'z'], how='inner')
                    tp = len(intersection_df)

                    union_df = pd.concat([gt_df, res_df]).drop_duplicates(subset=['x', 'y', 'z'])
                    matrix[class_gt, class_res] = tp / len(union_df)

        max_value_indice = np.zeros((matrix.shape[0],1))
        for j in range(matrix.shape[0]):
            max_value_indice[j] = np.argmax(matrix[j])
        np.savetxt(os.path.join(col_path, files[i]), max_value_indice, fmt="%d")
        np.savetxt(os.path.join(iou_path, files[i]), matrix, fmt="%.3f", delimiter=" ")