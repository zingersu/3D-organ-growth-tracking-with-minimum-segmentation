import os
import shutil

import numpy as np
from pycpd import DeformableRegistration

CPD_TOLERANCE = 0.001
CPD_MAX_ITERATION = 100
MIN_ANGLE = np.pi / 35
MAX_DISTANCE_DIFF = 0.15
MIN_ORGAN_POINTS = 25

def sort_by_number(file_name):
    return int(os.path.splitext(file_name)[0].split("_")[1])

def save_points(path, data):
    np.savetxt(path, data, fmt="%f %f %f %d", delimiter=" ")

def calculate_angle(vector1, vector2):
    cos_theta = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def same_sequence(source_file, target_file):
    """True if the two frames belong to the same animal / part sequence."""
    return all(
        source_file.split("_")[k] == target_file.split("_")[k]
        for k in (2, 3, 4)
    )


def make_cpd_callback(cpd_folder, all_label, source_file_norm):
    """
    Callback for pycpd: saves the registered source + target points once
    the tolerance / iteration limit is reached.  Rows are ordered as
    [source (Y); target (X)] so that the labels stay aligned with ``all_label``.
    """
    def visualize(iteration, error, diff, X, Y):
        if diff < CPD_TOLERANCE or iteration > CPD_MAX_ITERATION:
            new_coo = np.vstack((Y, X))
            new_data = np.hstack((new_coo, all_label))
            save_points(os.path.join(cpd_folder, source_file_norm), new_data)

    return visualize


def compress_contiguous_labels(label_col):
    """Re-index labels to 0, 1, 2, ... following their sorted unique values."""
    unique_labels = np.unique(label_col)
    label_mapping = {lab: idx for idx, lab in enumerate(unique_labels)}
    flat = np.squeeze(label_col)
    return np.array([label_mapping[lab] for lab in flat]).reshape(-1, 1)


def reset_organ_labels(label_data, organ_indices, check_offset):
    """
    Set ``organ_indices`` to label 0 (old organ) and recompress labels if necessary.
    """
    label_data[organ_indices, 3] = 0
    label_col = label_data[:, 3:]
    n_unique = len(np.unique(label_col))
    max_label = np.max(label_col)
    if n_unique - check_offset == max_label:
        return label_col
    return compress_contiguous_labels(label_col)


def compute_match_arrays(t_data, t1_data, new_organ_t, new_organ_t1, center_point):
    """
    Build the (target organ, source organ) matrices of direction-vector angle
    and of distance-to-center difference.
    """
    angle_array = np.full((len(new_organ_t1), len(new_organ_t)), np.inf)
    distance_diff_array = np.full((len(new_organ_t1), len(new_organ_t)), np.inf)

    for i, t1_lab in enumerate(new_organ_t1):
        t1_center = np.mean(t1_data[np.where(t1_data == t1_lab)[0], :3], axis=0)
        direction_t1 = t1_center - center_point

        for j, t_lab in enumerate(new_organ_t):
            t_center = np.mean(t_data[np.where(t_data == t_lab)[0], :3], axis=0)
            direction_t = t_center - center_point

            angle_array[i, j] = calculate_angle(direction_t, direction_t1)
            distance_diff_array[i, j] = abs(
                np.linalg.norm(t_center - center_point)
                - np.linalg.norm(t1_center - center_point)
            )

    return angle_array, distance_diff_array


def greedy_organ_matching(angle_array, distance_diff_array):
    """
    Greedily match target organs to source organs, one pair per row / column.
    Yields ``(target_row_index, source_col_index)`` pairs from the smallest
    angle upwards, stopping when the angle or distance-difference threshold is violated.
    """
    deleted_rows, deleted_cols = set(), set()
    n_rows, n_cols = angle_array.shape

    while len(deleted_rows) < n_rows and len(deleted_cols) < n_cols:
        rem_angle = np.delete(angle_array, list(deleted_rows), axis=0)
        rem_angle = np.delete(rem_angle, list(deleted_cols), axis=1)
        rem_diff = np.delete(distance_diff_array, list(deleted_rows), axis=0)
        rem_diff = np.delete(rem_diff, list(deleted_cols), axis=1)

        if np.min(rem_angle) >= MIN_ANGLE:
            break

        min_idx = np.unravel_index(np.argmin(rem_angle), rem_angle.shape)
        min_angle = rem_angle[min_idx]
        min_diff = rem_diff[min_idx]

        if not (min_angle < MIN_ANGLE and min_diff < MAX_DISTANCE_DIFF):
            break

        orig_row, orig_col = np.where(angle_array == min_angle)
        yield orig_row[0], orig_col[0]
        deleted_rows.add(orig_row[0])
        deleted_cols.add(orig_col[0])


def reconcile_adjacent_frames(source_file_norm, target_file_norm, ini_folder, cpd_folder):
    """
    Run CPD between frame T (source) and T+1 (target), match duplicated
    organs, and keep only the larger copy.
    """
    source_points = np.loadtxt(os.path.join(ini_folder, source_file_norm))
    target_points = np.loadtxt(os.path.join(ini_folder, target_file_norm))

    organ_t = np.unique(source_points[:, 3])
    new_organ_t = organ_t[organ_t != 0]
    organ_t1 = np.unique(target_points[:, 3])
    new_organ_t1 = organ_t1[organ_t1 != 0]

    if len(new_organ_t) == 0 or len(new_organ_t1) == 0:
        return source_points, target_points, False, False

    Y = source_points[:, :3]
    source_label = source_points[:, -1:]
    X = target_points[:, :3]
    target_label = target_points[:, -1:]
    all_label = np.vstack((source_label, target_label))

    callback = make_cpd_callback(cpd_folder, all_label, source_file_norm)
    DeformableRegistration(**{"X": X, "Y": Y, "low_rank": False}).register(callback)

    new_data = np.loadtxt(os.path.join(cpd_folder, source_file_norm))
    center_point = new_data[np.argmin(new_data[:, 1]), :3]

    n_source = len(source_label)
    t_data = new_data[:n_source, :]
    t1_data = new_data[n_source:, :]

    angle_array, distance_diff_array = compute_match_arrays(
        t_data, t1_data, new_organ_t, new_organ_t1, center_point
    )

    source_modified = False
    target_modified = False

    for orig_row, orig_col in greedy_organ_matching(angle_array, distance_diff_array):
        t1_lab = new_organ_t1[orig_row]
        t_lab = new_organ_t[orig_col]
        t1_new_indice = np.where(t1_data == t1_lab)[0]
        t_new_indice = np.where(t_data == t_lab)[0]

        if len(t_data[t_new_indice]) > MIN_ORGAN_POINTS:
            # Keep the source organ; reset the target copy to old-organ label.
            t1_label = reset_organ_labels(t1_data, t1_new_indice, check_offset=1)
            target_points = np.hstack((target_points[:, :3], t1_label))
            target_modified = True
        else:
            # Keep the target organ; reset the source copy to old-organ label.
            t_label = reset_organ_labels(t_data, t_new_indice, check_offset=0)
            source_points = np.hstack((source_points[:, :3], t_label))
            source_modified = True

    return source_points, target_points, source_modified, target_modified


def unify_label_pairs(labels_first, labels_second):
    """
    Map each (source_label, target_label) pair to a shared label id.
    """
    label_pairs = np.column_stack((labels_first, labels_second))
    pair_to_new = {}
    new_labels = np.zeros(len(label_pairs), dtype=int)
    next_id = 1

    for i, pair in enumerate(label_pairs):
        if 0 in pair:
            new_labels[i] = 0
            continue
        key = tuple(pair)
        if key not in pair_to_new:
            pair_to_new[key] = next_id
            next_id += 1
        new_labels[i] = pair_to_new[key]

    return new_labels


def merge_frame_labels(source_folder, target_folder, save_folder):
    """
    Reconcile the per-frame edits written to the ``Merge_results_cpd_as_source`` and ``Merge_results_cpd_as_target`` folders:
    a frame modified in both passes is written with unified labels;
    files touched by only one pass are copied verbatim.
    """
    source_files = set(os.listdir(source_folder))
    target_files = set(os.listdir(target_folder))

    for file in source_files & target_files:
        first_data = np.loadtxt(os.path.join(source_folder, file))
        second_data = np.loadtxt(os.path.join(target_folder, file))
        first_data[:, 3] = unify_label_pairs(first_data[:, 3], second_data[:, 3])
        save_points(os.path.join(save_folder, file), first_data)

    for file in source_files - target_files:
        shutil.copy(os.path.join(source_folder, file), save_folder)

    for file in target_files - source_files:
        shutil.copy(os.path.join(target_folder, file), save_folder)


def main():
    home_path = os.path.dirname(os.getcwd())
    middle_result = os.path.join(home_path, "Middle_results")
    ini_folder = os.path.join(middle_result, "Results_cluster_labelling")
    save_folder = os.path.join(middle_result, "Predict_results")
    source_save_folder = os.path.join(middle_result, "Results_cpd_as_source")
    target_save_folder = os.path.join(middle_result, "Results_cpd_as_target")
    cpd_folder = os.path.join(middle_result, "Results_cpd_folder")

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(source_save_folder, exist_ok= True)
    os.makedirs(target_save_folder, exist_ok=True)
    os.makedirs(cpd_folder, exist_ok=True)

    ini_files = sorted(os.listdir(ini_folder), key=sort_by_number)

    # Copy every input as the default output and collect the files to be CPD-matched (maize / sorghum are excluded).
    remain_files = []
    for file in ini_files:
        shutil.copy(os.path.join(ini_folder, file), save_folder)
        if file.split("_")[2] not in ("maize", "sorghum"):
            remain_files.append(file)

    # Adjacent-frame reconciliation.
    for source_file_norm, target_file_norm in zip(remain_files[:-1], remain_files[1:]):
        if not same_sequence(source_file_norm, target_file_norm):
            continue

        source_points, target_points, source_modified, target_modified = (
            reconcile_adjacent_frames(
                source_file_norm, target_file_norm, ini_folder, cpd_folder
            )
        )
        if target_modified:
            save_points(os.path.join(target_save_folder, target_file_norm), target_points)
        if source_modified:
            save_points(os.path.join(source_save_folder, source_file_norm), source_points)

    merge_frame_labels(source_save_folder, target_save_folder, save_folder)


if __name__ == "__main__":
    main()