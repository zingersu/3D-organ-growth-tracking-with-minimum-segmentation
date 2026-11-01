import os
import numpy as np

def sort_by_number(file_name):
    return int(file_name.split('_')[1])

def save_points(path, points):
    np.savetxt(path, points, delimiter=" ", fmt="%f %f %f %d")


def remap_labels_to_contiguous(labels):
    """
    Map each unique label value to a consecutive integer starting at 0,
    following the sorted order of the original unique labels.
    """
    unique_label = np.unique(labels)
    label_mapping = {old: new for new, old in enumerate(unique_label)}
    return np.vectorize(label_mapping.get)(labels)


def split_old_and_new_organ(data, sem_label):
    """
    Split ``data`` into old-organ and new-organ point sets based on
    ``sem_label`` (0 = old organ, anything else = new organ).
    """
    old_organ_points = data[sem_label == 0, :]
    new_organ_points = data[sem_label != 0, :]
    return old_organ_points, new_organ_points


def main():
    home_path = os.path.dirname(os.getcwd())
    middle_result = os.path.join(home_path, "Middle_results")
    ini_file_folder = os.path.join(middle_result, "Predict_results")
    new_organ_folder = os.path.join(os.path.dirname(home_path), "instance_new_organ(3D-NOD)")
    old_organ_folder = os.path.join(os.path.dirname(home_path), "instance_old_organ(3D-NOD)")

    os.makedirs(new_organ_folder, exist_ok=True)
    os.makedirs(old_organ_folder, exist_ok=True)

    all_files = sorted(os.listdir(ini_file_folder), key=sort_by_number)

    for file_name in all_files:
        full_path = os.path.join(ini_file_folder, file_name)
        data = np.loadtxt(full_path)
        sem_label = data[:, -1].astype(int)

        # Remap label values to consecutive integers 0, 1, 2, ...
        sem_label = remap_labels_to_contiguous(sem_label)
        data[:, -1] = sem_label

        old_organ_points, new_organ_points = split_old_and_new_organ(data, sem_label)
        save_points(os.path.join(old_organ_folder, file_name), old_organ_points)

        if len(new_organ_points) == 0:
            continue
        save_points(os.path.join(new_organ_folder, file_name), new_organ_points)

if __name__ == "__main__":
    main()