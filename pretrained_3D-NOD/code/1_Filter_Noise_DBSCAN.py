import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

def sort_by_number(file_name):
    return int(os.path.splitext(file_name)[0].split("_")[1])

def save_points(path, data):
    np.savetxt(path, data, delimiter=" ", fmt="%f %f %f %d")

def load_with_zero_label(src_path):
    data = np.loadtxt(src_path)
    sem_label = np.zeros((len(data), 1))
    return np.hstack((data[:, :3], sem_label))


def dbscan_knn_relabel(points, labels, epsilon, filter_small_clusters):
    """
    For each semantic label, run DBSCAN and collect indices to be "reset":
    all DBSCAN noise points, plus — when ``filter_small_clusters`` is True —
    points belonging to small clusters (threshold 100 for label 0,
    15 for any other label).  The reset points are then relabeled by
    iterative KNN majority voting from the remaining ("main") points.

    ``labels`` is modified in place.
    """
    reset_indices = np.array([], dtype=int)
    for lab in np.unique(labels):
        instance_indices = np.where(labels == lab)[0]
        instance_points = points[instance_indices]
        cluster_labels = DBSCAN(eps=epsilon, min_samples=8).fit(instance_points).labels_

        if filter_small_clusters:
            unique_clusters, count_point = np.unique(
                cluster_labels[cluster_labels >= 0], return_counts=True
            )
            threshold = 100 if lab == 0 else 15
            small_clusters = unique_clusters[count_point <= threshold]
            to_reset = np.append(small_clusters, -1)
            reset_indices = np.append(
                reset_indices,
                instance_indices[np.isin(cluster_labels, to_reset)],
            )
        else:
            reset_indices = np.append(
                reset_indices,
                instance_indices[cluster_labels == -1],
            )

    if len(reset_indices) == 0:
        return

    stay_indices = np.setdiff1d(np.arange(len(points)), reset_indices)
    main_points = points[stay_indices]
    main_label = labels[stay_indices]
    knn = NearestNeighbors(radius=1.5 * epsilon).fit(main_points)

    pending = reset_indices.astype(int)
    while len(pending) > 0:
        newly_labeled = []
        unlabeled = []
        for idx in pending:
            neighbors_idx = knn.radius_neighbors(
                [points[idx]], return_distance=False
            )[0]
            if len(neighbors_idx) > 0:
                neighbor_labels = main_label[neighbors_idx].astype(int)
                labels[idx] = np.bincount(neighbor_labels).argmax()
                newly_labeled.append(idx)
            else:
                unlabeled.append(idx)

        pending = unlabeled
        if len(newly_labeled) == 0:
            labels[pending] = 0
            break

        main_points = np.vstack([main_points, points[newly_labeled]])
        main_label = np.concatenate([main_label, labels[newly_labeled]])
        knn.fit(main_points)


def remove_embedded_new_organ(points, labels, epsilon):
    """
    Step 2: run DBSCAN on the "new organ" points and remove the clusters
    that are embedded inside the old organ (fixes new-organ misidentification
    in 3D_NOD).  ``labels`` is modified in place.

    Returns False if there are no "new organ" points to check, True otherwise.
    """
    new_organ_points = points[labels == 1]
    if len(new_organ_points) == 0:
        return False

    clusters = DBSCAN(eps=epsilon, min_samples=8).fit(new_organ_points).labels_
    tree_old = cKDTree(points[labels == 0])

    cluster_connections = []
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            continue
        cluster_points = new_organ_points[clusters == cluster_id]
        if cluster_points.shape[0] < 2:
            continue
        dist_matrix = squareform(pdist(cluster_points))
        max_distance = np.max(dist_matrix)
        i, j = np.where(dist_matrix == max_distance)[0]
        point1, point2 = cluster_points[i], cluster_points[j]
        distances1, _ = tree_old.query(point1.reshape(1, -1), k=1)
        distances2, _ = tree_old.query(point2.reshape(1, -1), k=1)
        connected = np.sum(distances1[0] < epsilon) + np.sum(distances2[0] < epsilon)
        cluster_connections.append((cluster_id, connected))

    embedded_ids = [cid for cid, conn in cluster_connections if conn >= 2]
    for cluster_id in embedded_ids:
        mask = clusters == cluster_id
        labels[np.where(labels == 1)[0][mask]] = 0

    return True


def main():
    home_path = os.path.dirname(os.getcwd())
    ini_folder = os.path.join(home_path, "3D-NOD_Predict_results")
    middle_result = os.path.join(home_path, "Middle_results")
    os.makedirs(middle_result, exist_ok=True)
    
    first_save_folder = os.path.join(middle_result, "Filter_Predict_results")
    save_folder = os.path.join(middle_result, "Final_Filter_Predict_results")

    os.makedirs(first_save_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)

    ini_files = sorted(os.listdir(ini_folder), key=sort_by_number)
    file_numbers = len(ini_files)
    remain_files = ini_files.copy()

    # The first file, and any file whose anatomical header differs from the
    # previous one, is written with an all-zero semantic label.
    for i in range(file_numbers - 1):
        source_file_norm = ini_files[i]
        target_file_norm = ini_files[i + 1]
        parts_to_check = [
            source_file_norm.split("_")[2] == target_file_norm.split("_")[2],
            source_file_norm.split("_")[3] == target_file_norm.split("_")[3],
            source_file_norm.split("_")[4] == target_file_norm.split("_")[4],
        ]

        if i == 0:
            save_points(
                os.path.join(save_folder, source_file_norm),
                load_with_zero_label(os.path.join(ini_folder, source_file_norm)),
            )
            remain_files.remove(source_file_norm)
        elif False in parts_to_check:
            save_points(
                os.path.join(save_folder, target_file_norm),
                load_with_zero_label(os.path.join(ini_folder, target_file_norm)),
            )
            remain_files.remove(target_file_norm)

    for file in remain_files:
        data = np.loadtxt(os.path.join(ini_folder, file))
        points = data[:, :3]
        labels = data[:, 3].copy()

        neigh = NearestNeighbors(n_neighbors=2).fit(points)
        distances, _ = neigh.kneighbors(points)
        epsilon = 2 * np.mean(distances[:, 1])

        if len(np.unique(labels)) == 1:
            save_points(os.path.join(save_folder, file), data)
            save_points(os.path.join(first_save_folder, file), data)
            continue

        # Step 1: preliminary filtering with DBSCAN and iterative KNN;
        # remove noise points and small clusters.
        dbscan_knn_relabel(points, labels, epsilon, filter_small_clusters=True)
        data[:, 3] = labels
        save_points(os.path.join(first_save_folder, file), data)

        # Step 2: second DBSCAN pass; remove "new organ" fragments that are
        # embedded inside the old organ (fixes new-organ misidentification in 3D_NOD).
        has_new_organ = remove_embedded_new_organ(points, labels, epsilon)
        data[:, 3] = labels
        if not has_new_organ:
            save_points(os.path.join(save_folder, file), data)
            continue

        # Step 3: noise cleanup.  The first pass can propagate labels from a
        # nearby embedded "new organ" to the noise points, leading to misclassification.
        if len(np.unique(labels)) == 1:
            save_path = os.path.join(save_folder, file)
            np.savetxt(save_path, data, delimiter=" ", fmt="%f %f %f %d")
            continue
        dbscan_knn_relabel(points, labels, epsilon, filter_small_clusters=False)
        data[:, 3] = labels
        save_points(os.path.join(save_folder, file), data)


if __name__ == "__main__":
    main()