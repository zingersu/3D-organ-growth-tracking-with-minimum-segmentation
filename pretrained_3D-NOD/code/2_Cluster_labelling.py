import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

MIN_CLUSTER_SIZE = 8

def sort_by_number(file_name):
    return int(os.path.splitext(file_name)[0].split("_")[1])

def save_points(path, data):
    np.savetxt(path, data, delimiter=" ", fmt="%f %f %f %d")

def average_nearest_neighbor_distance(points):
    """Mean distance from each point to its nearest neighbor."""
    neigh = NearestNeighbors(n_neighbors=2).fit(points)
    distances, _ = neigh.kneighbors(points)
    return np.mean(distances[:, 1])


def hierarchical_dbscan(instance_points, eps_steps, min_samples_step):
    """
    Layered DBSCAN: run DBSCAN from a small eps to a large eps, merging
    clusters that overlap with clusters found at an earlier (smaller) eps.
    This performs better than a single-layer DBSCAN.

    Returns an array of instance labels (-1 means unassigned).
    """
    final_ins_labels = np.full(instance_points.shape[0], -1)
    next_label = 0

    for eps, min_samples in zip(eps_steps, min_samples_step):
        cluster_labels = DBSCAN(
            eps=eps, min_samples=int(min_samples)
        ).fit(instance_points).labels_

        for cluster_label in np.unique(cluster_labels):
            if cluster_label == -1:
                continue
            cluster_mask = cluster_labels == cluster_label

            if np.all(final_ins_labels[cluster_mask] != -1):
                continue

            existing_labels = final_ins_labels[cluster_mask][
                final_ins_labels[cluster_mask] != -1
            ]
            if len(existing_labels) > 0:
                # Merge this cluster into the smallest existing label.
                merged_label = np.min(existing_labels)
                final_ins_labels[cluster_mask] = merged_label
                for existing_label in np.unique(existing_labels):
                    if existing_label != merged_label:
                        final_ins_labels[
                            final_ins_labels == existing_label
                        ] = merged_label
            else:
                final_ins_labels[cluster_mask] = next_label
                next_label += 1

    return final_ins_labels


def nearest_cluster_for_noise(instance_points, final_ins_labels, unique_clusters):
    """
    For every noise point (label -1) find the closest cluster among
    ``unique_clusters``.

    Returns (noise_indices, nearest_cluster_ids), where the k-th entry of
    nearest_cluster_ids corresponds to noise_indices[k].
    """
    noise_indices = np.where(final_ins_labels == -1)[0]
    nearest_cluster_ids = np.empty(len(noise_indices), dtype=int)

    for k, noise_idx in enumerate(noise_indices):
        noise_point = instance_points[noise_idx]
        min_distance = float("inf")
        nearest_cluster = -1

        for cluster in unique_clusters:
            cluster_points = instance_points[final_ins_labels == cluster]
            distances = np.linalg.norm(cluster_points - noise_point, axis=1)
            closest_distance = np.min(distances)
            if closest_distance < min_distance:
                min_distance = closest_distance
                nearest_cluster = cluster

        nearest_cluster_ids[k] = nearest_cluster

    return noise_indices, nearest_cluster_ids


def relabel_new_organ(points, labels, eps_steps, min_samples_step):
    """
    Give every point with semantic label 1 a per-instance cluster id:
      1. run layered DBSCAN on the label-1 points;
      2. drop clusters smaller than ``MIN_CLUSTER_SIZE``;
      3. assign each DBSCAN noise point to its nearest cluster.

    Returns the updated label vector (same length as ``labels``).
    """
    instance_indices = np.where(labels == 1)[0]
    instance_points = points[instance_indices]

    final_ins_labels = hierarchical_dbscan(instance_points, eps_steps, min_samples_step)

    unique_clusters, count_point = np.unique(final_ins_labels[final_ins_labels >= 0], return_counts=True)
    small_clusters = unique_clusters[count_point < MIN_CLUSTER_SIZE]
    for small_cluster in small_clusters:
        final_ins_labels[final_ins_labels == small_cluster] = 0

    new_labels = labels.copy()
    for i, idx in enumerate(instance_indices):
        if final_ins_labels[i] != -1:
            new_labels[idx] = final_ins_labels[i] + 1

    noise_indices, nearest_cluster_ids = nearest_cluster_for_noise(instance_points, final_ins_labels, unique_clusters)
    for noise_idx, nearest_cluster in zip(noise_indices, nearest_cluster_ids):
        new_labels[instance_indices[noise_idx]] = nearest_cluster + 1

    return new_labels


def main():
    home_path = os.path.dirname(os.getcwd())
    middle_result = os.path.join(home_path, "Middle_results")
    ini_folder = os.path.join(middle_result, "Final_Filter_Predict_results")
    save_folder = os.path.join(middle_result, "Results_cluster_labelling")
    os.makedirs(save_folder, exist_ok=True)

    ini_files = sorted(os.listdir(ini_folder), key=sort_by_number)
    min_samples_step = np.linspace(8, 4, 3)

    for file in ini_files:
        data = np.loadtxt(os.path.join(ini_folder, file))
        points = data[:, :3]
        labels = data[:, 3].copy()
        uni_label = np.unique(labels)

        average_distance = average_nearest_neighbor_distance(points)
        eps_steps = np.linspace(2 * average_distance, 3 * average_distance, 3)

        save_path = os.path.join(save_folder, file)

        if len(uni_label) == 1:
            save_points(save_path, data)
            continue

        new_labels = relabel_new_organ(points, labels, eps_steps, min_samples_step)
        data = np.hstack((points, new_labels.reshape(-1, 1)))
        save_points(save_path, data)

if __name__ == "__main__":
    main()