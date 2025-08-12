import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from sklearn.decomposition import PCA


def renew_label(source_points, train_result):

    centers, source_uni_label = compute_centers(source_points)        # compute the centriods of each organ

    # calculate the distance matrix among organs at the t moment
    distance_matrix = np.zeros((int(centers.shape[0]), int(centers.shape[0])))
    for i in range(centers.shape[0]):
        for j in range(centers.shape[0]):
            distance_matrix[i, j] = np.linalg.norm(centers[i] - centers[j])

    points = train_result[:, :3]
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(points)
    distances, _ = neigh.kneighbors(points)
    nearest_distances = distances[:, 1]
    average_distance = np.mean(nearest_distances)    # calculate the average distance in the point cloud

    # Initialize the parameters of the hierarchical DBSCAN
    min_samples_step = np.linspace(8, 6, 2)
    eps_min = 2 * average_distance
    eps_max = 2.5 * average_distance
    eps_steps = np.linspace(eps_min, eps_max, 2)

    instance_labels = train_result[:, 3]
    unique_labels = np.unique(instance_labels)
    updated_labels = instance_labels.copy()

    single_cluster_points = []
    single_cluster_label = []
    small_cluster_indices = []       # index of points in the clusters that need to be reset label
    threshold = 100

    label_cluster_info = {}
    for label in unique_labels:
        instance_indices = np.where(updated_labels == label)[0]
        instance_points = train_result[instance_indices, :3]

        final_ins_labels = np.full(instance_points.shape[0], -1)
        next_label = 0

        final_ins_labels = clusters_generation(eps_steps, min_samples_step, instance_points, final_ins_labels, next_label)      # 3.5.1 Clusters generation

        if label == 0:                                                                      # 3.5.2 Stem clusters merging
            final_ins_labels = merge_stem_clusters(train_result, updated_labels, label, final_ins_labels, eps_max, None, k=0)

        # Record the clustering results for each label to avoid secondary clustering
        label_cluster_info[label] = {
            "indices": instance_indices,
            "points": instance_points,
            "final_ins_labels": final_ins_labels
        }

        unique_clusters, count_point = np.unique(final_ins_labels[final_ins_labels >= 0], return_counts=True)
        if len(unique_clusters) <= 1:                       # the organ is represented by only one cluster(no screening required)
            single_cluster_points.append(instance_points)
            single_cluster_label.append(int(label))
        else:
                                                                                            # 3.5.3 Cluster Screening
            cluster_connections = judge_end_point(points, updated_labels, label, unique_clusters, final_ins_labels, eps_min)
            label_cluster_info[label]["cluster_connections"] = cluster_connections

            cluster2_ids = [cid for cid, connections in cluster_connections if connections >= 2]
            choice_id = np.setdiff1d(unique_clusters, np.array(cluster2_ids))

            cluster_point_map = dict(zip(unique_clusters, count_point))
            cluster_points = [cluster_point_map[cid] for cid in cluster2_ids]
            choice_num = [cluster_point_map[cid] for cid in choice_id]

            if len(choice_id) == 1 and max(cluster_points) < 3 * choice_num[0]:             # (i) End point-based preliminary screening
                for cluster_id in cluster2_ids:
                    cluster_points_mask = (final_ins_labels == cluster_id)
                    small_cluster_indice = np.where(updated_labels == label)[0][cluster_points_mask]

                    for idx in small_cluster_indice:
                        small_cluster_indices.append(idx)

                choice_points_mask = (final_ins_labels == choice_id[0])
                choice_indice = np.where(updated_labels == label)[0][choice_points_mask]
                largest_cluster_points = points[choice_indice]
                single_cluster_points.append(largest_cluster_points)
                single_cluster_label.append(int(label))
            else:                                                                            # (ii) Screening based on cluster size
                large_clusters = compare_cluster_size(count_point, unique_clusters, threshold)
                label_cluster_info[label]["large_clusters"] = large_clusters

                if len(large_clusters) == 0:
                    largest_cluster = max(unique_clusters, key=lambda c: np.sum(final_ins_labels == c))
                    largest_cluster_points = instance_points[final_ins_labels == largest_cluster]
                    small_cluster_indice = instance_indices[final_ins_labels != largest_cluster]

                    for idx in small_cluster_indice:
                        small_cluster_indices.append(idx)
                    single_cluster_points.append(largest_cluster_points)
                    single_cluster_label.append(int(label))

                if len(large_clusters) == 1:
                    largest_cluster = large_clusters[0]
                    largest_cluster_points = instance_points[final_ins_labels == largest_cluster]
                    small_cluster_indice = instance_indices[final_ins_labels != largest_cluster]

                    for idx in small_cluster_indice:
                        small_cluster_indices.append(idx)
                    single_cluster_points.append(largest_cluster_points)
                    single_cluster_label.append(int(label))

    single_cluster_label = np.array(single_cluster_label).astype(int)
    other_cluster_label = np.setdiff1d(unique_labels, single_cluster_label)
    for lab in other_cluster_label:
        instance_indices = np.where(updated_labels == lab)[0]
        instance_points = train_result[instance_indices, :3]
        single_cluster_points.insert(int(lab), instance_points)

    updated_labels = instance_labels.copy()
    merge_map = {}
    for label in unique_labels:
        info = label_cluster_info[label]
        instance_indices = info["indices"]
        instance_points = info["points"]
        final_ins_labels = info["final_ins_labels"].copy()

        if label == 0:
            final_ins_labels, merge_map = merge_stem_clusters(train_result, updated_labels, label, final_ins_labels, eps_max, merge_map, k=1)
        unique_clusters, count_point = np.unique(final_ins_labels[final_ins_labels >= 0], return_counts=True)

        if len(unique_clusters) > 1:                                                      # (iii) Screening based on relative organ positions
            cluster_connections = info["cluster_connections"]

            cluster2_ids = [cid for cid, connections in cluster_connections if connections >= 2]
            choice_id = np.setdiff1d(unique_clusters, np.array(cluster2_ids))

            cluster_point_map = dict(zip(unique_clusters, count_point))
            cluster_points = [cluster_point_map[cid] for cid in cluster2_ids]
            choice_num = [cluster_point_map[cid] for cid in choice_id]

            if len(choice_id) != 1 or max(cluster_points) >= 3 * choice_num[0]:
                large_clusters = info["large_clusters"]

                multiple_clusters = []
                points_num = []
                if len(large_clusters) > 1:
                    instance_large_clusters = []
                    for cluster in large_clusters:
                        cluster_points = instance_points[final_ins_labels == cluster]
                        instance_large_clusters.append(cluster_points)
                        points_num.append(int(len(cluster_points)))
                    multiple_clusters.extend(instance_large_clusters)

                diff = compute_centriod_diff(multiple_clusters, single_cluster_points, unique_labels, source_uni_label, distance_matrix, label)
                small_cluster_indices = noise_selection(multiple_clusters, points_num, instance_indices,
                                                                final_ins_labels, diff, small_cluster_indices, large_clusters)

    for label, new_idx in merge_map.items():
        updated_labels[new_idx] = label

    updated_labels = hole_filling(small_cluster_indices, train_result, updated_labels, eps_max)         # 3.5.4 Hole filling

    return updated_labels, average_distance


def compute_centers(source_points):            # compute the centriods of each organ
    source_uni_label = np.unique(source_points[:, 3])
    centers = np.full((int(np.max(source_uni_label)) + 1, 3), 100.0)
    for label in source_uni_label:
        label_points = source_points[source_points[:, 3] == label, :3]
        center = label_points.mean(axis=0)
        centers[int(label), :] = center

    centers = np.array(centers)
    centers = centers[~np.all(centers == 100, axis=1)]

    return centers, source_uni_label


def clusters_generation(eps_steps, min_samples_step, instance_points, final_ins_labels, next_label):
    # hierarchical DBSCAN
    for eps, min_samples in zip(eps_steps, min_samples_step):
        db = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(instance_points)
        cluster_labels = db.labels_

        for cluster_label in np.unique(cluster_labels):
            if cluster_label == -1:
                continue

            cluster_mask = (cluster_labels == cluster_label)
            if np.all(final_ins_labels[cluster_mask] != -1):
                continue

            existing_labels = final_ins_labels[cluster_mask][final_ins_labels[cluster_mask] != -1]
            if len(existing_labels) > 0:
                merged_label = np.min(existing_labels)
                final_ins_labels[cluster_mask] = merged_label

                for existing_label in np.unique(existing_labels):
                    if existing_label != merged_label:
                        final_ins_labels[final_ins_labels == existing_label] = merged_label
            else:
                final_ins_labels[cluster_mask] = next_label
                next_label += 1
    return final_ins_labels



def merge_stem_clusters(train_result, updated_labels, label, final_ins_labels, eps_max, merge_map, k):
    indices = np.where(updated_labels == label)[0]
    result = train_result[indices]
    all_indices = np.arange(train_result.shape[0])
    other_indices = np.setdiff1d(all_indices, indices)
    other_points = train_result[other_indices]

    clusters = {}
    for i in range(len(final_ins_labels)):
        cluster_label = final_ins_labels[i]
        if cluster_label != -1:
            if cluster_label not in clusters:
                clusters[cluster_label] = []
            clusters[cluster_label].append(result[i, :3])

    for cluster_label in clusters:
        clusters[cluster_label] = np.array(clusters[cluster_label])

    labels_to_merge = set()
    for first_label in clusters:
        for second_label in clusters:
            if first_label >= second_label:
                continue
            first_cluster = clusters[first_label]
            second_cluster = clusters[second_label]
            if len(first_cluster) <= 15 or len(second_cluster) <= 15:
                continue
            distances = cdist(first_cluster, second_cluster, metric='euclidean')
            min_distance = np.min(distances)
            i, j = np.where(distances == min_distance)

            first_point = first_cluster[i[0], :3]
            second_point = second_cluster[j[0], :3]

            vector = first_point - second_point
            vector = vector / np.linalg.norm(vector)

            first_vector = compute_normal_vector(first_cluster)
            second_vector = compute_normal_vector(second_cluster)

            first_angle = vector.dot(first_vector) / (np.linalg.norm(vector) * np.linalg.norm(first_vector))
            second_angle = vector.dot(second_vector) / (np.linalg.norm(vector) * np.linalg.norm(second_vector))
            knn = NearestNeighbors(radius=3 * eps_max)
            knn.fit(other_points[:, :3])

            neighbors_idx = knn.radius_neighbors([first_point], return_distance=False)[0]
            neighbors_jdx = knn.radius_neighbors([second_point], return_distance=False)[0]
            intersect = np.intersect1d(neighbors_idx, neighbors_jdx)

            if min_distance < 6 * eps_max and intersect.size > 0 and (1 - abs(first_angle)) > 0.7 and (1 - abs(second_angle)) > 0.7:   # Merging condition
                labels_to_merge.add(first_label)
                labels_to_merge.add(second_label)
                if k == 1:
                    new_idx = other_indices[intersect]
                    merge_map[label] = new_idx

    if len(clusters) != 0:
        new_label = max(clusters.keys()) + 1
        for cluster_label in labels_to_merge:
            final_ins_labels[final_ins_labels == cluster_label] = new_label

    if k == 0:
        return final_ins_labels
    else:
        return final_ins_labels, merge_map


def compute_normal_vector(points):
    # Calculate the normal vector of a given set of points using PCA
    pca = PCA(n_components=3)
    pca.fit(points)
    normal_vector = pca.components_[-1]
    return normal_vector




def judge_end_point(points, updated_labels, label, unique_clusters, final_ins_labels, eps_min):
    tree_old = cKDTree(points[updated_labels != label])
    cluster_connections = []

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_points_mask = (final_ins_labels == cluster_id)
            indices = np.where(updated_labels == label)[0][cluster_points_mask]
            continue

        cluster_points = points[updated_labels == label][final_ins_labels == cluster_id]
        if cluster_points.shape[0] < 2:
            continue

        dist_matrix = squareform(pdist(cluster_points))
        max_distance = np.max(dist_matrix)
        i, j = np.where(dist_matrix == max_distance)[0]
        point1, point2 = cluster_points[i], cluster_points[j]

        distances1, _ = tree_old.query(point1.reshape(1, -1), k=1)
        distances2, _ = tree_old.query(point2.reshape(1, -1), k=1)

        connected = np.sum((distances1[0] < 2 * eps_min)) + np.sum((distances2[0] < 2 * eps_min))
        cluster_connections.append((cluster_id, connected))

    return cluster_connections



def compare_cluster_size(count_point, unique_clusters, threshold):
    large_clusters = []
    for i, count in enumerate(count_point):
        if count > threshold:
            large_clusters.append(unique_clusters[i])

    if len(large_clusters) > 0:
        new_list = [count_point[i] for i, cluster in enumerate(unique_clusters) if cluster in large_clusters]
        largest_cluster_count = max(new_list)

        for i in range(len(count_point)):
            if count_point[i] != largest_cluster_count and abs(count_point[i] - largest_cluster_count) <= 30:
                if unique_clusters[i] not in large_clusters:
                    large_clusters.append(unique_clusters[i])
    else:
        for i in range(len(count_point)):
            for j in range(i + 1, len(count_point)):
                if abs(count_point[i] - count_point[j]) <= 30:
                    union_label = np.append(unique_clusters[i], unique_clusters[j])
                    remaining_clusters = np.setdiff1d(unique_clusters, union_label)
                    remaining_counts = [count_point[k] for k in range(len(count_point)) if
                                        unique_clusters[k] in remaining_clusters]
                    if len(remaining_counts) == 0:
                        if unique_clusters[i] not in large_clusters:
                            large_clusters.append(unique_clusters[i])
                        if unique_clusters[j] not in large_clusters:
                            large_clusters.append(unique_clusters[j])
                    else:
                        if (count_point[i] > max(remaining_counts)) and (count_point[j] > max(remaining_counts)):
                            if unique_clusters[i] not in large_clusters:
                                large_clusters.append(unique_clusters[i])
                            if unique_clusters[j] not in large_clusters:
                                large_clusters.append(unique_clusters[j])

    return large_clusters


def compute_centriod_diff(multiple_clusters, single_cluster_points, unique_labels, source_uni_label, distance_matrix, label):
    diff = []
    for multi_clusters in multiple_clusters:
        multi_cluster_centers = np.mean(multi_clusters, axis=0)  # Calculate the center of each large cluster in the instance

        # Calculate the distance between the center of the cluster and the centers of all other organs
        distances = [np.linalg.norm(multi_cluster_centers - np.mean(single_cluster, axis=0)) for single_cluster in single_cluster_points]
        distances = np.array(distances)
        label_index = np.where(unique_labels == int(label))[0][0]
        distances = np.delete(distances, int(label_index))
        corres_distances = distance_matrix[int(label_index)]
        missing_label = np.setdiff1d(source_uni_label, unique_labels)
        if len(missing_label) != 0:
            corres_distances = np.delete(corres_distances, [int(missing_label), int(label_index)])
        else:
            corres_distances = np.delete(corres_distances, int(label_index))
        distance_diff = np.mean((distances - corres_distances) ** 2)
        diff.append(distance_diff)

    return diff


def noise_selection(multiple_clusters, points_num, instance_indices, final_ins_labels, diff, small_cluster_indices, large_clusters):

    if multiple_clusters:
        point_thres = max(points_num) * 0.1
        max_diff_points = max(points_num) - min(points_num)

        min_value = min(diff)
        min_index = diff.index(min_value)
        if max_diff_points <= point_thres:
            small_cluster_indice = instance_indices[final_ins_labels != large_clusters[min_index]]
        else:
            dist_index_point = points_num[min_index]
            diff.remove(min_value)
            if min(diff) / min_value > 3.5:
                small_cluster_indice = instance_indices[final_ins_labels != large_clusters[min_index]]
            else:
                if min_value < min(diff) / 1.5 and dist_index_point > max(points_num) / int(min(diff) / min_value):
                    small_cluster_indice = instance_indices[final_ins_labels != large_clusters[min_index]]
                else:
                    new_index = points_num.index(max(points_num))
                    small_cluster_indice = instance_indices[final_ins_labels != large_clusters[new_index]]

        for idx in small_cluster_indice:
            small_cluster_indices.append(idx)

    return small_cluster_indices


def hole_filling(small_cluster_indices, train_result, updated_labels, eps_max):
    small_cluster_indices = np.array(small_cluster_indices)
    main_points_indices = np.setdiff1d(range(len(train_result)), small_cluster_indices)
    main_points = train_result[main_points_indices, :3]
    main_labels = updated_labels[main_points_indices]

    knn = NearestNeighbors(radius=eps_max)
    knn.fit(main_points)

    while len(small_cluster_indices) > 0:
        newly_labeled = []
        unlabeled_indices = []

        for idx in small_cluster_indices:
            neighbors_idx = knn.radius_neighbors([train_result[idx, :3]], return_distance=False)[0]

            if len(neighbors_idx) > 0:
                neighbor_labels = main_labels[neighbors_idx].astype(int)
                most_common_label = np.bincount(neighbor_labels).argmax()
                updated_labels[idx] = most_common_label
                newly_labeled.append(idx)
            else:
                unlabeled_indices.append(idx)

        small_cluster_indices = unlabeled_indices

        if len(newly_labeled) == 0:
            break

        newly_labeled_points = train_result[newly_labeled, :3]
        newly_labeled_labels = updated_labels[newly_labeled]

        main_points = np.vstack([main_points, newly_labeled_points])
        main_labels = np.concatenate([main_labels, newly_labeled_labels])
        knn.fit(main_points)

    return updated_labels