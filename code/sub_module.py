import numpy as np
import pandas as pd
import os

from ICP import ICP
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist


def Inhomogeneous_sampling(sample_count, points, FPS):
    labels = points[:, -1:]
    uni_label, counts = np.unique(labels, return_counts=True)

    average_count = sample_count / len(uni_label)           # Compute the average count of each label
    total_sampled = 0
    all_sampled_points = []
    for lab in uni_label:
        points_for_label = points[np.where(labels == lab)[0]]
        count_for_label = len(points_for_label)

        if count_for_label < average_count:
            all_sampled_points.append(points_for_label)
            total_sampled += count_for_label
        else:
            continue

    if len(all_sampled_points) == 0:
        sample_points = FPS._call__(points, sample_count)
    else:
        sample_points = np.vstack(all_sampled_points)
        remaining_needed = sample_count - total_sampled
        remaining_points = FPS._call__(points, remaining_needed)
        sample_points = np.vstack((sample_points, remaining_points))

    return sample_points, uni_label


def ICP_Registration_main(sample_points, sample_target_points, sample_count, Backward_Registration):
    if Backward_Registration:
        A = sample_target_points[:, :3]
        B = sample_points[:, :3]
    else:
        A = sample_points[:, :3]
        B = sample_target_points[:, :3]

    target_label = sample_target_points[:, -1].reshape(-1, 1)
    source_label = sample_points[:, -1].reshape(-1, 1)

    NN = sample_count
    all_label = np.vstack((source_label, target_label))
    num_tests = 100

    icp_algorithm = ICP(A, B, max_iterations=50, tolerance=1e-6)
    icp_algorithm.test_best_fit(num_tests)
    best_T = icp_algorithm.test_icp(num_tests)

    C = np.ones((NN, 4))
    C[:, 0:3] = A

    D = np.ones((NN, 4))
    D[:, 0:3] = B

    E = np.dot(best_T, C.T).T

    if Backward_Registration:
        all_point = np.vstack((D, E))
    else:
        all_point = np.vstack((E, D))
    all_point[:, 3:] = all_label

    M = all_point[:sample_count, :]
    N = all_point[sample_count:, :]

    return M, N, all_label

def visualize_backward(iteration, error, diff, X, Y, label, source_file_norm, target_file_norm, CPD_folder):

    tolerance = 0.001
    max_iteration = 100
    if diff < tolerance or iteration > max_iteration:
        save_last_name = CPD_folder + "\\" + "_".join(source_file_norm.split("_")[:-1]) + " & " + target_file_norm.split("_")[-2] + ".txt"
        new_coo = np.vstack((X, Y))
        np.savetxt(save_last_name, np.hstack((new_coo, label)), fmt="%f %f %f %d", delimiter=" ")

def visualize_forward(iteration, error, diff, X, Y, label, source_file_norm, target_file_norm, CPD_folder):

    tolerance = 0.001
    max_iteration = 100
    if diff < tolerance or iteration > max_iteration:
        save_last_name = CPD_folder + "\\" + "_".join(source_file_norm.split("_")[:-1]) + " & " + target_file_norm.split("_")[-2] + ".txt"
        new_coo = np.vstack((Y, X))
        np.savetxt(save_last_name, np.hstack((new_coo, label)), fmt="%f %f %f %d", delimiter=" ")


def create_input(data, sample_count, graph_path):

    input_len = len(data)
    arr = []
    label = []
    number = []

    for line in data[0:]:
        x, y, z = list(line[:3])
        l = list(line[3:4].astype(int))
        arr.append([x, y, z])
        label.append(l)

    graph_test = []

    if input_len == 2 * sample_count:
        A = kneighbors_graph(arr, 5, mode='distance', include_self=False)
    else:
        A = kneighbors_graph(arr, 20, mode='distance', include_self=False)

    C = A.toarray()
    point_size = int(input_len)


    for i in range(point_size):
        for j, k in zip(A.getrow(i).indices, A.getrow(i).data):
            initial = i
            end = j
            weight = k
            graph_test.append([initial, end, weight])

    restore_path = os.path.join(graph_path, "a")
    if not os.path.exists(restore_path):
        os.makedirs(restore_path)

    f2 = open(os.path.join(restore_path, "a.cites"), 'w')
    for knnnum in graph_test:
        print(knnnum[0], '	', knnnum[1], end='\n', file=f2, sep='')
    f2.close()

    for i in range(point_size):
        number.append(i)

    newLabel = [int(x) for item in label for x in item]

    arrMatrix = np.mat(arr)
    data_xyz_txt = arrMatrix
    data_xyzDF = pd.DataFrame(data_xyz_txt, dtype=float)

    np.savetxt(graph_path + "\\" + "sparse_matrix.txt", C, fmt='%.6f', delimiter=' ')
    np.savetxt(graph_path + "\\" + "coordinate_matrix.txt", arrMatrix, fmt='%.06f', delimiter=' ')

    weight_arr = []
    for i in C:
        for j in i:
            if (j != 0):
                weight_arr.append(j)

    if input_len == 2 * sample_count:
        K = 5
    else:
        K = 20
    h = sum(weight_arr) / (point_size * K)

    each_row_weight = []
    row_weight = 0
    weight = 0

    for i in range(point_size):
        for j, k in zip(A.getrow(i).indices, A.getrow(i).data):
            initial = i
            end = j
            weight = np.exp(-k / (2 * h))
            row_weight += weight
        each_row_weight.append(row_weight)
        row_weight = 0

    for i in range(point_size):
        for j, k in zip(A.getrow(i).indices, A.getrow(i).data):
            initial = i
            end = j
            weight = np.exp(-k / (2 * h)) / each_row_weight[i]
            C[i][j] = weight

    data_txt = C
    data_txtDF = pd.DataFrame(data_txt, dtype=float)

    data_txtDF = pd.concat([data_txtDF, data_xyzDF], axis=1)
    data_txtDF.insert(loc=0, column="number", value=number)
    data_txtDF.insert(loc=point_size + 4, column="label", value=newLabel)

    xls_file_path = graph_path + "\\" + "content.xls"
    data_txtDF.to_csv(xls_file_path, index=False, header=None, sep=' ')
    df = pd.read_csv(xls_file_path, engine='python', header=None)

    content_file_path = os.path.join(restore_path, "a.content")
    with open(content_file_path, 'w') as f:
        for index, row in df.iterrows():
            line = ' '.join(str(val) for val in row)
            f.write(f"{line}\n")

    print(f"Data has been successfully written to {content_file_path}")



def Intersection(sample_count, Backward_result, Forward_result):
    first_array = []
    second_array = []

    for i in range(sample_count):
        if Backward_result[i, 3] == Forward_result[i, 3]:
            first_array.append([*Backward_result[i]])
        else:
            second_array.append([*Backward_result[i], Forward_result[i, 3]])

    first_array = np.array(first_array)
    second_array = np.array(second_array)

    return first_array, second_array


def compute_label_diff(sample_points, result_points):           # Filering 中 Case 1的计算
    unique_labels = np.unique(sample_points[:, 3])
    total_diff = 0

    for label in unique_labels:
        sample_count = np.sum(sample_points[:, 3] == label)
        result_count = np.sum(result_points[:, 3] == label)
        total_diff += abs(result_count - sample_count)

    return total_diff


def Filtering(first_array, second_array, sample_points, sample_count, Backward_result, Forward_result, FPS):
    unique_labels, label_count = np.unique(first_array[:, 3], return_counts=True)
    source_label = sample_points[:, -1]
    source_label_num = len(np.unique(source_label))

    if (sample_count - len(first_array)) >= sample_count / source_label_num and len(
            unique_labels) != source_label_num:                                      # Case 1
        b_diff = compute_label_diff(sample_points, Backward_result)
        c_diff = compute_label_diff(sample_points, Forward_result)
        if b_diff <= c_diff:
            result_array = Backward_result
        else:
            result_array = Forward_result
    else:                                                                             # Case 2
        standard = int(sample_count / (2 * len(unique_labels)))

        filtered_labels = unique_labels[label_count < standard]
        filtered_counts = label_count[label_count < standard]
        sorted_indices = np.argsort(filtered_counts)
        low_count_labels = filtered_labels[sorted_indices]
        if len(unique_labels) != source_label_num:
            missing_labels = np.setdiff1d(np.unique(source_label), unique_labels)
            low_count_labels = np.append(missing_labels, low_count_labels)

        if len(low_count_labels) > 0:
            add_result = []
            for uni_lab in low_count_labels:

                first_label_points = second_array[second_array[:, 3] == uni_lab]
                second_label_points = second_array[second_array[:, 4] == uni_lab]

                if len(first_label_points) != 0 and len(second_label_points) == 0:
                    label_point = first_label_points
                elif len(first_label_points) == 0 and len(second_label_points) != 0:
                    label_point = second_label_points
                elif len(first_label_points) != 0 and len(second_label_points) != 0:
                    label_point = np.vstack((first_label_points, second_label_points))
                    second_array = second_array[(second_array[:, 3] != uni_lab) & (second_array[:, 4] != uni_lab)]
                else:
                    label_point = None

                if label_point is not None:
                    for point in label_point:
                        add_result.append([*point[:3], uni_lab])

            if add_result:
                add_result = np.array(add_result)
                new_first_array = np.vstack((first_array, add_result))
            else:
                new_first_array = first_array
        else:
            new_first_array = first_array

        if len(new_first_array) < sample_count:
            indices = np.random.choice(len(new_first_array), sample_count - len(new_first_array), replace=True)
            result_array = new_first_array[indices]
            result_array = np.vstack((new_first_array, result_array))
        else:
            result_array = FPS._call__(new_first_array, sample_count)

    return result_array


def eliminate_BFL(new_coordinates, train_result, source_points, new_label, new_t_labels, used_t_labels, average_distance):
    new_points = new_coordinates[new_coordinates[:, 3] == new_label, :3]
    merged = False

    for t_label in new_t_labels:
        if t_label in used_t_labels:
            continue  # t_label has been assigned to other new label

        propagated_points = train_result[train_result[:, 3] == t_label, :3]
        ini_points = source_points[source_points[:, 3] == t_label, :3]
        if len(propagated_points) > 0:
            distances = cdist(new_points, propagated_points, metric='euclidean')
            min_dist = np.min(distances)

            other_points = train_result[train_result[:, 3] != t_label, :3]
            other_dist = cdist(new_points, other_points, metric='euclidean')
            other_min_dist = np.min(other_dist)
            if (min_dist < 3 * average_distance) and len(propagated_points) < len(ini_points):
                new_coordinates[new_coordinates[:, 3] == new_label, 3] = t_label
                used_t_labels.add(t_label)
                merged = True
            else:
                if (min_dist < 1.5 * average_distance) and (other_min_dist > 1.5 * average_distance) and len(
                        propagated_points) < 200:
                    new_coordinates[new_coordinates[:, 3] == new_label, 3] = t_label
                    used_t_labels.add(t_label)
                    merged = True

    return merged