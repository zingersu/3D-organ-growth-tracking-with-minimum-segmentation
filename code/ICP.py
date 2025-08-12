import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

class ICP:
    def __init__(self, A, B, max_iterations=50, tolerance=0.000001):
        self.A = A
        self.B = B
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def best_fit_transform(self, A, B):
        assert A.shape == B.shape
        m = A.shape[1]               # get number of dimensions

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = centroid_B.T - np.dot(R, centroid_A.T)

        T = np.identity(m + 1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t

    def nearest_neighbor(self, src, dst):

        assert src.shape == dst.shape
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    def icp(self, init_pose=None):

        assert self.A.shape == self.B.shape
        m = self.A.shape[1]                # get number of dimensions

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m + 1, self.A.shape[0]))
        dst = np.ones((m + 1, self.B.shape[0]))
        src[:m, :] = np.copy(self.A.T)
        dst[:m, :] = np.copy(self.B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        for i in range(self.max_iterations):
            distances, indices = self.nearest_neighbor(src[:m, :].T, dst[:m, :].T)

            # compute the transformation between the current source and nearest destination points
            T, _, _ = self.best_fit_transform(src[:m, :].T, dst[:m, indices].T)

            src = np.dot(T, src)                          # update the current source

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < self.tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        T, _, _ = self.best_fit_transform(self.A, src[:m, :].T)

        return T, distances, i

    def test_icp(self, num_tests):
        total_time = 0
        min_avg_distance = float('inf')
        best_T = None

        for i in range(num_tests):
            # Run ICP
            start = time.time()
            T, distances, iterations = self.icp(init_pose=None)

            avg_distance = np.mean(distances)
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                best_T = T

            total_time += time.time() - start

        print('minimum average distance: {:.8f}'.format(min_avg_distance))
        print('corresponding best transformation:')
        print(best_T)

        return best_T

    def test_best_fit(self, num_tests):
        total_time = 0

        for i in range(num_tests):
            start = time.time()
            T, R1, t1 = self.best_fit_transform(self.A, self.B)
            total_time += time.time() - start

            # Make C a homogeneous representation of B
            N = self.B.shape[0]
            C = np.ones((N, 4))
            C[:, 0:3] = self.B

            # Transform C
            C = np.dot(T, C.T).T

        print('best fit time: {:.3f}'.format(total_time / num_tests))
        return