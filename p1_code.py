import numpy as np
np.random.seed(0)
import sys

"""
Andrew Notes
* should be familiar with a couple things in numpy
    * become familiar with reshape
    * he did np.arange(30) that way not random
    * if you dont know one of the numbers -1 means nupy figures it out
    * x - cj.reshape(1, -1)
    * stacked clusters centers on first axis made 3d to do all at once
    * square 
    * sum over dim axis (last one)
    
    
Questions
    * I though we are minimizing within cluster SSE - is SSE between centroids the same? as done in lab
    * Cant some init cluster centers randomly just never be assignd any points? 
"""


class KMeans:
    def __init__(self, k):
        self.k = k
        self.centers = None
        pass

    def center_init(self, X):
        m, d = X.shape
        # TODO should we pick centers randomly? or pick randomly k points from the set as centers?
        # this would get aroudn the initialization issue where some centers are never interacted with
        # centers = np.random.rand(self.k, 1, d)
        centers = X[np.random.choice(m, size=self.k, replace=False)].reshape(self.k, 1, d)   # get around issue but need to check
        return centers

    def fit(self, X):
        m, d = X.shape

        def sse(center, data):
            return ((data - center)**2).sum(axis=-1).T

        def in_cluster_sse(centers, labels, data):
            error = 0
            for i, center in enumerate(centers):
                error += sse(center, data[np.where(labels == i)]).sum()
            return error


        # set assign each point to the cluster that minimizes its distance
            # vectorized - calc dist from each point to center resulting in matrix of dist for ea center
            # columnar argmin of that matrix is assigned cluster
        # using l2 squared = (x - mu)^2


        # TEST
        # X = np.arange(30).reshape(-1, 3) + 5
        # centers = np.arange(15).reshape(5, 1, 3)
        # centers = np.random.rand(5, 1, 3)
        centers = self.center_init(X)

        thresh = 1e-2

        clust_labels = np.argmin(sse(centers, X), axis=1)
        old_error = in_cluster_sse(centers, clust_labels, X)
        centers = np.array([X[np.where(clust_labels == i)].mean(axis=0) for i in range(centers.shape[0])]) \
            .reshape(self.k, 1, d)
        clust_labels = np.argmin(sse(centers, X), axis=1)
        error = in_cluster_sse(centers, clust_labels, X)

        while old_error - error > thresh:
            print(old_error - error)
            old_error = error

            centers = np.array([X[np.where(clust_labels==i)].mean(axis=0) for i in range(centers.shape[0])])\
                .reshape(self.k, 1, d)

            # for clust_num, row in enumerate(new_centers):
            #     if np.any(np.isnan(row)):
            #         for now just reseting to old center
                    # new_centers[clust_num] = centers[clust_num]

            clust_labels = np.argmin(sse(centers, X), axis=1)
            # TODO: for this andrew did the error between new and old centroids - is it the same thing?
            error = in_cluster_sse(centers, clust_labels, X)

        self.centers = centers.reshape(self.k, d)

        #SEEMS TO BE WORKING! annoyed that i write that twice
        # TODO got a better way to do this write once in jupyter



    def transform(self, X):
        # label based on learned cluster centers - basically first part of fit
        pass


class KMeanspp(KMeans):
    # can just override init for this I think
    def __init__(self):
        pass


class KMeans1D:
    def __init__(self):
        pass


def main(file, path, k, init):
    X = np.genfromtxt(path, delimiter=',')
    # TODO: what about y? is that just for our own viz/debugging?

    kmeans = KMeans(k=k)
    kmeans.fit(X)
    print(kmeans.centers.shape)


if __name__ == "__main__":
    file, path, k, init = sys.argv
    main(file, path, int(k), init)





