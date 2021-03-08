import numpy as np
# np.random.seed(0)
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
    
Notes
    * something like - cluster in one of the high dimensions where the furthest cluster to init is 
        in the same "cluster" might be happening? 
"""


class KMeans:
    def __init__(self, k):
        self.k = k
        self.centers = None
        self.init_centers = None
        self.inertia_ = None

    @staticmethod
    def sse(center, data):
        return ((data - center) ** 2).sum(axis=-1).T

    @staticmethod
    def in_cluster_sse(centers, labels, data):
        error = 0
        for i, center in enumerate(centers):
            error += KMeans.sse(center, data[np.where(labels == i)]).sum()
        return error

    @staticmethod
    def get_cluster_labels(centers, X):
        return np.argmin(KMeans.sse(centers, X), axis=1)

    def center_init(self, X):
        m, d = X.shape
        centers = np.random.rand(self.k, 1, d)
        # centers = X[np.random.choice(m, size=self.k, replace=False)].reshape(self.k, 1, d)   # get around issue but need to check
        return centers

    def fit(self, X):
        m, d = X.shape

        # TEST
        # X = np.arange(30).reshape(-1, 3) + 5
        # centers = np.arange(15).reshape(5, 1, 3)
        # m, d = X.shape
        # centers = np.random.rand(5, 1, 3)

        # init centers, initial calculations
        centers = self.center_init(X)
        self.init_centers = centers
        thresh = 1e-2

        clust_labels = self.get_cluster_labels(centers, X)
        error = self.in_cluster_sse(centers, clust_labels, X)

        # find new centers until convergence
        converged = False
        while not converged:
            old_centers = centers

            centers = np.array([X[np.where(clust_labels==i)].mean(axis=0) for i in range(centers.shape[0])])\
                .reshape(self.k, 1, d)

            # reset to old center if no points assigned
            for clust_num, row in enumerate(centers):
                if np.any(np.isnan(row)):
                    # for now just reseting to old center - seems like this is what andrew did
                    centers[clust_num] = old_centers[clust_num]

            clust_labels = self.get_cluster_labels(centers, X)
            # TODO: for this andrew did the error between new and old centroids - is it the same thing?
            new_error = self.in_cluster_sse(centers, clust_labels, X)
            converged = error - new_error < thresh
            # print(error - new_error)
            error = new_error

        self.inertia_ = error
        self.centers = centers.reshape(self.k, d)
        return self


    def transform(self, X):
        return self.get_cluster_labels(self.centers.reshape(self.k, 1, X.shape[1]), X)


class KMeanspp(KMeans):
    # can just override init for this I think
    def __init__(self, k):
        super().__init__(k)

    def center_init(self, X):
        # todo: is the distance we care about here just vanila distance? - D2 is then SSE
        m, d = X.shape
        for i in range(self.k):
            if i == 0:
                centers = X[np.random.choice(m, size=1, replace=False)].reshape(1, 1, d)
            else:
                # new_center = X[np.argmax(self.sse(centers, X).min(axis=1))].reshape(1, 1, d) # max spread
                probabilities = self.sse(centers, X).min(axis=1)        # smallest sq dist to any clust
                probabilities = probabilities / probabilities.sum()     # normalize - sum to 1
                new_center = X[np.random.choice(m, size=1, replace=False, p=probabilities)].reshape(1, 1, d)
                centers = np.append(centers, new_center, axis=0)
        self.centers = centers
        return centers


class KMeans1D:
    def __init__(self):
        pass


def main(file, path, k, init):
    X = np.genfromtxt(path, delimiter=',')
    # TODO: what about y? is that just for our own viz/debugging?

    if init == "random":
        kmeans = KMeans(k=k)
        kmeans.fit(X)
        print(kmeans.centers.shape)
        # print(kmeans.centers)
        labels = kmeans.transform(X)
    elif init == "k-means++":
        kmeanspp = KMeanspp(k=k)
        kmeanspp.fit(X)
        print(kmeanspp.centers.shape)
        # kmeanspp.center_init(X)
    elif init == '1d':
        pass
    else:
        raise ValueError("Invalid Argument for \"init\" ")


if __name__ == "__main__":
    file, path, k, init = sys.argv
    main(file, path, int(k), init)





