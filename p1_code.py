import numpy as np
from sklearn.decomposition import PCA
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
    * can error ever go up when adding clusters? local optima? Show SKlearn example 
        * yes it can, if dataset doesnt follow kmeans assumptions (like spherical clusters)
    * KMeans++ probability waiting is SSE, no?
        * yes, cahnge in centers is equivalent though
    * I thought we are minimizing within cluster SSE - is SSE between centroids the same? as done in lab
        * the objective function for KM is SSE within cluster right? So this should be what 
            we compare to the threshold? Is change in centers equivalent? (what i saw in lab)
        * yes 
    * Cant some init cluster centers randomly just never be assignd any points? 
        * for random initialization - they just never are assigned points so they are reassigned
            to original cluster
        * yes, possible
            * assumes clusters are spherical
            * thats why we can see stuff like that 
            * our assumed structure is bad
    * is midterm review acutally due? is it a homework?
        * will get some bonus points if we do it 
        * 20% for no answer was an oversight
        
    * AM I DOING PROBABILITY WEIGHTING RIGHT?
    * does 1d need to use the "trick"?
    * add max iter
    
Notes
    * next: 
        * need to see if I need to use the trick to speed up 1d
        * figure out the disagreement distance and implement
        * produce desired plots 
        * write report 
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
        converged = False

        # find new centers until convergence
        while not converged:
            old_centers = centers

            centers = np.array([X[np.where(clust_labels == i)].mean(axis=0) for i in range(centers.shape[0])]) \
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
                probabilities = self.sse(centers, X).min(axis=1)  # smallest sq dist to any clust
                probabilities = probabilities / probabilities.sum()  # normalize - sum to 1
                new_center = X[np.random.choice(m, size=1, replace=False, p=probabilities)].reshape(1, 1, d)
                centers = np.append(centers, new_center, axis=0)
        self.centers = centers
        return centers


class KMeans1D(KMeans):
    # TODO: do i need to do "trick"
    def __init__(self, k):
        super(KMeans1D, self).__init__(k)
        # self.k = k
        # self.centers = None

    @staticmethod
    def dimension_reduction(X):
        pca = PCA(n_components=1)
        return pca.fit_transform(X)

    @staticmethod
    def sse_rep(points):
        mu = np.mean(points)
        return np.sum((points - mu) ** 2)

    def fit(self, X):
        X = self.dimension_reduction(X)
        # test data
        X = np.sort(X)
        # X = np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
        # X = np.array([0, 1, 2, 8, 9, 10, 15, 16, 17, 22, 23, 24]).reshape(-1, 1)
        m, d = X.shape

        cost_table = np.zeros((m, self.k))
        selection_table = np.zeros_like(cost_table)

        for col in range(selection_table.shape[1]):
            selection_table[:, col] = col

        for i in range(0, m):
            sse = self.sse_rep(X[:i + 1])
            cost_table[i, 0] = sse
            selection_table[i, 0] = 0

        for l in range(1, self.k):
            for i in range(l + 1, m):
                candidate_costs = np.array(
                    [cost_table[j, l - 1] + self.sse_rep(X[j + 1:i + 1]) for j in range(l - 1, i - 1)])
                selection_table[i, l] = np.argmin(candidate_costs) + l
                cost_table[i, l] = np.min(candidate_costs)

        j = m - 1
        selections = []
        for k in range(self.k - 1, -1 ,-1):
            j = int(selection_table[j, k])
            selections.append(j)

        centers = list()
        selections = sorted(selections)
        for left, right in zip(selections[:-1], selections[1:]):
            centers.append(np.mean(X[left:right]))
        centers.append(np.mean(X[right:m]))

        self.centers = np.array(centers)

    def transform(self, X):
        X = self.dimension_reduction(X)
        return self.get_cluster_labels(self.centers.reshape(self.k, 1, X.shape[1]), X)

def write_output_justin(X, labels, title):
    data = np.append(X, labels.reshape(-1, 1), axis=1)
    np.savetxt(title, data, delimiter=",")


def main(file, path, k, init):
    X = np.genfromtxt(path, delimiter=',')
    # TODO: what about y? is that just for our own viz/debugging?

    if init == "random":
        kmeans = KMeans(k=k)
        kmeans.fit(X)
        labels = kmeans.transform(X)
        write_output_justin(X, labels, "random.txt")
    elif init == "k-means++":
        kmeanspp = KMeanspp(k=k)
        kmeanspp.fit(X)
        labels = kmeanspp.transform(X)
        write_output_justin(X, labels, "kpp.txt")
    elif init == '1d':
        kmeans = KMeans1D(k=k)
        kmeans.fit(X)
        labels = kmeans.transform(X)
        write_output_justin(X, labels, "1d.txt")
    else:
        raise ValueError("Invalid Argument for \"init\" ")


if __name__ == "__main__":
    file, path, k, init = sys.argv
    main(file, path, int(k), init)
