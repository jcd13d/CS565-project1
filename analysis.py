from p1_code import KMeans, KMeanspp
import numpy as np
import itertools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as KMeansSK
from sklearn import datasets
import matplotlib.pyplot as plt
import os


def plots(X, labels, title, plots_dir):
    unique_labels = np.unique(labels)
    np.random.shuffle(unique_labels)
    for label in unique_labels:
        x = X[np.where(labels == label)]
        plt.scatter(x[:,0], x[:,1], s=1, label=label, alpha=0.2)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "{0}_clusters".format(title)))
    plt.close()


def elbow_plot(max_k, data, kmeans_obj, title, show=False):
    # Elbow Plot
    # TODO: why is loss jagged! shouldnt go up with more clusters?? Unless stuck in local optima?
    # TODO: looks normal with the 2d simple data!
    loss = []
    for k_val in range(1, max_k + 1):
        loss.append(kmeans_obj(k_val).fit(data).inertia_)
    plt.plot([i for i in range(len(loss))], loss)
    plt.title(title)
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join("plots", "{0}.jpg".format(title)))
    plt.close()


def plot_centers(X, centers, title, show=False):
    # Plotting Init points
    plt.scatter(X[:,0], X[:,1], s=1)
    plt.scatter(centers[:, 0], centers[:, 1], alpha=0.4)
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join("plots", "{0}.jpg".format(title)))
    plt.close()


def disagreement_distance(c1, c2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    assert c1.shape[0] == c1.shape[0]
    sum = 0
    index = np.arange(c1.shape[0])
    for x1, x2 in itertools.combinations(index, 2):
        sum += disagreement(x1, c1, x2, c2)
    return sum


def disagreement(x1, c1, x2, c2):
    disagree_1 = (c1[x1] == c1[x2]) and (c2[x1] != c2[x2])
    disagree_2 = (c1[x1] != c1[x2]) and (c2[x1] == c2[x2])
    disagree = disagree_2 or disagree_1
    return int(disagree)


if __name__ == "__main__":
    path = "X.csv"
    true_labels_path = 'y.csv'
    out_path = "plots"
    k = 5
    max_k_elbow = 20
    use_dummy_data = False
    dummy_data_centers = 4

    X = np.genfromtxt(path, delimiter=',')
    y_true = np.genfromtxt(true_labels_path, delimiter=',')

    # 2d debugging data
    if use_dummy_data:
        X, y_true = datasets.make_blobs(
            10000, centers=dummy_data_centers, n_features=dummy_data_centers, random_state=0)
        k = dummy_data_centers

    elbow_plot(max_k=max_k_elbow, data=X, kmeans_obj=KMeansSK, title="sklearn_kmeans_elbow")
    elbow_plot(max_k=max_k_elbow, data=X, kmeans_obj=KMeanspp, title="kmeanspp_elbow")
    elbow_plot(max_k=max_k_elbow, data=X, kmeans_obj=KMeans, title="random_elbow")

    kmeans = KMeanspp(k=k)
    calc_labels = kmeans.fit(X).transform(X)

    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    init_centers = kmeans.init_centers.reshape(k, -1)
    init_centers = pca.transform(init_centers)

    plot_centers(X, init_centers, "initial_centers_2d")

    plots(X, y_true, "Actual_Classes_Major_Axes", out_path)
    plots(X, calc_labels, "Calc_Classes_Major_Axes", out_path)

