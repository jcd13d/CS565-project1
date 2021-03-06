from p1_code import KMeans, KMeanspp
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt
import os


def plots(X, labels, title, plots_dir):
    print(title)
    print(np.unique(labels))
    for label in np.unique(labels):
        x = X[np.where(labels == label)]
        plt.scatter(x[:,0], x[:,1], s=1, label=label)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "{0}_clusters".format(title)))
    plt.close()


if __name__ == "__main__":
    path = "X.csv"
    true_labels_path = 'y.csv'
    out_path = "plots"

    X = np.genfromtxt(path, delimiter=',')
    y_true = np.genfromtxt(true_labels_path, delimiter=',')

    # 2d debugging data
    X, y_true = datasets.make_blobs(10000, random_state=0)
    centers = KMeanspp(3).center_init(X)
    k = 3


    # calc_labels = KMeans(k=3).fit(X).transform(X)
    # calc_labels = KMeanspp(k=3).fit(X).transform(X)
    kmeans = KMeanspp(k=k)
    # kmeans = KMeans(k=k)
    calc_labels = kmeans.fit(X).transform(X)
    print(np.unique(calc_labels))

    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    init_centers = kmeans.init_centers.reshape(k, -1)
    init_centers = pca.transform(init_centers)

    # Plotting Init points
    plt.scatter(X[:,0], X[:,1], s=1)
    plt.scatter(init_centers[:, 0], init_centers[:, 1], alpha=0.4)
    plt.show()
    plt.close()

    plots(X, y_true, "Actual_Classes_Major_Axes", out_path)
    plots(X, calc_labels, "Calc_Classes_Major_Axes", out_path)

