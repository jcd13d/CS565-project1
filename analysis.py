from p1_code import KMeans, KMeanspp
import numpy as np
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


def elbow_plot(max_k, data, kmeans_obj, title):
    # Elbow Plot
    # TODO: why is loss jagged! shouldnt go up with more clusters?? Unless stuck in local optima?
    # TODO: looks normal with the 2d simple data!
    loss = []
    for k_val in range(1, max_k + 1):
        # loss.append(KMeanspp(k_val).fit(X).loss)
        loss.append(kmeans_obj(k_val).fit(data).inertia_) # for sklearn kmeans
    plt.plot([i for i in range(len(loss))], loss)
    plt.title(title)
    plt.show()
    plt.close()


if __name__ == "__main__":
    path = "X.csv"
    true_labels_path = 'y.csv'
    out_path = "plots"

    X = np.genfromtxt(path, delimiter=',')
    y_true = np.genfromtxt(true_labels_path, delimiter=',')

    # 2d debugging data
    # X, y_true = datasets.make_blobs(10000, n_features=4, random_state=0)
    # centers = KMeanspp(3).center_init(X)
    k = 5

    elbow_plot(max_k=9, data=X, kmeans_obj=KMeansSK, title="sklearn kmeans")
    elbow_plot(max_k=9, data=X, kmeans_obj=KMeanspp, title="kmeans++")

    # calc_labels = KMeans(k=3).fit(X).transform(X)
    # calc_labels = KMeanspp(k=3).fit(X).transform(X)
    # kmeans = KMeanspp(k=k)
    # kmeans = KMeansSK(k)
    kmeans = KMeans(k=k)
    calc_labels = kmeans.fit(X).transform(X)
    # calc_labels = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    # init_centers = kmeans.init_centers.reshape(k, -1)
    # init_centers = pca.transform(init_centers)
    #
    # # Plotting Init points
    # plt.scatter(X[:,0], X[:,1], s=1)
    # plt.scatter(init_centers[:, 0], init_centers[:, 1], alpha=0.4)
    # plt.show()
    # plt.close()

    plots(X, y_true, "Actual_Classes_Major_Axes", out_path)
    plots(X, calc_labels, "Calc_Classes_Major_Axes", out_path)

