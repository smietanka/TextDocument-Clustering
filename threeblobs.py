import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles

plt.figure(1)
X1, Y1 = make_blobs(n_features=4, centers=3)

plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=35, edgecolor='k')

plt.show()