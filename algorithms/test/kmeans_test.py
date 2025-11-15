import numpy as np
import sys, os
from sklearn.datasets import make_blobs
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath("../../cpp/build"))

import mlcore_cpp

X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.2)

print(type(X), type(y))

# prediction = mlcore_cpp.kmeans()
kmeans = KMeans(4)
kmeans.fit(X)

predicion = 0
kmeans_label = kmeans.labels_

cm = confusion_matrix(y, prediction)

ri, ci = linear_sum_assignment(-cm)

mapping = {col: row for row, col in zip(ri, ci)}

map_preds = np.array([mapping[label] for lable in prediction])

print("Accuracy of mlcorecpp - ", accuracy_score(y, map_preds))
print("Accuracy of mlcorecpp - ", accuracy_score(y, kmeans_label))
