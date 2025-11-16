from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.cluster import KMeans
import sys
import os


# Load C++ module
sys.path.append(os.path.abspath("../../cpp/build"))
import mlcore_cpp


def run_single_test(X, y, centers, name):
    print(f"\n=== Testing dataset: {name} ===")

    # sklearn KMeans
    sk = KMeans(n_clusters=centers, random_state=42)
    sk.fit(X)
    sk_labels = sk.labels_

    # C++ KMeans
    cpp = mlcore_cpp.Kmeans(centers, 100)
    cpp.fit(X)
    cpp_labels = cpp.predict(X)

    # Align labels (Hungarian algorithm)
    cm = confusion_matrix(y, cpp_labels)
    row_idx, col_idx = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_idx, col_idx)}
    cpp_labels_mapped = np.array([mapping[l] for l in cpp_labels])

    # Accuracy
    cpp_acc = accuracy_score(y, cpp_labels_mapped)
    sk_acc = accuracy_score(y, sk_labels)

    print(f"Accuracy (C++ KMeans): {cpp_acc:.4f}")
    print(f"Accuracy (sklearn):   {sk_acc:.4f}")

    return cpp_acc, sk_acc


def test_kmeans_all():
    results = {}

    # 1) Normal blobs (your original dataset)
    X1, y1 = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.2)
    results["blobs"] = run_single_test(X1, y1, centers=4, name="Blobs")

    # 2) Moons dataset
    X2, y2 = make_moons(n_samples=300, noise=0.08, random_state=42)
    X2 = StandardScaler().fit_transform(X2)
    results["moons"] = run_single_test(X2, y2, centers=2, name="Moons")

    # 3) Circles dataset
    X3, y3 = make_circles(n_samples=300, noise=0.07, factor=0.5, random_state=42)
    X3 = StandardScaler().fit_transform(X3)
    results["circles"] = run_single_test(X3, y3, centers=2, name="Circles")

    # 4) Anisotropic blobs
    X4, y4 = make_blobs(n_samples=300, centers=4, random_state=42)
    transformation = [[0.6, -0.6], [-0.6, 0.6]]
    X4 = np.dot(X4, transformation)
    results["anisotropic"] = run_single_test(
        X4, y4, centers=4, name="Anisotropic blobs"
    )

    print("\n===== ALL TEST RESULTS =====")
    for name, (cpp_acc, sk_acc) in results.items():
        print(f"{name}: C++={cpp_acc:.4f}, sklearn={sk_acc:.4f}")

    return results


if __name__ == "__main__":
    test_kmeans_all()
