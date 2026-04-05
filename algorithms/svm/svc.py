import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC as SklearnSVC

# NOTE: Main Steps of SVM
# 1. start with data in relatively low dimension
# 2. move data into higher dimension
# 3. find Support Vector Classifier which seperates data into 2 groups
# 4. to make this mathematically possible, SVM uses Kernel Trick
#
# IMPORTANT: Kernel Trick reduces the amount of computation required for SVM
#            by avoiding math that transforms data from low to high dimension
# Trick:
# 1. let say d=2(degree) and kernel=polynomial kernel
# 2. the kernel computes 2D relationships betweeen each pair of observations
# 3. Those relationships are used to find the Support Vector Classifier
# 4. can find d using cross validation


class SVC:
    def __init__(self, kernel="polynomial", gamma=1.0, coef=0.0, degree=3) -> None:
        self.kernel = kernel
        self.coef = coef
        self.gamma = gamma
        self.degree = degree
        self.alpha = None

    # NOTE: svm library -> https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cdbd75ae4e662fda67dcb19277258792b/sklearn/svm/src/libsvm/svm.cpp
    # line 342
    # formula  : (gamma * (x.T*x') + coef) ^ degree
    # Intution : (gamma * similariy score  + bias)^d
    # where:
    #   gamma - scaling param adjust inluence of similariy score
    #   x  - a point
    #   x' - another point with which its relation is being discovered
    #   coef - bias
    #   d  - degree
    #
    # IMPORTANT: Simplified: (a*b + r)^d
    # TODO: learn why dot product calcuates or Intution of dot product for similariy score
    def polynomial_kernel(
        self, x: pd.Series, x_dash: pd.Series, gamma: float, coef: float, degree: int
    ):
        return np.power(gamma * np.dot(x.T, x_dash) + coef, degree)

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, iter=100
    ) -> None:
        m, _ = X.shape
        X = X.values if hasattr(X, "values") else X
        y = y.values if hasattr(y, "values") else y

        # 1. Initialize weights
        self.alpha = np.zeros(m)
        self.b = 0.0
        y = np.where(y <= 0, -1, 1)
        tol = 1e-3
        C = 1.0

        # 2. compute kernel matrix
        kernel_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                # TODO: apply different kernels here
                kernel_matrix[i, j] = self.polynomial_kernel(
                    X[i], X[j], self.gamma, self.coef, self.degree
                )

        # 3. training SMO(sequential minimal optimization)
        for _ in range(iter):
            for i in range(m):
                # 3.1 compute prediction
                f_i = np.sum(self.alpha * y * kernel_matrix[:, i]) + self.b
                E_i = f_i - y[i]  # error

                # Step 3.2: Check if alpha[i] needs update
                if (y[i] * E_i < -tol and self.alpha[i] < C) or (
                    y[i] * E_i > tol and self.alpha[i] > 0
                ):
                    # pick random j != i
                    j = np.random.randint(0, m)
                    while j == i:
                        j = np.random.randint(0, m)

                    # repeat 3.1
                    f_j = np.sum(self.alpha * y * kernel_matrix[:, j]) + self.b
                    E_j = f_j - y[j]  # error

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    # Step 3.4: Compute L and H (constraints)
                    if y[i] != y[j]:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(C, C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_i_old + alpha_j_old - C)
                        H = min(C, alpha_i_old + alpha_j_old)

                    if L == H:
                        continue

                    # Step 3.5: Compute eta
                    eta = (
                        2 * kernel_matrix[i, j]
                        - kernel_matrix[i, i]
                        - kernel_matrix[j, j]
                    )
                    if eta >= 0:
                        continue

                    # Step 3.6: Update alpha[j]
                    self.alpha[j] -= (y[j] * (E_i - E_j)) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Step 3.7: Update alpha[i]
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # Step 3.8: Update bias b
                    b1 = (
                        self.b
                        - E_i
                        - y[i] * (self.alpha[i] - alpha_i_old) * kernel_matrix[i, i]
                        - y[j] * (self.alpha[j] - alpha_j_old) * kernel_matrix[i, j]
                    )

                    b2 = (
                        self.b
                        - E_j
                        - y[i] * (self.alpha[i] - alpha_i_old) * kernel_matrix[i, j]
                        - y[j] * (self.alpha[j] - alpha_j_old) * kernel_matrix[j, j]
                    )

                    if 0 < self.alpha[i] < C:
                        self.b = b1
                    elif 0 < self.alpha[j] < C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

        # Save training data
        self.X = X
        self.y = y

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.alpha is None:
            return pd.Series()
        X = X.values if hasattr(X, "values") else X
        predictions = []
        m = X.shape[0]

        for x in X:
            s = 0.0
            for i in range(len(self.alpha)):
                if self.alpha[i] > 1e-5:
                    s += (
                        self.alpha[i]
                        * self.y[i]
                        * self.polynomial_kernel(
                            self.X[i], x, self.gamma, self.coef, self.degree
                        )
                    )

            s += self.b
            predictions.append(np.sign(s))

        return pd.Series(predictions)

    def score(self, y_pred: pd.Series, y_real: pd.Series) -> float:
        return accuracy_score(y_pred, y_real)


def plot_decision_boundary(model, X, y):
    import matplotlib.pyplot as plt

    X = X.values if hasattr(X, "values") else X
    y = y.values if hasattr(y, "values") else y

    # Create grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Flatten grid for prediction
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict on grid
    Z = model.predict(grid)
    Z = Z.values.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)

    # Plot actual points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")

    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.show()


def predict_make_blobs():
    X, y = make_blobs(n_samples=200, centers=2, random_state=42)
    y = np.where(y == 0, -1, 1)

    X = pd.DataFrame(X)
    y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # my SVM
    model = SVC(kernel="polynomial", gamma=1.0, coef=1.0, degree=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Your SVM Accuracy: {acc:.4f}")

    # sklearn SVM
    sk_model = SklearnSVC(kernel="poly", gamma=1.0, coef0=1.0, degree=2)
    sk_model.fit(X_train, y_train)

    sk_pred = sk_model.predict(X_test)

    sk_acc = accuracy_score(y_test, sk_pred)
    print(f"Sklearn SVM Accuracy: {sk_acc:.4f}")
    plot_decision_boundary(model, X_train, y_train)


def predict_iris():
    from sklearn.datasets import load_iris

    data = load_iris()
    # X = data.data[:, 2:4]  # petal length & width (better separation)
    X = data.data
    y = data.target

    y = np.where(y == 0, -1, 1)

    X = pd.DataFrame(X)
    y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVC(kernel="polynomial", gamma=1.0, coef=1.0, degree=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Your SVM Accuracy: {acc:.4f}")

    # Sklearn SVM
    sk_model = SklearnSVC(kernel="poly", gamma=1.0, coef0=1.0, degree=2)
    sk_model.fit(X_train, y_train)

    sk_pred = sk_model.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_pred)
    print(f"Sklearn SVM Accuracy: {sk_acc:.4f}")

    # plot_decision_boundary(model, X_train, y_train)


if __name__ == "__main__":
    predict_iris()
    # predict_make_blobs()
