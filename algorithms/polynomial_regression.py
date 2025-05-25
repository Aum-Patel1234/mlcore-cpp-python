from textwrap import indent
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np


class CustomPolynomialFeatures:
    # FORMULA: the total number of combinations in Polynomial Feature with degree=d and n parameters
    # no of combinations = (n+d)!/d!n!
    degree: int
    include_bias: bool

    def __init__(self, degree=2, include_bias=True) -> None:
        self.degree = degree
        self.include_bias = include_bias

    def fact(self, num) -> int:
        if num == 1 or num == 0:
            return 1
        return num * self.fact(num - 1)

    def fit_transform(self, x: np.ndarray) -> np.ndarray | None:
        counter = 0
        if x.ndim != 2:
            print("Ndimensions not equal to 2")
            return

        n, n_features = x.shape
        arr = np.ones((n, 1)) if self.include_bias else np.empty((n, 0))
        arr = np.hstack((arr, x))
        print(
            "Rows - ",
            n,
            "\tColumns - ",
            n_features,
            "Shape after bias and deg 1 - ",
            arr.shape,
        )

        for deg in range(2, self.degree + 1):  # iterating over all degrees
            indices = [[i] for i in range(n_features)]
            # print(indices)
            for _ in range(deg - 1):
                newIndices = []
                for comb in indices:
                    for i in range(comb[-1], n_features):
                        newIndices.append(comb + [i])
                        # print(comb, comb + [i])
                indices = newIndices
                # print(indices)
            for comb in indices:
                new_feature = np.ones(n)
                for idx in comb:
                    new_feature *= x[:, idx]
                arr = np.hstack((arr, new_feature.reshape(-1, 1)))

        print(
            x.shape,
            "\tArr shape : ",
            arr.shape,
            "\nTotal shape should be : ",
            (
                self.fact(n_features + self.degree)
                / (self.fact(n_features) * self.fact(self.degree))
            ),
        )
        return arr


if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    #
    # y = np.array([0, 1, 0, 1, 0])

    # Apply custom polynomial features
    poly = CustomPolynomialFeatures(degree=3, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(y_train)

    # # Fit logistic regression
    # clf = LogisticRegression(max_iter=10000)
    # clf.fit(X_train_poly, y_train)
    #
    # # Predict and evaluate
    # y_pred = clf.predict(X_test_poly)
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("\nClassification Report:\n", classification_report(y_test, y_pred))
