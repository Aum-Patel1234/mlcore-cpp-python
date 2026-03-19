from enum import unique
from math import pi, sqrt
from re import sub
from sklearn.naive_bayes import GaussianNB
from typing import Mapping, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_breast_cancer, load_wine


# NOTE:
# 1. Split data by class
# 2. Compute mean per feature
# 3. Compute variance per feature
# 4. Calculate class prior probabilities
# 5. Compute Gaussian likelihoods for features
#
# Predictions:
# Multiply likelihoods and prior
# Compare classes, choose highest
class GaussianNaiveBayesClassifier:
    def __init__(self) -> None:
        self.prior_probs = {}
        # likelihoods -> {feature:{class: {mean: , standard_deviation: }}}
        self.likelihoods = {}

    def get_standard_deviation(self, col: pd.Series) -> float:
        mean = col.mean()
        n = len(col)
        return sqrt(((col - mean) ** 2).sum() / n)

    def gaussianPDF(self, val: float, standard_deviation: float, mean: float):
        eps = 1e-9  # avoid division by zero
        standard_deviation = max(standard_deviation, eps)

        den = standard_deviation * sqrt(2 * pi)
        num = np.exp(-((val - mean) ** 2) / (2 * (standard_deviation**2)))
        return num / den

    # normal categorical probability
    def fitObjCol(self, col: pd.Series, y: pd.Series) -> None:
        self.likelihoods[col.name] = {}

        unique = col.unique()
        k = len(unique)

        for cls in y.unique():
            subset = col[y == cls]
            n = len(subset)
            counts = subset.value_counts()

            probs = {}
            for c in unique:
                count = counts.get(c, 0)
                probs[c] = (count + 1) / (n + k)
            self.likelihoods[col.name][cls] = probs

    def fitNumericalCol(self, col: pd.Series, y: pd.Series) -> None:
        self.likelihoods[col.name] = {}

        for cls in y.unique():
            subset = col[y == cls]
            mean = subset.mean()
            standard_deviation = self.get_standard_deviation(subset)

            self.likelihoods[col.name][cls] = {
                "mean": mean,
                "standard_deviation": standard_deviation,
            }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        n = len(y)

        for unique in y.unique():
            count = (unique == y).sum()
            self.prior_probs[unique] = count / n

        for col in X.columns:
            if pd.api.types.is_object_dtype(X[col]):
                self.fitObjCol(X[col], y)
            else:
                self.fitNumericalCol(X[col], y)

    # NOTE:
    # score = log(prior prob) + [log(prob | features)]
    # max score is the class which it belongs to
    def predictRow(self, row):
        classes = list(self.prior_probs.keys())
        max_score: float = -float("inf")
        pred_cls = classes[-1]

        for cls in classes:
            prior_prob = self.prior_probs.get(cls, 1e-9)
            score: float = np.log(prior_prob)

            for feature, val in row.items():

                if (
                    isinstance(self.likelihoods[feature][cls], dict)
                    and "mean" not in self.likelihoods[feature][cls]
                ):  # obj feature
                    prob = self.likelihoods[feature][cls].get(val, 1e-9)
                    score += np.log(prob)
                else:
                    params = self.likelihoods[feature][cls]
                    mean = params["mean"]
                    standard_deviation = params["standard_deviation"]
                    score += np.log(self.gaussianPDF(val, standard_deviation, mean))

            if score > max_score:
                max_score = score
                pred_cls = cls

        return pred_cls

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predicitons = []

        for _, row in X.iterrows():
            predicitons.append(self.predictRow(row))

        return pd.Series(predicitons)

    def score(self, X_test: pd.DataFrame, y_real: pd.Series) -> float:
        return accuracy_score(self.predict(X_test), y_real)


def predict_iris():
    iris = load_iris()

    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # my Model
    my_model = GaussianNaiveBayesClassifier()
    my_model.fit(X_train, y_train)
    y_pred_my = my_model.predict(X_test)

    # Sklearn Model
    sk_model = GaussianNB()
    sk_model.fit(X_train, y_train)

    y_pred_sk = sk_model.predict(X_test)

    print("My Model Accuracy:", accuracy_score(y_test, y_pred_my))
    print("Sklearn Accuracy:", accuracy_score(y_test, y_pred_sk))


def predict_wine():
    wine = load_wine()

    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    my_model = GaussianNaiveBayesClassifier()
    my_model.fit(X_train, y_train)
    y_pred_my = my_model.predict(X_test)

    sk_model = GaussianNB()
    sk_model.fit(X_train, y_train)
    y_pred_sk = sk_model.predict(X_test)

    print("My Model Accuracy:", accuracy_score(y_test, y_pred_my))
    print("Sklearn Accuracy:", accuracy_score(y_test, y_pred_sk))


def predict_breast_cancer():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    my_model = GaussianNaiveBayesClassifier()
    my_model.fit(X_train, y_train)
    y_pred_my = my_model.predict(X_test)

    sk_model = GaussianNB()
    sk_model.fit(X_train, y_train)
    y_pred_sk = sk_model.predict(X_test)

    print("My Model Accuracy:", accuracy_score(y_test, y_pred_my))
    print("Sklearn Accuracy:", accuracy_score(y_test, y_pred_sk))


if __name__ == "__main__":
    predict_iris()
    predict_wine()
    predict_breast_cancer()
