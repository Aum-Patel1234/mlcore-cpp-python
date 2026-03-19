from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import MultinomialNB
from typing import Mapping, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


class MultiNomialNaiveBayesClassifier:
    def __init__(self, k_bins=10, strategy="uniform") -> None:
        self.k_bins: int = k_bins
        self.strategy: str = strategy
        # NOTE: {feature: {value_or_bin: {class: probability}}}
        self.probs: Dict[str, Dict[str, Dict[str, float]]] = {}
        # target, prior_prob
        self.target_prob = {}

    def fitObjCol(self, col: pd.Series, y: pd.Series) -> None:
        probs: Dict[str, Dict[str, float]] = {}

        uniques = col.unique()
        k = len(uniques)

        for unique in uniques:
            probs[unique] = {}

        for c in y.unique():
            col_given_class = col[y == c]
            n = len(col_given_class)

            for unique in uniques:
                count = (col_given_class == unique).sum()
                probs[unique][c] = (count + 1) / (n + k)  # laplace smoothing

        self.probs[col.name] = probs

    def fitNumericalCol(self, col: pd.Series, y: pd.Series) -> None:
        col_name = col.name
        # TODO: can apply different stategies here

        probs: Dict[str, Dict[str, float]] = {}

        min, max = col.min(), col.max()
        diff = (max - min) / self.k_bins
        classes = y.unique()

        for i in range(self.k_bins):
            low = min + i * diff
            high = low + diff

            key: str = f"{low}_{high}"
            probs[key] = {}

            for c in classes:
                col_given_class = col[y == c]
                n_class = len(col_given_class)

                # handle last bin
                if i == self.k_bins - 1:
                    mask = (col_given_class >= low) & (col_given_class <= high)
                else:
                    mask = (col_given_class >= low) & (col_given_class < high)

                count = mask.sum()

                # Laplace smoothing
                probs[key][c] = (count + 1) / (n_class + self.k_bins)

        self.probs[col_name] = probs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        n = len(y)

        # prior probs
        for c in y.unique():
            count = (y == c).sum()
            self.target_prob[c] = count / n

        for col in X.columns:
            if pd.api.types.is_object_dtype(X[col]):
                self.fitObjCol(X[col], y)
            else:
                self.fitNumericalCol(X[col], y)

    def _get_bin_key(self, feature, value):
        keys = list(self.probs[feature].keys())

        for key in keys:
            low, high = map(float, key.split("_"))
            if low <= value < high:
                return key

        return keys[-1]

    def predict_row(self, row):
        class_probs = {}

        for c in self.target_prob:
            prob = self.target_prob[c]

            for feature in row.index:
                value = row[feature]

                if feature not in self.probs:
                    continue

                if isinstance(value, str):
                    if value in self.probs[feature]:
                        prob *= self.probs[feature][value].get(c, 1e-9)

                else:
                    bin_key = self._get_bin_key(feature, value)

                    if bin_key:
                        prob *= self.probs[feature][bin_key].get(c, 1e-9)

            class_probs[c] = prob

        return max(class_probs, key=class_probs.get)

    def predict(self, X: pd.DataFrame) -> list:
        predictions = []

        for _, row in X.iterrows():
            predictions.append(self.predict_row(row))

        return predictions

    def score(self, X_test: pd.DataFrame, y_real: pd.Series) -> float:
        y_test = self.predict(X_test)
        # print(y_test, "\n", y_real)
        return accuracy_score(y_test, y_real)


from sklearn.datasets import load_wine


def predict_wine():
    data = load_wine()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # my implementation
    model = MultiNomialNaiveBayesClassifier()
    model.fit(X_train, y_train)

    print("My Multinomial NB Accuracy:", model.score(X_test, y_test))

    # sklearn MultinomialNB
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform")

    X_train_binned = discretizer.fit_transform(X_train)
    X_test_binned = discretizer.transform(X_test)

    sk_model = MultinomialNB()
    sk_model.fit(X_train_binned, y_train)

    sk_predictions = sk_model.predict(X_test_binned)
    sk_acc = accuracy_score(y_test, sk_predictions)

    print("Sklearn MultinomialNB Accuracy:", sk_acc)


def predict_iris():
    data = load_iris()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # my implementation
    model = MultiNomialNaiveBayesClassifier()
    model.fit(X_train, y_train)

    my_predictions = model.predict(X_test)

    print("My Multinomial NB Accuracy:", model.score(X_test, y_test))

    # Sklearn MultinomialNB

    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform")

    X_train_binned = discretizer.fit_transform(X_train)
    X_test_binned = discretizer.transform(X_test)

    sk_model = MultinomialNB()
    sk_model.fit(X_train_binned, y_train)

    sk_predictions = sk_model.predict(X_test_binned)
    sk_acc = accuracy_score(y_test, sk_predictions)

    print("Sklearn MultinomialNB Accuracy:", sk_acc)


def shuffled_label_test():

    data = load_iris()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # shuffle labels
    y = y.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultiNomialNaiveBayesClassifier()
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    print("Accuracy with shuffled labels:", acc)


if __name__ == "__main__":
    predict_iris()
    shuffled_label_test()
    predict_wine()
