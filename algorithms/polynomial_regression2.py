from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


import numpy as np


class CustomPolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def _generate_exponents(self, n_features, degree):
        """
        Generates all combinations of feature exponents for a given degree.
        For example: for degree=2, n_features=2 → [(2,0), (1,1), (0,2)]
        """

        def helper(n, d, prefix, result):
            if n == 1:
                result.append(prefix + [d])
                return
            for i in range(d + 1):
                helper(n - 1, d - i, prefix + [i], result)

        result = []
        for total_deg in range(0 if self.include_bias else 1, degree + 1):
            helper(n_features, total_deg, [], result)
        return result

    def fit_transform(self, x: np.ndarray):
        if x.ndim != 2:
            raise ValueError("Input must be a 2D array")

        n_samples, n_features = x.shape
        exponents_list = self._generate_exponents(n_features, self.degree)

        transformed = []
        for exponents in exponents_list:
            term = np.ones(n_samples)
            for i in range(n_features):
                term *= x[:, i] ** exponents[i]
            transformed.append(term.reshape(-1, 1))

        result = np.hstack(transformed)
        print(
            f"(Samples: {n_samples}, Original Features: {n_features}) → Transformed shape: {result.shape}"
        )
        return result


if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    poly = CustomPolynomialFeatures(degree=3, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
