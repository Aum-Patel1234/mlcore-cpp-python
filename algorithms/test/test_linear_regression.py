import numpy as np
import sys, os

sys.path.append(os.path.abspath("../../cpp/build"))

import mlcore_cpp
from sklearn.linear_model import LinearRegression as SklearnLinearRegression


def test_linear_regression():
    # Create dummy linear data: y = 3x + 2
    X = np.array([[1], [2], [3], [4], [5]], dtype=np.double)
    # y = np.array([[5], [8], [11], [14], [17]], dtype=np.float64)
    # X = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y = np.array([2, 4, 6, 8, 10], dtype=np.double)

    # Instantiate LinearRegression
    model = mlcore_cpp.LinearRegression(X, y, 2000)
    print("initialized")

    model.printSlopeIntercept()
    # Fit the model
    model.fit()

    # Print learned slope and intercept
    model.printSlopeIntercept()
    print("\n")
    model.normalEquationFit()
    print("\n")
    model_sklearn = SklearnLinearRegression()
    model_sklearn.fit(X, y)
    print(f"Scikit-learn Slopes - {model_sklearn.coef_}")
    print(f"Scikit-learn Intercept - {model_sklearn.intercept_}")

    # Predict for new values
    X_test = np.array([[6], [7], [8]], dtype=np.float64)
    predictions = model.predict(X_test)

    print("Predictions for [[6], [7], [8]]:", predictions)


if __name__ == "__main__":
    print(__name__)
    test_linear_regression()
