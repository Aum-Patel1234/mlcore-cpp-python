import numpy as np
import sys, os

sys.path.append(os.path.abspath("../../cpp/build"))

import mlcore_cpp


def test_linear_regression():
    # Create dummy linear data: y = 3x + 2
    X = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
    y = np.array([[5], [8], [11], [14], [17]], dtype=np.float64)

    # Combine X and y horizontally to pass to constructor
    data = np.hstack((X, y))

    # Instantiate LinearRegression
    model = mlcore_cpp.LinearRegression(data, 1000, 0.01)

    # Fit the model
    model.fit()

    # Print learned slope and intercept
    model.printSlopeIntercept()

    # Predict for new values
    X_test = np.array([[6], [7], [8]], dtype=np.float64)
    predictions = model.predict(X_test)

    print("Predictions for [[6], [7], [8]]:", predictions)


if __name__ == "__main__":
    print(__name__)
    test_linear_regression()
