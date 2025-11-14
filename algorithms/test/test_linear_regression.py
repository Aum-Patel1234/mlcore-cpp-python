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


def test_regularization_all():
    print("\n=== Testing Regularization Model (All Types) ===")

    # dataset: y = 2x
    X = np.array([[1], [2], [3], [4], [5]], dtype=np.double)
    y = np.array([2, 4, 6, 8, 10], dtype=np.double)

    ###########################################################################
    # L2 RIDGE
    ###########################################################################
    print("\n--- L2 Regularization (Ridge) ---")
    model_l2 = mlcore_cpp.Regularization(
        X,
        y,
        iterations=1000,
        type=mlcore_cpp.RegType.L2,
        learning_rate=0.01,
        lambda1=0.0,
        lambda2=0.1,
    )

    model_l2.fit()
    model_l2.printSlopeIntercept()

    preds_l2 = model_l2.predict(np.array([[6], [7], [8]], dtype=np.double))
    print("Predictions L2:", preds_l2)

    from sklearn.linear_model import Ridge

    sk_l2 = Ridge(alpha=0.1)
    sk_l2.fit(X, y)
    print("Sklearn Ridge Coef:", sk_l2.coef_, "Intercept:", sk_l2.intercept_)

    ###########################################################################
    # L1 LASSO
    ###########################################################################
    print("\n--- L1 Regularization (Lasso) ---")
    model_l1 = mlcore_cpp.Regularization(
        X,
        y,
        iterations=1000,
        type=mlcore_cpp.RegType.L1,
        learning_rate=0.01,
        lambda1=0.1,  # Only lambda1 matters for L1
        lambda2=0.0,
    )

    model_l1.fit()
    model_l1.printSlopeIntercept()

    preds_l1 = model_l1.predict(np.array([[6], [7], [8]], dtype=np.double))
    print("Predictions L1:", preds_l1)

    from sklearn.linear_model import Lasso

    sk_l1 = Lasso(alpha=0.1)
    sk_l1.fit(X, y)
    print("Sklearn Lasso Coef:", sk_l1.coef_, "Intercept:", sk_l1.intercept_)

    ###########################################################################
    # ELASTIC NET
    ###########################################################################
    print("\n--- Elastic Net ---")
    model_en = mlcore_cpp.Regularization(
        X,
        y,
        iterations=1000,
        type=mlcore_cpp.RegType.Elastic,
        learning_rate=0.01,
        lambda1=0.1,
        lambda2=0.1,
    )

    model_en.fit()
    model_en.printSlopeIntercept()

    preds_en = model_en.predict(np.array([[6], [7], [8]], dtype=np.double))
    print("Predictions Elastic:", preds_en)

    from sklearn.linear_model import ElasticNet

    sk_en = ElasticNet(alpha=0.1, l1_ratio=0.5)  # λ1=λ2 mixture
    sk_en.fit(X, y)
    print("Sklearn Elastic Coef:", sk_en.coef_, "Intercept:", sk_en.intercept_)


if __name__ == "__main__":
    print(__name__)
    test_linear_regression()
    test_regularization_all()
