from scipy.stats import alpha
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np


def caclSigmoid(val: int | float):
    return 1 / (1 + np.exp(-val))


# def calcBCECostFunct(val: int | float, y):
#     return y * np.log(val) + (1 - y) * np.log(1 - val)


def calcGradient(h, y):
    return h - y


def updateCoef(coef: np.ndarray, alpha: float, deltaJ: np.ndarray) -> np.ndarray:
    return coef - alpha * deltaJ


class CustomLogisticRegression:
    max_iter: int
    x: np.ndarray
    y: np.ndarray
    coef: np.ndarray
    alpha: float
    m: int

    def __init__(self, x: np.ndarray, y: np.ndarray, max_iter=100, alpha=0.1) -> None:
        self.x = np.hstack((np.ones((x.shape[0], 1)), x))  # Add bias term
        self.y = y
        self.max_iter = max_iter
        self.coef = np.zeros(self.x.shape[1])  # includes bias
        self.m = x.shape[0]
        self.alpha = alpha

        # print("X (features with bias):\n", self.x.shape)
        # print("\nY (labels):\n", self.y.shape)
        # print(f"\nMax Iterations: {self.max_iter}")
        # print(f"\nShape of X: {self.x.shape}")
        # print(f"Initialized Coefficients (bias + weights): {self.coef}")

    def fit(self):
        for _ in range(self.max_iter):
            # Step 1 : Hypothesis Function
            z = np.dot(self.x, self.coef)  # shape: (m,)
            h = caclSigmoid(z)
            # print(h)
            # Step 2 : calculate cost Function
            # 📉 Binary Cross Entropy (Log Loss) Cost Function:
            # FORMULA: J(θ) = -1/m * Σ [ yᵢ * log(ŷᵢ) + (1 - yᵢ) * log(1 - ŷᵢ) ]
            # where:
            #   - ŷᵢ = sigmoid(θᵗxᵢ)
            #   - yᵢ is the true label (0 or 1)
            #   - m is the number of training examples
            # costs = self.y * np.log(h) + (1 - self.y) * np.log(1 - h)
            epsilon = 1e-10  # to avoid log(0)
            costs = self.y * np.log(h + epsilon) + (1 - self.y) * np.log(
                1 - h + epsilon
            )
            cost = -np.mean(costs)
            # print(cost)

            # Step 3 : Compute Gradiesnts
            # 🔁 Gradient of the Cost Function (for Logistic Regression):
            # FORMULA: ∇J(θ) = (1/m) * Xᵗ · (sigmoid(X · θ) - y)
            # where:
            #   - X is the (m × n) input matrix with bias column (1s) here i have considered bias as 1
            #   - θ is the (n × 1) parameter vector (including bias)
            #   - y is the (m × 1) true labels vector
            #   - sigmoid(z) = 1 / (1 + e^(-z))
            # vfun2 = np.vectorize(calcGradient)
            # gradient = vfun2(h, y)
            # deltaJ = np.zeros(self.x.shape[1])
            # for i in range(self.m):
            #     sum: float = 0
            #     for idx, val in enumerate(gradient):
            #         sum += val * self.x[idx, i]
            #     sum = (1 / self.m) * sum
            #     deltaJ[i] = sum
            # for all jj, done in one go. ✅✅
            gradient = (1 / self.m) * np.dot(self.x.T, (h - self.y))
            # print(f"Theta {i} : ", sum, gradient, self.x[:, i])

            # Step 4 : Update Coefficients
            # Parameter Update Formula (Gradient Descent):
            # NOTE: θ := θ - α * ∇J(θ)
            #
            # Where:
            # θ       → vector of parameters (including bias)
            # α       → learning rate (step size)
            # ∇J(θ)   → gradient of the cost function w.r.t θ
            #
            # For logistic regression:
            # FORMULA:∇J(θ) = (1/m) * Σ (h(xᵢ) - yᵢ) * xᵢ
            self.coef = updateCoef(self.coef, self.alpha, gradient)

            print(f"Iteration {_}: Cost = {cost:.4f}, Mean h = {np.mean(h):.4f}")
            print(f"Gradient (first 5): {gradient[:5]}")
            print(f"Weights (first 5): {self.coef[:5]}")
        print(self.coef.shape)

    def score(self, x: np.ndarray, y: np.ndarray):
        if x.shape[1] + 1 == self.coef.shape[0]:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
        z = np.dot(x, self.coef)

        h = 1 / (1 + np.exp(-z))

        predictions = (h >= 0.5).astype(int)
        print(predictions)

        accuracy = np.mean(predictions == y)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

    def predict(self, x: np.ndarray | pd.DataFrame):
        pass


if __name__ == "__main__":
    # print("From this file")
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Skleran Model : ", model.score(X_test, y_test))

    # x = pd.DataFrame({"x1": [1, 2, 3], "x2": [2, 1, 3]})
    # y = pd.Series([0, 1, 1])
    # lg = CustomLogisticRegression(x.values, y.values)
    lg = CustomLogisticRegression(X_train, y_train, alpha=0.0005)
    lg.fit()
    print(lg.score(X_test, y_test))
    # y_pred = model.predict(X_test)

    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("\nClassification Report:\n", classification_report(y_test, y_pred))
