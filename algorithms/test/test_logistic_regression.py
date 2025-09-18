import numpy as np
import sys, os

sys.path.append(os.path.abspath("../../cpp/build"))
import mlcore_cpp
from sklearn.linear_model import LogisticRegression as SkLogReg

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([0, 1, 1, 1], dtype=np.float64).reshape(-1, 1)

# ---- Your C++ Logistic Regression ----
log_reg = mlcore_cpp.LogisticRegression(alpha=0.01)
iter = log_reg.fit(X, y, iterations=230)

print("\n[C++ Logistic Regression Results]")
log_reg.printSlopeIntercept()

preds_cpp = log_reg.predict(X)
print(
    "Predictions (C++):", np.round(preds_cpp.reshape(1, 4), 3)
)  # rounded to 3 decimals


# ---- Sklearn Logistic Regression ----
sk_model = SkLogReg(fit_intercept=True, solver="lbfgs")
sk_model.fit(X, y.ravel())

print("\n[Sklearn Logistic Regression Results]")
print("Weights (theta):", np.round(sk_model.coef_, 3))
print("Bias (intercept):", np.round(sk_model.intercept_, 3))
print("Number of iterations run:", sk_model.n_iter_)

preds_sklearn = sk_model.predict_proba(X)[:, 1]
print("Predictions (Sklearn):", np.round(preds_sklearn, 3))
