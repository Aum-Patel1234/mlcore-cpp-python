import numpy as np
import sys, os
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath("../../cpp/build"))
import mlcore_cpp

X_toy = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y_toy = np.array([0, 1, 1, 1], dtype=np.float64).reshape(-1, 1)

log_reg = mlcore_cpp.LogisticRegression(alpha=0.01)
iter_toy = log_reg.fit(X_toy, y_toy, iterations=230)

print("\n[C++ Logistic Regression Results - Toy Data]")
log_reg.printSlopeIntercept()
preds_cpp_toy = log_reg.predict(X_toy)
print("Predictions (C++):", np.round(preds_cpp_toy.reshape(1, 4), 3))

sk_model_toy = SkLogReg(fit_intercept=True, solver="lbfgs")
sk_model_toy.fit(X_toy, y_toy.ravel())
print("\n[Sklearn Logistic Regression Results - Toy Data]")
print("Weights (theta):", np.round(sk_model_toy.coef_, 3))
print("Bias (intercept):", np.round(sk_model_toy.intercept_, 3))
print("Number of iterations run:", sk_model_toy.n_iter_)
preds_sklearn_toy = sk_model_toy.predict_proba(X_toy)[:, 1]
print("Predictions (Sklearn):", np.round(preds_sklearn_toy, 3))


# ---- Real dataset: Breast Cancer---
data = load_breast_cancer()
X_real = data.data
y_real = data.target.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X_real_scaled = scaler.fit_transform(X_real)

X_train, X_test, y_train, y_test = train_test_split(
    X_real_scaled, y_real, test_size=0.3, random_state=42
)

log_reg_real = mlcore_cpp.LogisticRegression(alpha=0.01)
iter_real = log_reg_real.fit(X_train, y_train, iterations=500)

print("\n[C++ Logistic Regression Results - Breast Cancer]")
# log_reg_real.printSlopeIntercept()
preds_cpp_real = log_reg_real.predict(X_test)
preds_cpp_real_class = (preds_cpp_real > 0.5).astype(int)
accuracy_cpp = np.mean(preds_cpp_real_class == y_test)
print("C++ Test Accuracy:", np.round(accuracy_cpp, 3))


# ---- Sklearn Logistic Regression ----
sk_model_real = SkLogReg(fit_intercept=True, solver="lbfgs", max_iter=500)
sk_model_real.fit(X_train, y_train.ravel())
preds_sklearn_real = sk_model_real.predict(X_test)
accuracy_sklearn = np.mean(preds_sklearn_real == y_test.ravel())

print("\n[Sklearn Logistic Regression Results - Breast Cancer]")
print("Weights (theta):", np.round(sk_model_real.coef_, 3))
print("Bias (intercept):", np.round(sk_model_real.intercept_, 3))
print("Number of iterations run:", sk_model_real.n_iter_)
print("Sklearn Test Accuracy:", np.round(accuracy_sklearn, 3))
