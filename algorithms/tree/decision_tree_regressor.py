import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as SkTree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
    ):
        self.feature_index: str | None = feature_index
        self.threshold = threshold
        self.left: Node | None = left
        self.right: Node | None = right
        self.value = value


class DecisionTreeRegressor:
    def __init__(self, max_depth=2, min_observations_leaf=20) -> None:
        self.min_observations_leaf = min_observations_leaf
        self.max_depth = max_depth
        pass

    def fit(self, X: pd.DataFrame, y: str | None) -> None:
        if y is None:
            raise ValueError("Target column name must be provided.")

        target = X[y]
        features = X.drop(columns=[y])

        self.root = self._build_tree(features, target)

    def _traverse_tree(self, x, node: Node):
        if node.value is not None:
            return node.value

        if isinstance(node.threshold, (int, float)):  # numerical
            if x[node.feature_index] < node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:  # categorical
            if x[node.feature_index] == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []

        for _, row in X.iterrows():
            pred = self._traverse_tree(row, self.root)
            preds.append(pred)

        return np.array(preds)

    def mse(self, y: np.ndarray | pd.Series) -> float:
        y = np.asarray(y)
        # FORMULA: MSE = 1/m * sum((y - y_mean)^2)
        return (np.square(y - y.mean())).mean()

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, curr_depth=0) -> Node:
        if curr_depth >= self.max_depth or len(X) < self.min_observations_leaf:
            return Node(value=y.mean())

        best_feature, threshold, min_wieghted_mse = self._best_split(X, y)
        if best_feature == None:
            return Node(value=y.mean())

        if pd.api.types.is_numeric_dtype(X[best_feature]):
            left_mask = X[best_feature] < threshold
            right_mask = X[best_feature] >= threshold
        else:
            left_mask = X[best_feature] == threshold
            right_mask = X[best_feature] != threshold

        left_df, right_df = X[left_mask], X[right_mask]
        y_left, y_right = y[left_mask], y[right_mask]

        left_subtree = self._build_tree(left_df, y_left, curr_depth + 1)
        right_subtree = self._build_tree(right_df, y_right, curr_depth + 1)

        return Node(
            feature_index=best_feature,
            threshold=threshold,
            left=left_subtree,
            right=right_subtree,
            value=None,
        )

    def _best_split(self, X: pd.DataFrame, y: pd.Series):
        min_weighted_mse = float("inf")
        best_feature = None
        best_threshold = None
        n = len(y)

        for col in X.columns:
            if col == y.name:
                continue

            # numerical feature
            if pd.api.types.is_numeric_dtype(X[col]):
                values = np.sort(X[col].unique())

                midpoints = (values[:-1] + values[1:]) / 2

                for midpt in midpoints:
                    left_mask = X[col] < midpt
                    right_mask = X[col] >= midpt

                    y_left = y[left_mask]
                    y_right = y[right_mask]

                    if len(y_left) == 0 or len(y_right) == 0:
                        continue

                    # Variance = MSE
                    # left_mse = y_left.var()
                    left_mse = self.mse(y_left)
                    right_mse = self.mse(y_right)

                    weighted_mse = (
                        len(y_left) / n * left_mse + len(y_right) / n * right_mse
                    )

                    if weighted_mse < min_weighted_mse:
                        best_feature = col
                        min_weighted_mse = weighted_mse
                        best_threshold = midpt

            # Categorical feature
            else:
                categories = X[col].unique()

                for category in categories:
                    left_mask = X[col] == category
                    right_mask = X[col] != category

                    y_left = y[left_mask]
                    y_right = y[right_mask]

                    if (
                        len(y_left) < self.min_observations_leaf
                        or len(y_right) < self.min_observations_leaf
                    ):
                        continue

                    left_mse = self.mse(y_left)
                    right_mse = self.mse(y_right)

                    weighted_mse = (
                        len(y_left) / n * left_mse + len(y_right) / n * right_mse
                    )

                    if weighted_mse < min_weighted_mse:
                        best_feature = col
                        best_threshold = category
                        min_weighted_mse = weighted_mse

        return best_feature, best_threshold, min_weighted_mse


def diabetes_pred():
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    # custom
    dtr = DecisionTreeRegressor(max_depth=4, min_observations_leaf=5)
    dtr.fit(train_df, "target")

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    your_preds = dtr.predict(X_test)
    your_rmse = np.sqrt(np.mean((your_preds - y_test) ** 2))

    # sklearn
    sk_model = SkTree(max_depth=4, min_samples_leaf=5, random_state=42)
    sk_model.fit(train_df.drop(columns=["target"]), train_df["target"])

    sk_preds = sk_model.predict(X_test)

    sk_rmse = np.sqrt(np.mean((sk_preds - y_test) ** 2))

    print("Your RMSE: ", your_rmse)
    print("Sklearn RMSE:", sk_rmse)


def housing_pred():
    data = fetch_california_housing()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    dtr = DecisionTreeRegressor(max_depth=5, min_observations_leaf=10)
    dtr.fit(train_df, "target")

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    your_preds = dtr.predict(X_test)
    your_rmse = np.sqrt(np.mean((your_preds - y_test) ** 2))

    sk_model = SkTree(max_depth=5, min_samples_leaf=10, random_state=42)
    sk_model.fit(train_df.drop(columns=["target"]), train_df["target"])
    sk_preds = sk_model.predict(X_test)
    sk_rmse = np.sqrt(np.mean((sk_preds - y_test) ** 2))

    print("Your RMSE: ", your_rmse)
    print("Sklearn RMSE:", sk_rmse)


if __name__ == "__main__":
    diabetes_pred()
    print()
    housing_pred()
