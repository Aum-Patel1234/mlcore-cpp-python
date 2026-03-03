import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree
from sklearn.metrics import accuracy_score


class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        info_gain=None,
        value=None,
    ):
        self.feature_index: str | None = feature_index
        self.threshold = threshold
        self.left: Node | None = left
        self.right: Node | None = right
        self.info_gain: float | None = info_gain
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, min_sample_split=2, max_depth=2):
        self.root = None
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

    def _majority_class(self, y: pd.Series):
        return y.value_counts().idxmax()

    def _split_dataset(self, df: pd.DataFrame, feature: str, value):
        left = df[df[feature] == value]
        right = df[df[feature] != value]

        return left, right

    def build_tree(
        self,
        df: pd.DataFrame,
        target_feature: pd.Series,
        curr_depth=0,
        metric: str = "gini",
    ) -> Node | None:
        if (
            curr_depth >= self.max_depth
            or len(df) < self.min_sample_split
            or target_feature.nunique() == 1
        ):
            leaf_value = self._majority_class(target_feature)
            return Node(value=leaf_value)

        best_feature, threshold, best_gain = self._best_split(
            df, target_feature, metric
        )

        if best_feature == None:
            leaf_value = self._majority_class(target_feature)
            return Node(value=leaf_value)

        # numerical col
        if threshold is not None:
            left_df = df[df[best_feature] < threshold]
            right_df = df[df[best_feature] >= threshold]

        # categoical col
        else:
            split_val = df[best_feature].mode()[0]
            left_df = df[df[best_feature] == split_val]
            right_df = df[df[best_feature] != split_val]
            threshold = split_val

        if len(left_df) == 0 or len(right_df) == 0:
            leaf_value = self._majority_class(target_feature)
            return Node(value=leaf_value)

        left_subtree = self.build_tree(
            left_df,
            target_feature=left_df[target_feature.name],
            curr_depth=curr_depth + 1,
            metric=metric,
        )
        right_subtree = self.build_tree(
            right_df,
            target_feature=right_df[target_feature.name],
            curr_depth=curr_depth + 1,
            metric=metric,
        )

        return Node(
            feature_index=best_feature,
            threshold=threshold,
            left=left_subtree,
            right=right_subtree,
            info_gain=best_gain,
        )

    def fit(self, X: pd.DataFrame, target_col: str | pd.Series, metric="gini") -> None:
        target = X[target_col] if isinstance(target_col, str) else target_col
        self.root = self.build_tree(X, target_feature=target, metric=metric)

    def _predict_row(self, row: pd.Series, node: Node):
        if node.value is not None:
            return node.value

        feature_val = row[node.feature_index]

        if isinstance(node.threshold, (int, float)):
            if feature_val < node.threshold:
                return self._predict_row(row, node.left)
            else:
                return self._predict_row(row, node.right)
        else:
            if feature_val == node.threshold:
                return self._predict_row(row, node.left)
            else:
                return self._predict_row(row, node.right)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = []

        for _, row in X.iterrows():
            preds.append(self._predict_row(row, self.root))

        return pd.Series(preds, index=X.index)

    def score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        return accuracy_score(self.predict(X), y)

    def _best_split(self, data: pd.DataFrame, target_feature: pd.Series, metric="ig"):
        """ "
        finds best feature split based on Information Gain by default
            ig = Information Gain
            gini = gini impurity
        """
        best_feature = None
        best_threshold = None
        best_metric = -float("inf")

        for col in data.columns:
            if col == target_feature.name:
                continue

            # Numeric features
            if pd.api.types.is_numeric_dtype(data[col]):
                values = np.sort(data[col].unique())

                midpoints = []
                for i in range(1, len(values)):
                    midpoints.append((values[i] + values[i - 1]) / 2)

                for midpoint in midpoints:
                    left = target_feature[data[col] < midpoint]
                    right = target_feature[data[col] >= midpoint]

                    if metric == "ig":
                        h_parent = self._getEntropy(target_feature)

                        w_left = len(left) / len(target_feature)
                        w_right = len(right) / len(target_feature)

                        feature_metric = (
                            h_parent
                            - w_left * self._getEntropy(left)
                            - w_right * self._getEntropy(right)
                        )
                    elif metric == "gini":
                        g_parent = self._getGiniIndex(target_feature)

                        w_left = len(left) / len(target_feature)
                        w_right = len(right) / len(target_feature)

                        feature_metric = (
                            g_parent
                            - w_left * self._getGiniIndex(left)
                            - w_right * self._getGiniIndex(right)
                        )

                    else:
                        raise ValueError("metric must be 'ig' or 'gini'")

                    if feature_metric > best_metric:
                        best_metric = feature_metric
                        best_feature = col
                        best_threshold = midpoint

            # Categorical features
            else:
                if metric == "ig":
                    feature_metric = self._getInformationGain(data[col], target_feature)

                elif metric == "gini":
                    parent_gini = self._getGiniIndex(target_feature)
                    weighted_gini = 0.0

                    for val in data[col].unique():
                        subset = target_feature[data[col] == val]
                        weight = subset.size / target_feature.size
                        weighted_gini += weight * self._getGiniIndex(subset)

                    feature_metric = parent_gini - weighted_gini
                else:
                    raise ValueError("metric must be 'ig' or 'gini'")

                if feature_metric > best_metric:
                    best_metric = feature_metric
                    best_feature = col
                    best_threshold = None
            # print(col, " - ", feature_metric)

        return best_feature, best_threshold, best_metric

    def _getEntropy(self, feature: pd.Series):
        # IMPORTANT: Entropy is surprise * probablity(surprise)
        # e = p(x) * log2(1/p(x))
        # e = p(x) * [log2(1) - log2(p(x))]
        # e = - p(x)*log2(p(x))
        # Entropy formula: Entropy = -Σ (p_i * log2(p_i))
        entropy = 0.0

        for val in feature.unique():
            pi = (feature == val).sum() / feature.size
            entropy += pi * np.log2(pi)

        assert -entropy >= -1e-9
        assert -entropy <= np.log2(feature.nunique()) + 1e-9
        return -entropy

    def _getInformationGain(
        self, feature: pd.Series, target_feature: pd.Series
    ) -> float:
        # Information Gain formula: IG = Entropy(Dataset) - Weighted_Entropy(Children)
        h_target = self._getEntropy(target_feature)
        weighted_entropy = 0.0

        for val in feature.unique():
            sv = target_feature[feature == val]
            pi = sv.size / target_feature.size
            weighted_entropy += pi * self._getEntropy(sv)

        ig = h_target - weighted_entropy
        assert ig >= -1e-9
        assert ig <= self._getEntropy(target_feature) + 1e-9
        return ig

    def _getGiniIndex(self, feature: pd.Series) -> float:
        # Gini Index formula: Gini = 1 - Σ (p_i^2)
        return 1 - np.sum(
            np.square(
                [(feature == val).sum() / feature.size for val in feature.unique()]
            )
        )


def test_on_iris_wine():
    iris = load_iris()
    wine = load_wine()

    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, train_size=0.7, random_state=0
    )

    # my tree
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(xtrain.join(ytrain), target_col="species")

    preds = dt.predict(xtest)

    print("\n=== Custom Tree (Iris) ===")
    print("Accuracy:", dt.score(xtest, ytest))

    # sklearn tree
    sk_dt = SkDecisionTree(max_depth=3, random_state=0)
    sk_dt.fit(xtrain, ytrain)

    sk_preds = sk_dt.predict(xtest)

    print("\n=== Sklearn Tree (Iris) ===")
    print("Accuracy:", accuracy_score(ytest, sk_preds))

    # wine
    Xw = pd.DataFrame(wine.data, columns=wine.feature_names)
    yw = pd.Series(wine.target, name="class")

    xtrain_w, xtest_w, ytrain_w, ytest_w = train_test_split(
        Xw, yw, train_size=0.7, random_state=0
    )

    dt_w = DecisionTreeClassifier(max_depth=3)
    dt_w.fit(xtrain_w.join(ytrain_w), target_col="class", metric="ig")
    preds_w = dt_w.predict(xtest_w)

    sk_dt_w = SkDecisionTree(max_depth=3, random_state=0)
    sk_dt_w.fit(xtrain_w, ytrain_w)
    sk_preds_w = sk_dt_w.predict(xtest_w)

    print("\n=== WINE ===")
    print("Custom:", dt_w.score(xtest_w, ytest_w))
    print("Sklearn:", accuracy_score(ytest_w, sk_preds_w))


if __name__ == "__main__":
    data = {
        "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast"],
        "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool"],
        "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal"],
        "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong"],
        "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes"],
    }

    df = pd.DataFrame(data)

    X = df.drop(columns=["Play"])
    y = df["Play"]

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, train_size=0.7, random_state=0
    )

    # My decision tree
    dt = DecisionTreeClassifier()
    dt.fit(xtrain.join(ytrain), target_col="Play")

    preds = dt.predict(xtest)

    print("=== Custom Decision Tree ===")
    print("Accuracy:", dt.score(xtest, ytest))

    # SKLEARN TREE
    X_encoded = X.copy()
    encoders = {}

    for col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        encoders[col] = le

    y_le = LabelEncoder()
    y_encoded = y_le.fit_transform(y)

    xtrain_s, xtest_s, ytrain_s, ytest_s = train_test_split(
        X_encoded, y_encoded, train_size=0.7, random_state=0
    )

    sk_dt = SkDecisionTree(max_depth=2, random_state=0)
    sk_dt.fit(xtrain_s, ytrain_s)

    sk_preds = sk_dt.predict(xtest_s)

    print("\n=== Sklearn Decision Tree ===")
    print("Accuracy:", accuracy_score(ytest_s, sk_preds))

    decoded_preds = y_le.inverse_transform(sk_preds)

    test_on_iris_wine()
    # for now there is no numerical support
