import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def compute_gini(y):
    if len(y) == 0:
        return 0
    p1 = np.mean(y == 1)
    p0 = 1 - p1
    return 1 - (p0**2 + p1**2)


def split_dataset(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]


def get_thresholds(col):
    values = np.unique(col)
    if len(values) <= 1:
        return []
    return (values[:-1] + values[1:]) / 2


def find_best_split(X, y, min_samples):

    parent_gini = compute_gini(y)

    best_gini = float("inf")
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]

    for f in range(n_features):

        thresholds = get_thresholds(X[:, f])

        for t in thresholds:

            Xl, yl, Xr, yr = split_dataset(X, y, f, t)

            if len(yl) < min_samples or len(yr) < min_samples:
                continue

            g = (len(yl)/len(y))*compute_gini(yl) + \
                (len(yr)/len(y))*compute_gini(yr)

            if g < best_gini:
                best_gini = g
                best_feature = f
                best_threshold = t

    if best_feature is None or best_gini >= parent_gini:
        return None, None, None

    return best_feature, best_threshold, best_gini


def leaf_value(y):
    return 1 if np.sum(y) >= len(y)/2 else 0


def build_tree(X, y, depth, max_depth, min_samples):

    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < min_samples:
        return leaf_value(y)

    feature, threshold, _ = find_best_split(X, y, min_samples)

    if feature is None:
        return leaf_value(y)

    Xl, yl, Xr, yr = split_dataset(X, y, feature, threshold)

    left = build_tree(Xl, yl, depth+1, max_depth, min_samples)
    right = build_tree(Xr, yr, depth+1, max_depth, min_samples)

    return {
        "feature": feature,
        "threshold": threshold,
        "left": left,
        "right": right
    }


def predict_one(x, tree):
    while isinstance(tree, dict):
        if x[tree["feature"]] <= tree["threshold"]:
            tree = tree["left"]
        else:
            tree = tree["right"]

    return tree


def predict_v2(X, tree):
    return np.array([predict_one(x, tree) for x in X])


def model_v5(X, y, max_depth=6, min_samples=10):
    return build_tree(X, y, 0, max_depth, min_samples)
def model_v6(X, y,X_test,y_test, max_depth=5, min_samples=15):
    tree = DecisionTreeClassifier(
        max_depth=5
    )
    
    tree.fit(X, y)
    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print("Sklearn Decision Tree Accuracy:", acc)