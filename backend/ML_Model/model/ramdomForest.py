import numpy as np
from .decisionTree import build_tree,predict_v2
def bootstrap_sample(X, y):    
    n = len(X)
    indices = np.random.choice(n, n, replace=True)
    return X[indices], y[indices]
def model_v7(X, y, n_trees=20, max_depth=5, min_samples=10):
    forest = []
    for _ in range(n_trees):
        Xb, yb = bootstrap_sample(X, y)
        tree= build_tree(Xb, yb, 0, max_depth, min_samples)
        forest.append(tree)
    return forest
def forest_predict(X, forest):
    all_preds = []
    for tree in forest:
        preds = predict_v2(X, tree)
        all_preds.append(preds)
    all_preds=np.array(all_preds)
    final_preds = []
    for col in all_preds.T:
        final_preds.append(1 if np.sum(col) >= len(col)/2 else 0)
    return np.array(final_preds)

