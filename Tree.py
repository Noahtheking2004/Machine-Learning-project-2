import numpy as np


# One spot in the tree: either a leaf or a split on a feature
class Node:
    def __init__(self, prediction):
        self.prediction = prediction   # mean target of the training rows here
        self.feature = None            # column used to split (None = leaf)
        self.threshold = None          # numeric split: x <= threshold goes left
        self.left = None
        self.right = None
        self.children = {}             # categorical split: {value: child node}

    # True if this node has no split
    def is_leaf(self):
        return self.feature is None

    # Pruning: throw away everything below this node
    def make_leaf(self):
        self.feature = self.threshold = self.left = self.right = None
        self.children = {}


# ---------------------------------------------------------- split search

# Best threshold for a numeric feature. Returns (error, threshold) or None
def numeric_split(x, y):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = len(y)

    # places where the value changes (only these can be split points)
    cut = np.nonzero(xs[:-1] < xs[1:])[0]
    if len(cut) == 0:
        return None

    # running sums let us score every split point quickly
    csum, csum2 = np.cumsum(ys), np.cumsum(ys ** 2)
    n_left = cut + 1.0
    n_right = n - n_left
    s_left, s2_left = csum[cut], csum2[cut]
    s_right, s2_right = csum[-1] - s_left, csum2[-1] - s2_left

    # squared error of the left side plus the right side
    sse = (s2_left - s_left ** 2 / n_left) + (s2_right - s_right ** 2 / n_right)

    best = np.argmin(sse)
    threshold = (xs[cut[best]] + xs[cut[best] + 1]) / 2   # midpoint
    return sse[best] / n, threshold


# Error of splitting a categorical feature into one child per value
def categorical_split(x, y):
    values, group = np.unique(x, return_inverse=True)
    if len(values) < 2:
        return None
    count = np.bincount(group)
    total = np.bincount(group, weights=y)
    total2 = np.bincount(group, weights=y ** 2)
    sse = (total2 - total ** 2 / count).sum()
    return sse / len(y)


# ------------------------------------------------------------- growing

# Grow the tree as far as it will go (no early stopping)
def grow(X, y, is_cat):
    node = Node(y.mean())
    if y.max() == y.min():             # all targets equal -> stop
        return node

    # try every feature and keep the split with the lowest error
    best_error, best_feature, best_threshold = None, None, None
    for j in range(X.shape[1]):
        if is_cat[j]:
            error, threshold = categorical_split(X[:, j], y), None
        else:
            result = numeric_split(X[:, j], y)
            error, threshold = (None, None) if result is None else result
        if error is not None and (best_error is None or error < best_error - 1e-12):
            best_error, best_feature, best_threshold = error, j, threshold
    if best_feature is None:           # nothing can split this node
        return node

    node.feature = best_feature
    node.threshold = best_threshold
    column = X[:, best_feature]
    if is_cat[best_feature]:           # one child per value
        for value in np.unique(column):
            rows = column == value
            node.children[value] = grow(X[rows], y[rows], is_cat)
    else:                              # two children: <= threshold and above
        rows = column <= best_threshold
        node.left = grow(X[rows], y[rows], is_cat)
        node.right = grow(X[~rows], y[~rows], is_cat)
    return node


# ---------------------------------------------------------- predicting

# Follow one row down the tree and return the leaf's prediction
def predict_one(node, x):
    while not node.is_leaf():
        if node.children:                          # categorical split
            if x[node.feature] not in node.children:
                break                              # unseen value -> use this node's mean
            node = node.children[x[node.feature]]
        elif x[node.feature] <= node.threshold:    # numeric split
            node = node.left
        else:
            node = node.right
    return node.prediction


# Predictions for every row of X
def predict(node, X):
    return np.array([predict_one(node, row) for row in X])


# Number of leaves in the tree
def count_leaves(node):
    if node.is_leaf():
        return 1
    kids = list(node.children.values()) if node.children else [node.left, node.right]
    return sum(count_leaves(k) for k in kids)


# ------------------------------------------------------------- pruning

# Reduced-error pruning, bottom-up. X, y = pruning rows that reach this node.
# Turns a node into a leaf if that is no worse than keeping the subtree
# (tolerance > 0 allows a slightly worse leaf, which prunes more).
# Returns the squared error of this node's subtree on those rows.
def prune(node, X, y, tolerance=0.0):
    leaf_error = ((y - node.prediction) ** 2).sum()
    if node.is_leaf():
        return leaf_error

    column = X[:, node.feature]
    if node.children:                              # categorical split
        subtree_error = 0.0
        seen = np.zeros(len(y), dtype=bool)
        for value, child in node.children.items():
            rows = column == value
            seen |= rows
            subtree_error += prune(child, X[rows], y[rows], tolerance)
        # rows with a value the tree never saw use this node's mean
        subtree_error += ((y[~seen] - node.prediction) ** 2).sum()
    else:                                          # numeric split
        rows = column <= node.threshold
        subtree_error = (prune(node.left, X[rows], y[rows], tolerance)
                         + prune(node.right, X[~rows], y[~rows], tolerance))

    if leaf_error <= subtree_error * (1 + tolerance) + 1e-12:
        node.make_leaf()
        return leaf_error
    return subtree_error