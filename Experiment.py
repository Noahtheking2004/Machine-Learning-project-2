import copy
import numpy as np
import Tree as tree

# stop with a clear message if the wrong file was loaded
if not hasattr(tree, "grow"):
    raise ImportError(f"Loaded the wrong file, it has no grow(): {tree.__file__}")


# Mean squared error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# 20% pruning set + 5x2 cross-validation. Returns the results of all 10 folds
def run_cv(X, y, is_cat, tolerances=(0.0,), prune_fraction=0.2, seed=0):
    rng = np.random.default_rng(seed)

    # set aside the pruning set
    order = rng.permutation(len(y))
    n_prune = int(len(y) * prune_fraction)
    X_prune, y_prune = X[order[:n_prune]], y[order[:n_prune]]
    X_cv, y_cv = X[order[n_prune:]], y[order[n_prune:]]

    results = {"baseline_mse": [], "unpruned_mse": [], "unpruned_leaves": [],
               "pruned_mse": {t: [] for t in tolerances},
               "pruned_leaves": {t: [] for t in tolerances}}

    # 5 repeats of 2-fold cross-validation
    for repeat in range(5):
        shuffled = rng.permutation(len(y_cv))
        half = len(y_cv) // 2
        fold_a, fold_b = shuffled[:half], shuffled[half:]

        for train, test in [(fold_a, fold_b), (fold_b, fold_a)]:
            # baseline: ignore the features and predict the training mean
            results["baseline_mse"].append(mse(y_cv[test], y_cv[train].mean()))

            # grow and test the full tree
            root = tree.grow(X_cv[train], y_cv[train], is_cat)
            results["unpruned_mse"].append(
                mse(y_cv[test], tree.predict(root, X_cv[test])))
            results["unpruned_leaves"].append(tree.count_leaves(root))

            # prune a copy for each tolerance and test again
            for t in tolerances:
                pruned = copy.deepcopy(root)
                tree.prune(pruned, X_prune, y_prune, t)
                results["pruned_mse"][t].append(
                    mse(y_cv[test], tree.predict(pruned, X_cv[test])))
                results["pruned_leaves"][t].append(tree.count_leaves(pruned))

    return results


# 5x2cv paired t-test on two lists of 10 fold scores. Returns (t, p-value)
def paired_ttest_5x2(a, b):
    diff = np.array(a) - np.array(b)
    d1, d2 = diff[0::2], diff[1::2]            # first / second fold of each repeat

    variance = (d1 - (d1 + d2) / 2) ** 2 + (d2 - (d1 + d2) / 2) ** 2
    if variance.mean() == 0:
        return float("nan"), float("nan")
    t = d1[0] / np.sqrt(variance.mean())

    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df=5)
    except ImportError:
        p = float("nan")
    return t, p