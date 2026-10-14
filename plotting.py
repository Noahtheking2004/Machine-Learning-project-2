import os
import numpy as np

from data_loader import load_dataset
from experiment import run_cv, paired_ttest_5x2
from plotting import plot_sweep, plot_summary, plot_folds

# changeable settings
DATASETS = ["abalone", "hardware", "forestfires"]
TOLERANCES = sorted([0, 0.05, 0.1, 0.25, 0.5, 1.0])
HIGHLIGHTS = [0.1]

PRUNE_FRACTION = 0.2
SEED = 0
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# Run the full experiment on one dataset, print its table, save its graph
def run_dataset(name):
    X, y, is_cat, names = load_dataset(name, DATA_DIR)
    print(f"\n===== {name}: {len(y)} examples, {X.shape[1]} features "
          f"({sum(is_cat)} categorical) =====")

    results = run_cv(X, y, is_cat, TOLERANCES, PRUNE_FRACTION, SEED)

    baseline = np.mean(results["baseline_mse"])
    unpruned = np.mean(results["unpruned_mse"])
    print(f"Predict the mean: MSE {baseline:.4f}")
    print(f"Unpruned: MSE {unpruned:.4f}, "
          f"leaves {np.mean(results['unpruned_leaves']):.1f}\n")
    print(f"{'tolerance':>10} {'pruned MSE':>11} {'leaves':>8} {'t-test p':>9}")
    for t in TOLERANCES:
        _, p = paired_ttest_5x2(results["unpruned_mse"], results["pruned_mse"][t])
        print(f"{t:>10g} {np.mean(results['pruned_mse'][t]):>11.4f} "
              f"{np.mean(results['pruned_leaves'][t]):>8.1f} {p:>9.4f}")

    # best tolerance = lowest average test MSE
    best = min(TOLERANCES, key=lambda t: np.mean(results["pruned_mse"][t]))
    print(f"\nBest tolerance: {best:g}. Graph saved as {name}_sweep.pdf / .png")

    plot_sweep(TOLERANCES, results, f"{name}: pruning tolerance",
               f"{name}_sweep", HIGHLIGHTS, best, "Best", baseline)

    return {"baseline": baseline, "unpruned": unpruned, "best": best,
            "pruned": np.mean(results["pruned_mse"][best]),
            "unpruned_folds": results["unpruned_mse"],
            "pruned_folds": results["pruned_mse"][best]}


# run all datasets, then print a summary and draw the comparison graphs
summary = {}
for name in DATASETS:
    try:
        summary[name] = run_dataset(name)
    except FileNotFoundError as e:
        print(f"\n===== {name}: skipped, data file not found ({e}) =====")

print("\n========== SUMMARY ==========")
print(f"{'dataset':>12} {'mean-guess':>11} {'unpruned':>11} {'pruned':>11} "
      f"{'R2 unpr.':>9} {'R2 pruned':>10} {'best tol':>9}")
for name, s in summary.items():
    r2_un = 1 - s["unpruned"] / s["baseline"]
    r2_pr = 1 - s["pruned"] / s["baseline"]
    print(f"{name:>12} {s['baseline']:>11.4f} {s['unpruned']:>11.4f} "
          f"{s['pruned']:>11.4f} {r2_un:>9.3f} {r2_pr:>10.3f} {s['best']:>9g}")
print("(R2 = 1 - MSE / mean-guess MSE. 0 = no better than guessing the mean.)")

if summary:
    plot_summary(summary, "summary_r2")
    plot_folds(summary, "summary_folds")
    print("Saved summary_r2.pdf/.png and summary_folds.pdf/.png")