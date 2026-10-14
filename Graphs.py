import numpy as np
import matplotlib.pyplot as plt


# Save a figure as PDF + PNG if a name was given
def _save(fig, save_as):
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as + ".pdf")
        fig.savefig(save_as + ".png", dpi=200)


# Open every graph that has been drawn. Call this once, at the very end
def show_all():
    plt.show()


# Top graph: test MSE per tolerance. Bottom graph: number of leaves.
# best = tolerance to star, highlights = extra values to mark,
# baseline = MSE of always predicting the training mean
def draw_sweep(tolerances, results, title, highlights=(), best=None,
               best_label="Best", baseline=None, save_as=None):
    tolerances = np.array(tolerances, float)
    mean_mse = np.array([np.mean(results["pruned_mse"][t]) for t in tolerances])
    std_mse = np.array([np.std(results["pruned_mse"][t], ddof=1) for t in tolerances])
    leaves = np.array([np.mean(results["pruned_leaves"][t]) for t in tolerances])
    unpruned_mse = np.mean(results["unpruned_mse"])
    unpruned_leaves = np.mean(results["unpruned_leaves"])

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # top: test MSE (mean +/- std over the 10 folds)
    top.errorbar(tolerances, mean_mse, yerr=std_mse, marker="o", capsize=3,
                 label="Pruned")
    top.axhline(unpruned_mse, color="k", linestyle="--", label="Unpruned")
    if baseline is not None:
        top.axhline(baseline, color="gray", linestyle=":", label="Predict the mean")
    if best is not None:
        i = int(np.argmin(np.abs(tolerances - best)))
        top.plot(tolerances[i], mean_mse[i], "r*", markersize=16,
                 label=f"{best_label}: tolerance={tolerances[i]:g}")
    top.set_ylabel("Test MSE")
    top.set_title(title)
    top.legend()

    # bottom: tree size
    bottom.plot(tolerances, leaves, marker="s", label="Pruned")
    bottom.axhline(unpruned_leaves, color="k", linestyle="--", label="Unpruned")
    bottom.set_ylabel("Number of leaves")
    bottom.set_xlabel("Pruning tolerance")
    bottom.legend()

    # vertical lines at the values you want to look at
    for value in highlights:
        top.axvline(value, color="gray", linestyle=":")
        bottom.axvline(value, color="gray", linestyle=":")

    _save(fig, save_as)
    return fig


# Bar chart comparing the datasets with R^2 = 1 - MSE / baseline MSE
# (0 = no better than predicting the mean, 1 = perfect)
def draw_summary(summary, save_as=None):
    names = list(summary)
    r2_unpruned = [1 - summary[n]["unpruned"] / summary[n]["baseline"] for n in names]
    r2_pruned = [1 - summary[n]["pruned"] / summary[n]["baseline"] for n in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, r2_unpruned, 0.4, label="Unpruned")
    ax.bar(x + 0.2, r2_pruned, 0.4, label="Pruned (best tolerance)")
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("R^2 (0 = same as predicting the mean)")
    ax.set_title("Model quality across datasets")
    ax.legend()

    _save(fig, save_as)
    return fig


# Box plot of the 10 fold scores, unpruned vs pruned, one panel per dataset
def draw_folds(summary, save_as=None):
    fig, axes = plt.subplots(1, len(summary), figsize=(4 * len(summary), 4))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, summary):
        s = summary[name]
        ax.boxplot([s["unpruned_folds"], s["pruned_folds"]])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Unpruned", "Pruned"])
        ax.axhline(s["baseline"], color="gray", linestyle=":",
                   label="Predict the mean")
        ax.set_title(name)
        ax.set_ylabel("Test MSE per fold")
        ax.legend()

    _save(fig, save_as)
    return fig