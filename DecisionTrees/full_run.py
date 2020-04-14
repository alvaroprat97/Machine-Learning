import warnings

import numpy as np

import config as cfg
import cross_val as cv
import plot_tree as pt
from tree import tree_learn, run_pruning
from evaluation import evaluate
from saved_tree import final_tree as saved_tree

warnings.filterwarnings("ignore")


def run_all():
    """Create the final tree"""
    choice = input(
        "Would you like to rerun hyperparameter tuning or just use the saved tree? rerun/saved \n"
    )
    if choice == "rerun":
        final_tree = create_final_tree()
    elif choice == "saved":
        final_tree = saved_tree
    else:
        print("Please enter a valid response")
        return

    final_scores = test_final_tree(final_tree, cfg.TEST_DATASET)
    print(f"The final scores on the unseen dataset are: {final_scores}")
    return final_scores


def create_final_tree() -> dict:
    """Create final tree based on best hyperparameters from param_tuning and plot"""

    data = np.loadtxt(f"{cfg.DATASET}_dataset.txt")

    # Run parameter tuning
    best_hyper, _, _, _ = cv.param_tuning(data, 10, 0.1)
    print(f"\nBest hyperparameters are: {best_hyper}")

    # Split data to conserve some "unseen" data for pruning evaluation
    split_size = int(0.2 * data.shape[0])
    np.random.shuffle(data)
    train, validate = cv.split(data, 0, split_size)

    non_pruned_tree, _ = tree_learn(
        train,
        depth=0,
        tree={},
        max_depth=best_hyper["depth"],
        reduction=best_hyper["boundary"],
    )

    # Always run pruning as we've seen it's always better
    pruned_tree, _ = run_pruning(non_pruned_tree, train, validate)

    # Plot both trees
    pt.plot_tree(non_pruned_tree, "before_pruning")
    pt.plot_tree(pruned_tree, "after_pruning")
    print(
        (
            "\nThe final tree before pruning and after pruning has been "
            "plotted and saved. You can check them in the root directory "
            "under the file names before_pruning.eps and after_pruning.eps\n"
        )
    )

    return pruned_tree


def test_final_tree(tree: dict, test_data: str = "unseen") -> dict:
    """Test final tree on whatever data you want."""
    data = np.loadtxt(f"{test_data}_dataset.txt")
    scores = evaluate(tree, data)

    return scores


if __name__ == "__main__":
    run_all()
