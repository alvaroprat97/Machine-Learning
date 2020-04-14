import copy

import numpy as np

from evaluation import evaluate
import config as cfg


def entropy(labels: np.array) -> float:
    """Calculates entropy based on an array of labels"""
    size = labels.size
    res = 0
    for i in np.unique(labels):
        pk = len(labels[labels == i]) / size
        res += pk * np.log2(pk)
    return -res


def remainder(s_left: np.array, s_right: np.array) -> float:
    """Calculates remainder from info_gain function."""
    norm_const = lambda x, y: x.size / (x.size + y.size)
    res = norm_const(s_left, s_right) * entropy(s_left) + norm_const(
        s_right, s_left
    ) * entropy(s_right)
    return res


def info_gain(s_all: np.array, bound) -> float:
    """Calculates the information gain on a split."""
    sorted_arr = np.sort(s_all, kind="heapsort")
    s_left = sorted_arr[sorted_arr[:, 0] > bound][
        :, 1
    ]  # labels are solely required for entropy calculation... hence only those are passed
    s_right = sorted_arr[sorted_arr[:, 0] < bound][:, 1]
    return entropy(sorted_arr[:, 1]) - remainder(s_left, s_right)


def get_boundaries(arr: np.array, reduction: float) -> np.array:
    """Retrieves the boundaries given a reduction size"""

    unique = np.unique(arr.copy())
    x = unique[:-1]
    y = unique[1:]
    boundary = (x + y) / 2

    return boundary[::reduction]


def find_split(data: np.array, reduction: float) -> dict:
    """Find best place to split data."""
    top_gain = 0
    split = {}
    for col in range(0, data.shape[1] - 1):  # don't include last col: label col
        for boundary in get_boundaries(data[:, col], reduction):
            temp_gain = info_gain(data[:, [col, -1]], boundary)
            if temp_gain > top_gain:
                top_gain = temp_gain
                split = {
                    "wifi_signal": col,
                    "dB": boundary,
                    "info_label": find_leaf_value(data),
                }  # unindent till out of first for loop...

    return split


def find_leaf_value(data: np.array) -> int:
    """Given a set of data, find most recurrent label"""
    unique, counts = np.unique(data[:, -1], return_counts=True)
    value = unique[np.argmax(counts)]
    return value


def split_data(arr: np.array, col: int, bound: float) -> tuple:
    """Sort and split data based on column and boundary."""
    arr = arr[arr[:, col].argsort()]  # sort array
    left = arr[arr[:, col] > bound]
    right = arr[arr[:, col] < bound]
    return left, right


def tree_learn(
    data: np.array, depth: int, tree: dict, max_depth: int, reduction: float
) -> dict:
    """Creates a decision tree based on a set of data and a max_depth parameter
    data : data to parse
    depth : keeps track of depth of decision tree
    tree : dictionnary of dictionnaries : this is the decision tree
    max_depth: depth at which to stop creating branches
    reduction: sets the boundary for information gain potential splits
    returns the completed decision tree"""

    # Stop recursion if we reach the max_depth of the tree
    if depth == max_depth:
        unique, counts = np.unique(data[:, -1], return_counts=True)
        tree = unique[np.argmax(counts)]  # Output most recurrent label
        return tree, depth
    # Stop recursion if there is only one label in the data segment
    if np.all(data[:, -1] == data[0, -1]):  # check if all labels are identical
        tree = data[0, -1]
        return tree, depth
    # Identify column and value on which to split data
    split = find_split(data, reduction)
    # Open up two new branches
    split["left"] = {}
    split["right"] = {}
    tree = split
    # Split the data based on the best values found
    l_data, r_data = split_data(data, split["wifi_signal"], split["dB"])
    # Recursively rerun function in both new branches
    l_branch, l_depth = tree_learn(
        l_data, depth + 1, split["left"], max_depth, reduction
    )
    r_branch, r_depth = tree_learn(
        r_data, depth + 1, split["right"], max_depth, reduction
    )
    tree["left"] = l_branch
    tree["right"] = r_branch

    return tree, max(l_depth, r_depth)


def evaluate_prune(tree: dict, test: np.array, base_score: float, track: list) -> dict:
    """Prune and evaluate whether we want to keep pruned tree or original tree."""
    original = copy.deepcopy(tree)  # keep copy of pre-pruned tree
    # Prune tree by replacing a branch with a single value
    leaf_value = get_nested_value(tree, track)["info_label"]
    set_nested_value(tree, track, leaf_value)
    # Score the new pruned tree
    prune_score = evaluate(tree, test)[cfg.METRIC_CHOICE]
    # Compare the pruned tree with the original tree
    if prune_score >= base_score:
        return tree, 1  # pruned
    return original, 0


def prune_tree(
    tree: dict,
    branch: dict,
    train: np.array,
    test: np.array,
    base_score: float,
    track: list,
    prune_count: int,
) -> dict:

    """Recursively loop through tree until you get to bottom
    and then prune appropriately once you reach the bottom.

    tree: nested dictionary decision tree
    branch: smaller segments of the tree as we go down the levels
    train: training dataset
    test: test set on which we evaluate. Used test to not confuse with evaluate function
    base_score: the original score of decision tree before pruning
    track: Keeps track of list of keys as we go down tree
    prune_count: number of branches transformed into leaves
    """
    # Initialize branch to be full tree before we recursively enter the branches
    if branch is None:
        branch = tree
    # Stop running once you reach a leaf
    if isinstance(branch, float):
        return tree, prune_count
    # One step look aheads allow us to see if next level down there are leafs on both
    # sides, in which case we can prune–or not
    if (isinstance(branch["left"], float)) and (isinstance(branch["right"], float)):
        # if pruning is worth it, the pruned tree becomes the base tree
        tree, prune_bool = evaluate_prune(tree, test, base_score, track)
        prune_count += prune_bool  # If pruned successful, add 1 to prune_count
        return tree, prune_count
    for i in ["left", "right"]:
        # Store track so we can use original with left and right
        track_copy = copy.deepcopy(track)
        track.append(i)
        # Recurisvely travel through tree until we get to penultimate depth
        tree, prune_count = prune_tree(
            tree, branch[i], train, test, base_score, track, prune_count
        )
        track = copy.deepcopy(
            track_copy
        )  # reinitialise track for right after done with left
    return tree, prune_count


def run_pruning(tree: dict, train: np.array, test: np.array) -> tuple:
    """Runs the prune_tree function until there is no more pruning to be done
    as it would not increase the evaluation score."""

    # Get first base score
    base_score = evaluate(tree, test)[cfg.METRIC_CHOICE]

    keep_pruning = True
    new_prune_count = 0
    while keep_pruning:
        old_prune_count = new_prune_count
        # Run one sweep of prunes at bottomest depths of tree
        tree, prunes = prune_tree(tree, None, train, test, base_score, [], 0)
        new_prune_count = old_prune_count + prunes  # Update prune count

        # Change Base score at each iteration
        base_score = evaluate(tree, test)[cfg.METRIC_CHOICE]
        # Stop pruning if no more prunes are being made
        if new_prune_count == old_prune_count:
            keep_pruning = False
    return tree, base_score


def get_nested_value(nested_dict: dict, key_list: list) -> dict:
    """Retrieve a value in a nested dictionary given a list of keys."""
    for k in key_list:
        nested_dict = nested_dict[k]
    return nested_dict


def set_nested_value(nested_dict: dict, key_list: list, value: float) -> None:
    """Set a value in a nested dictionary given a list of keys."""
    for key in key_list[:-1]:
        nested_dict = nested_dict.setdefault(key, {})
    nested_dict[key_list[-1]] = value
