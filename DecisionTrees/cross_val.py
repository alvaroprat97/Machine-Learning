import numpy as np
from tree import tree_learn, run_pruning
from evaluation import evaluate, compute_cm
import config as cfg


def split(arr, pos, n):
    """takes an array, splits it in 2 uneven arrays.
    arr : array : The array to be split
    pos : int : the position on which the split occurs
    n : int : the number of rows to take when splitting
    the upper matrix is of size n, the lower one is (size of arr) - N
    returns upper, lower"""

    upper = arr[pos : pos + n]
    lower = np.vstack((arr[:pos], arr[pos + n :]))

    return upper, lower


def hyperparamters_list() -> list:
    """Define the different potential combinations of hyperparameters using config lists"""
    hyparameters = []
    for depth in cfg.DEPTHS:
        for boundary in cfg.BOUNDARIES:
            for prune in cfg.PRUNING:
                hyparameters.append(
                    {"depth": depth, "boundary": boundary, "prune": prune}
                )
    return hyparameters


def cross_validation(data: np.array, folds: int, test_percentage: float) -> tuple:
    """Conduct cross validation
        data : array : this is the complete dataset used
        folds : int : how many folds are used during cross validation
        test_percentage : float : the percentage of the dataset to be used for testing
        returns "we'll see"""

    # copy data to avoid damaging the dataset
    df = data.copy()
    split_size = int(test_percentage * df.shape[0])

    # Collecting the list of hyperparameters
    hyperparameters = hyperparamters_list()

    # Shuffle data
    np.random.shuffle(df)

    # Initialise score trackers
    global_best_hyperparams = []
    global_scores = []

    # Splitting Test from Validation and Training
    for i in range(folds):
        print(f"-------------------- Test/Train separation {i} --------------------")
        test_set, train_and_validate = split(df, i * split_size, split_size)
        # Splitting Validation from Training
        err_all_hyperparams = []
        # For all the sets of hyperparemeters dictionnary
        for hyp in hyperparameters:
            print(f"Running with hyperparameters: {hyp}")
            moi = []  # metric of interest: eg F1
            for j in range(folds - 1):
                # Split data into validation and training set
                validate, train = split(train_and_validate, j * split_size, split_size)

                # Train tree on training data
                tree, _ = tree_learn(
                    train, 0, tree={}, max_depth=hyp["depth"], reduction=hyp["boundary"]
                )
                # Pruning
                if hyp["prune"]:
                    tree, metric_scores = run_pruning(tree, train, validate)
                else:  # Return score of unpruned tree
                    metric_scores = evaluate(tree, validate)[cfg.METRIC_CHOICE]

                moi.append(metric_scores)  # metric_scores is a single metric

            # Make error estimate of all folds for a given hyperparam
            err_all_hyperparams.append(np.mean(moi))

        # Identify most recurrent hyperparameter
        best_hyper = hyperparameters[np.argmax(err_all_hyperparams)]
        global_best_hyperparams.append(best_hyper)

        # Final evaluation with test data per fold
        tree, _ = tree_learn(
            train_and_validate,
            0,
            tree={},
            max_depth=best_hyper["depth"],
            reduction=best_hyper["boundary"],
        )

        # Pruning
        if best_hyper["prune"]:
            tree, _ = run_pruning(tree, train_and_validate, test_set)
        # Store and append all scores for the test set
        score_hyper = evaluate(tree, test_set)
        global_scores.append(score_hyper)

    # Create dictionary of best hyperparameters for all test sets.
    base_dict = {}
    for key in global_scores[0]:
        base_dict[key] = 0
        for i in range(np.size(global_scores)):
            base_dict[key] += global_scores[i][key]
        base_dict[key] = base_dict[key] / (i + 1)

    return global_best_hyperparams, global_scores, base_dict


def param_tuning(data: np.array, folds: int, test_percentage: float) -> tuple:
    """Run parameter tuning."""
    df = data.copy()
    split_size = int(test_percentage * df.shape[0])

    hyperparameters = hyperparamters_list()
    np.random.shuffle(df)

    err_all_hyperparams = []
    err_all_hyperparams_var = []

    cm_list = []

    # For all the sets of hyperparemeters dictionnary
    for hyp in hyperparameters:
        print(f"Running with hyperparameters: {hyp}")
        moi = []  # metric of interest: eg F1

        cm = np.zeros((4, 4))
        for i in range(folds):

            # Split data into evaluation and training datasets
            eval_set, train_set = split(df, i * split_size, split_size)

            # Train tree on training data
            tree, _ = tree_learn(
                train_set, 0, tree={}, max_depth=hyp["depth"], reduction=hyp["boundary"]
            )
            # Run pruning if hyperparameter says to do so
            if hyp["prune"]:
                tree, _ = run_pruning(tree, train_set, eval_set)
            metric_scores = evaluate(tree, eval_set)  # [cfg.METRIC_CHOICE]

            moi.append(
                metric_scores
            )  # metric_scores is F1 only currently as se tin run_pruning output

            cm += compute_cm(eval_set, tree)  # Confusion matrix of each fold

        # Average out the confusion matrix for each hyperparameter
        cm = cm / (i+1)
        cm_list.append(cm)

        arr_scores = np.zeros((4, np.size(moi)))
        metrics = ["uar", "uap", "f1", "uac"]
        for index, key in enumerate(metrics):
            for i in range(np.size(moi)):
                arr_scores[index, i] = moi[i][key]
        variances = np.var(arr_scores, axis=1)
        means = np.mean(arr_scores, axis=1)

        # Format scores to display
        formatted_scores = {i: round(means[c], 2) for c, i in enumerate(metrics)}
        print(f"Final scores for above hyperparameters: {formatted_scores}\n")

        err_all_hyperparams.append(means)
        err_all_hyperparams_var.append(variances)

    err = np.array(err_all_hyperparams)
    var = np.array(err_all_hyperparams_var)
    best_hyper = hyperparamters_list()[np.argmax(err[:, 2])]

    # Display final confusion matrix
    formatted_cm = np.round(cm_list[np.argmax(err[:, 2])], 2)
    print("\nBest averaged confusion matrix : \n", formatted_cm)

    return best_hyper, err, var, cm_list[np.argmax(err[:, 2])]
