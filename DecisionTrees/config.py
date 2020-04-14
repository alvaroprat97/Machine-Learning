NB_LABELS = 4
METRIC_CHOICE = "f1"  # f1, uar, uac, uap

# HYPERPARAMETERS
DEPTHS = [9]
BOUNDARIES = [4]
PRUNING = [True]

DATASET = "noisy"  # clean or noisy
TEST_DATASET = "clean"  # name of an unseen dataset used to evaluate our final tree
