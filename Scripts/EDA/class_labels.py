import glob
import re
import numpy as np
import torch
from Scripts import utils
import pandas as pd


def pull_age_labels():

    Ages = list(pd.read_csv("Data\Metadata\meta_data_test.csv")["age"]) + list(pd.read_csv("Data\Metadata\meta_data_train.csv")["age"])
    Ages = np.array(Ages)

    Ages = utils.transform_ages(Ages)

    return Ages


def get_age_class_weights():

    ages = pull_age_labels()

    class_weights = []
    labels = count_labels(ages.tolist())
    # order dict by key
    labels = dict(sorted(labels.items()))

    # W_i = n_samples / (n_classes * n_samples_class_i)
    for label in labels:
        class_weights.append(len(ages) / (len(labels) * labels[label]))

    return torch.Tensor(class_weights)


# def get_gender_class_weights


def count_labels(labels):
    """Count the number of labels in a list of labels

    Args:
        labels (list): list of labels

    Returns:
        dict: dictionary containing the number of labels
    """
    label_count = {}
    for label in labels:

        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1

    return label_count


# d = count_labels(Ages)
# print("d")

# age_bins_to_labels = {
#     (1, 10): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     (11, 20): [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     (21, 30): [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     (31, 40): [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     (41, 50): [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#     (51, 60): [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#     (61, 70): [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#     (71, 80): [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#     (81, 90): [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#     (91, 100): [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     (101, 110): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#     (111, 120): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# }

# {v: k for k, v in age_bins_to_labels.items()}


# def inver_dict(d):
#     """Invert a dictionary

#     Args:
#         d (dict): dictionary to invert

#     Returns:
#         dict: inverted dictionary
#     """
#     return {v: k for k, v in d.items()}
