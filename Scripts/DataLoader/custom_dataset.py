import glob
import re
import os
import csv

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .. import utils


class IMDBDataset(Dataset):
    """Custom Dataset class for loading images from the UTKFace dataset"""

    def __init__(self, test_or_train, folder_directory="Data\imdb_crop", premean=False):
        """Initialize the dataset

        Args:
            test_or_train (str): "test" or "train"
            folder_directory (str): path to the folder containing the images
            premean (bool): whether or not the image mean has been calculated yet
        """
        self.folder_directory = folder_directory
        if test_or_train == "test":
            self.df_path = "Data\Metadata\meta_data_test.csv"
        elif test_or_train == "train":
            self.df_path = "Data\Metadata\meta_data_train.csv"
        else:
            raise ValueError("test_or_train must be 'test' or 'train'")

        self.pre_mean = premean
        if not self.pre_mean:
            self.channel_means = torch.load(r"Data\mean.pt").to("cpu")
            self.channel_stds = torch.load(r"Data\std.pt").to("cpu")

        self.df = pd.read_csv(self.df_path)

        assert os.path.exists(self.df_path), "Path to metadata does not exist"

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.df)

    def __getitem__(self, index):
        """Return the image and label at the given index"""

        image_path, age, gender = self.df.iloc[index].values.ravel()

        # Load image into Tensor
        image_path = os.path.join(self.folder_directory, image_path)
        image = Image.open(image_path).resize((224, 224))
        image = np.array(image)

        image = torch.Tensor(image)
        # Standardnise image
        if not self.pre_mean:
            image = (image - self.channel_means) / self.channel_stds

        # Convert labels to ints and drop date label
        label = tuple([int(age), int(gender)])
        label = torch.Tensor(label)

        # shape assertions
        assert image.shape == (224, 224, 3)
        assert len(label) == 2

        return image, label


class MyCollate:
    """Custom collate function for the UTKFace dataset"""

    def __init__(self, batch_size):
        """Initialize the collate function

        Args:
            batch_size (int): size of the batch
        """
        self.batch_size = batch_size

    def __call__(self, batch):
        """Return the batched images and labels

        Args:
            batch (list): list of tuples containing the images and labels
        """
        images, labels = zip(*batch)

        # Convert images to tensor and reorder dimensions
        images = torch.stack(images, dim=0)
        images = images.transpose(1, 3).float()

        # Convert labels to tensor
        labels = torch.stack(labels, dim=0)

        # Transform ages to categorial labels
        ages = np.array(labels[:, 0]).astype(int)
        ages = utils.transform_ages(ages)
        ages = ages.type(torch.LongTensor)

        gender = labels[:, 1]

        labels = (ages, gender)

        # shape assertions
        assert images.shape[1:] == (3, 224, 224)  # Ignore batch_size
        assert len(labels) == 2

        return images, labels
