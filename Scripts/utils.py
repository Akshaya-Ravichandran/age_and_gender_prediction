import glob
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torchmetrics import F1Score, Accuracy, Precision, Recall
from scipy.io import loadmat
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

np.random.rand(42)


def create_test_folder(
    folder_name="Data/UTKFace_test", test_train_split=0.2, random_seed=42
):
    """Create folder which contains all test images

    Args:
        folder_name (str, optional): . Defaults to 'Data/UTKFace_test'.
        test_train_split (float, optional): split of test and train data. Defaults to 0.8.
        random_seed (int, optional): random seed for reproducibility. Defaults to 42.
    """

    # Set random seed
    np.random.seed(random_seed)

    # Create folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Get names of all images
    images = glob.glob("Data/UTKFace/*.jpg")

    # check all 3 labels exist in image name, if not remove image
    images_to_remove = [image for image in images if image.count("_") != 3]
    for image in images_to_remove:
        os.remove(image)
        images.remove(image)

    # Select test subset
    size_of_test_set = int(len(images) * test_train_split)
    test_images = np.random.choice(images, size=size_of_test_set, replace=False)

    # Move all test images to test folder
    for image in test_images:
        os.rename(image, folder_name + "/" + image.split("\\")[-1])

    print("done creating test folder")


class PytorchTraining:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        val_loader,
        optimizer,
        scheduler,
        age_loss_fn,
        gender_loss_fn,
        device=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.age_loss_fn = age_loss_fn
        self.gender_loss_fn = gender_loss_fn

        # TODO: remove this?
        self.train_losses_ = []
        self.val_losses_ = []

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model = self.model.to(self.device)

    def forward_pass_and_losses(
        self, data, age_target, gender_target, return_scores=False
    ):
        """Forward pass through the model and calculate the losses."""
        # forward
        age_score, gender_score = self.model(data)

        # Loss
        age_loss = self.age_loss_fn(age_score, age_target)
        gender_loss = self.gender_loss_fn(gender_score, gender_target)
        loss = age_loss + gender_loss

        if return_scores:
            return loss, age_score, gender_score
        else:
            return loss

    def train(self, epochs, path, print_freq=100, unfreeze_on_epoch=-1):
        """Train the model.

        Args:
            epochs (int): Number of epochs to train for
            path (str): Path to save the model to
            print_freq (str, optional): frequency to print the loss. Defaults to None.
            unfreeze_on_epoch (int, optional): unfreeze the model on this epoch. Defaults to None.
        """
        self.train_losses = []

        running_train_print_loss = []
        running_val_loss = []

        self.test_losses = []
        for epoch in range(epochs):

            # Unfreeze model params
            if epoch == unfreeze_on_epoch:
                print("Unfreezing model parameters")
                for param in self.model.parameters():
                    param.requires_grad = True

            epoch_train_losses = []
            for batch_idx, (data, targets) in enumerate(self.train_loader):

                self.model.train()

                # Send data to device
                (
                    data,
                    age_target,
                    gender_target,
                ) = self.__send_data_to_device(data, targets)

                # forward and loss
                loss = self.forward_pass_and_losses(
                    data, age_target, gender_target
                )

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                self.optimizer.step()

                # Record loss
                epoch_train_losses.append(loss.item())
                running_train_print_loss.append(loss.item())

                running_val_loss.append(self.__get_val_loss())

                # Print losses every print_freq batches and evaluate scheduler
                if print_freq and batch_idx % print_freq == 0:

                    val_loss = np.mean(running_val_loss)
                    train_loss = np.mean(running_train_print_loss)

                    self.train_losses_.append(train_loss)
                    self.val_losses_.append(val_loss)

                    print(
                        f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(self.train_loader)} Train_Loss: {train_loss:.4f} Val_Loss: {val_loss:.4f}"
                    )
                    self.scheduler.step(val_loss)

                    # Reset running losses
                    running_train_print_loss = []
                    running_val_loss = []

            train_loss_for_epoch = np.mean(epoch_train_losses)
            self.train_losses.append(train_loss_for_epoch)
            self.test_losses.append(self.evaluate())
            print(
                f"Epoch [{epoch}/{epochs}] Train loss: {self.train_losses[-1]:.4f} test loss: {self.test_losses[-1]:.4f}"
            )

        torch.save(self.model.state_dict(), path)

    def plot_losses(self):
        """Plot the training and test losses."""

        plt.plot(self.train_losses, label="train")
        plt.plot(self.test_losses, label="test")

        plt.title("Training and Test Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.legend()
        plt.show()
        # TODO: plot point we unfreeze params

    def plot_losses_(self):
        """Plot the training and test losses."""

        plt.plot(self.train_losses_, label="train")
        plt.plot(self.val_losses_, label="test")

        plt.title("Training and Test Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.legend()
        plt.show()
        # TODO: plot point we unfreeze params

    def evaluate(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        with torch.no_grad():
            losses = []
            for batch_idx, (data, targets) in enumerate(self.test_loader):

                # Send data to device
                (
                    data,
                    age_target,
                    gender_target,
                ) = self.__send_data_to_device(data, targets)

                # forward and loss
                loss = self.forward_pass_and_losses(
                    data, age_target, gender_target
                )
                losses.append(loss.item())

        return np.mean(losses)

    def __get_val_loss(self):

        with torch.no_grad():

            self.model.eval()

            # Send data to device
            X, y = next(iter(self.val_loader))
            data, age_target, gender_target = self.__send_data_to_device(X, y)

            # Use val loss to step scheduler
            val_loss = self.forward_pass_and_losses(
                data, age_target, gender_target
            )

        return val_loss.item()

    def __send_data_to_device(self, data, targets):
        """Send data to device and splits targets out"""

        data = data.to(self.device)

        age_target, gender_target = targets
        age_target = age_target.to(self.device)
        gender_target = gender_target.to(self.device)

        return data, age_target, gender_target

    def calculate_metrics(self):

        test_preds = self.__generate_predictions(self.test_loader)
        train_preds = self.__generate_predictions(self.train_loader)

        # Save predictions
        test_preds.to_csv("Data/Metrics/test_preds.csv", index=False)
        train_preds.to_csv("Data/Metrics/train_preds.csv", index=False)

        # Calculate metrics
        self.__calculate_metrics(test_preds, "test")
        self.__calculate_metrics(train_preds, "train")

    # ----------------- subroutines to help calculate metrics ----------------- #

    def __generate_predictions(self, loader):
        """Generate predictions for the data and targets.

        Args:
            data (torch.Tensor): Data to generate predictions for
            targets (torch.Tensor): Targets to generate predictions for

        Returns:
            [type]: [description]
        """
        # Send data to device
        self.model.eval()
        frames = []
        print("Generating predictions...")
        for _, (data, targets) in tqdm(enumerate(loader), total=len(loader)):

            # Send data to device
            data, age_target, gender_target = self.__send_data_to_device(
                data, targets
            )

            # forward and loss
            with torch.no_grad():
                loss, age_score, gender_score = self.forward_pass_and_losses(
                    data, age_target, gender_target, True
                )

            # Generate predictions
            age_pred, gender_pred = self.model.transform_scores_to_predictions(
                age_score, gender_score
            )

            # store preds and targets
            df = pd.DataFrame(
                {
                    "age_actual": age_target.to("cpu"),
                    "gender_actual": gender_target.to("cpu"),
                    "age_pred": age_pred.to("cpu"),
                    "gender_pred": gender_pred.to("cpu"),
                }
            )
            frames.append(df)

        predctions = pd.concat(frames)
        return predctions

    def __calculate_metrics(self, df, name):
        """Sub routine to calculate metrics for a dataframe of predictions and targets.

        Args:
            df (pd.DataFrame): Dataframe of predictions and targets
            name (str): Name of the dataset
        """

        # Age metrics
        matrix = confusion_matrix(df["age_actual"], df["age_pred"])
        sns.heatmap(matrix, annot=True)
        plt.savefig(f"Data/Plots/age_confusion_matrix_{name}.png")
        plt.close()
        age_cks = cohen_kappa_score(
            df["age_actual"], df["age_pred"], weights="quadratic"
        )
        age_acc = accuracy_score(df["age_actual"], df["age_pred"])

        # Gender metrics
        gender_f1 = f1_score(df["gender_actual"], df["gender_pred"])
        gender_acc = accuracy_score(df["gender_actual"], df["gender_pred"])
        gender_precision = precision_score(
            df["gender_actual"], df["gender_pred"]
        )
        gender_recall = recall_score(df["gender_actual"], df["gender_pred"])

        # Create dataframe
        metrics = pd.DataFrame(
            {
                "age_cohen_kappa_score": [age_cks],
                "age_accuracy": [age_acc],
                "gender_f1": gender_f1,
                "gender_acc": gender_acc,
                "gender_precision": gender_precision,
                "gender_recall": gender_recall,
            }
        )

        metrics.to_csv(f"Data/Metrics/Metrics_{name}.csv")

    def generate_salient_maps(self):
        """Generate salient maps for the model."""

        # Send data to device
        X, y = next(iter(self.test_loader))


def get_number_of_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def transform_ages(ages):
    """Transform ages to ordinal cat encodings

    Args:
        ages (np.array): array of ages
    """

    # age_bins_to_labels = {
    #     (0, 10): 0,
    #     (11, 20): 1,
    #     (21, 30): 2,
    #     (31, 40): 3,
    #     (41, 50): 4,
    #     (51, 60): 5,
    #     (61, 70): 6,
    #     (71, 80): 7,
    #     (81, 90): 8,
    #     (91, 100): 9,
    # }
    age_bins_to_labels = {
        (0, 12): 0,
        (13, 17): 1,
        (18, 29): 2,
        (30, 39): 3,
        (40, 49): 4,
        (50, 59): 5,
        (60, 69): 6,
        (70, 100): 7,
    }
    age_brackets = age_bins_to_labels.keys()
    ages_to_age_bins = dict()
    for age_bracket in age_brackets:
        for age in range(age_bracket[0], age_bracket[1] + 1):
            ages_to_age_bins[age] = age_bracket

    outputs = []
    for age in ages:
        age_bracket = ages_to_age_bins[age.item()]
        ordinal_label = age_bins_to_labels[age_bracket]
        outputs.append(ordinal_label)
    ages = torch.Tensor(outputs)

    return ages
