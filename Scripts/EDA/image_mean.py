from Scripts.DataLoader.custom_dataset import IMDBDataset, MyCollate
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def calculate_image_mean():
    """Calculate the mean image of the training set."""

    # Train config
    train_batch_size = 2048
    train_dataset = IMDBDataset("train", "Data\imdb_crop_processed", premean=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=MyCollate(train_batch_size),
    )

    # Test config
    test_batch_size = 2048
    test_dataset = IMDBDataset("test", "Data\imdb_crop_processed", premean=True)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=MyCollate(test_batch_size),
    )
    mean_images = __dataloader_mean_images(train_dataloader) + __dataloader_mean_images(test_dataloader)

    mean_image = torch.mean(torch.stack(mean_images), dim=0)
    mean_image = mean_image.transpose(0, 2)

    torch.save(mean_image, r"Data\mean_image.pt")

    # Calculate and store mean and std across channels
    get_mean_and_std(mean_image)


def __dataloader_mean_images(dataloader):
    """Calculate the mean images of a dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader to calculate the mean images of.

    Returns:
        list: List of mean images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean_images = []
    for batch in tqdm(dataloader):
        images = batch[0].to(device)
        mean_image_for_batch = torch.mean(images, dim=0)
        mean_images.append(mean_image_for_batch)

    return mean_images


def plot_mean_image():
    """Plot the mean image of the training set."""
    import matplotlib.pyplot as plt
    import numpy as np

    mean_image = torch.load(r"Data\mean_image.pt").to("cpu")
    plt.imshow(mean_image.to(torch.int64))
    plt.savefig(r"Data\plots\mean_image.png")


def get_mean_and_std(mean_image):
    """Get the mean and standard deviation of the mean image across all 3 channels.

    Args:
        mean_image (_type_): Mean image.
    """
    mean_image = mean_image.to("cpu")
    mean_image = mean_image.to(torch.float64)
    means = torch.mean(mean_image, dim=[0, 1])
    stds = torch.std(mean_image, dim=[0, 1])

    # save mean and std
    torch.save(means, r"Data\mean.pt")
    torch.save(stds, r"Data\std.pt")


if __name__ == "__main__":
    mean = calculate_image_mean()
    plot_mean_image()
