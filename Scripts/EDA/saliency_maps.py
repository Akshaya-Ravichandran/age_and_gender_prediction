import torch
from Scripts.Modelling.final_model import MultiTaskNet
from Scripts.DataLoader.custom_dataset import IMDBDataset, MyCollate
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# channel_means = torch.load(r"Data\mean.pt").to("cpu")
# channel_stds = torch.load(r"Data\std.pt").to("cpu")


# # Test config
# test_batch_size = 1
# test_dataset = IMDBDataset("test")
# test_dataloader = DataLoader(
#     test_dataset,
#     batch_size=test_batch_size,
#     shuffle=True,
#     collate_fn=MyCollate(test_batch_size),
# )

# # load in model
# model = MultiTaskNet(pretrained=True, freeze_mid_layers=False)
# model = model.to(device)

# # load model
# model.load_state_dict(torch.load("model.pt"))
# model.eval()

# X, Y = next(iter(test_dataloader))
# X = X.to(device)
# X.requires_grad_()
# age_score, gender_score = model(X)

# gender_score.backward(retain_graph=True)

# saliency, _ = torch.max(X.grad.data.abs(), dim=1)
# saliency = torch.squeeze(saliency.to("cpu"))


# # denormalise image
# image = torch.squeeze(X).transpose(0, 2).to("cpu")
# image = (image * channel_stds) + channel_means
# image = image.to(torch.int64)

# plt.imshow(image)
# plt.imshow(saliency, cmap=plt.cm.hot, alpha=0.5)
# plt.savefig(r"Data\Plots\saliency_maps\gender\test.png")
# plt.close()


# age_index = age_score.argmax().item()
# age_output = age_score[0, age_index]

# age_output.backward()
# saliency, _ = torch.max(X.grad.data.abs(), dim=1)
# saliency = torch.squeeze(saliency.to("cpu"))

# plt.imshow(image)
# plt.imshow(saliency, cmap=plt.cm.hot, alpha=0.5)
# plt.savefig(r"Data\Plots\saliency_maps\age\test.png")
# plt.close()


def create_saliency_maps(n_maps=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channel_means = torch.load(r"Data\mean.pt").to("cpu")
    channel_stds = torch.load(r"Data\std.pt").to("cpu")

    # Load in model and set to eval mode
    model = MultiTaskNet(pretrained=True, freeze_mid_layers=False)
    model = model.to(device)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    # Data loader
    test_batch_size = 1
    test_dataset = IMDBDataset("test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        collate_fn=MyCollate(test_batch_size),
    )

    # Loop through data loader and create saliency maps
    for i in range(n_maps):
        X, Y = next(iter(test_dataloader))
        X = X.to(device)
        X.requires_grad_()
        age_score, gender_score = model(X)

        # Gender saliency map
        gender_score.backward(retain_graph=True)
        saliency_gender, _ = torch.max(X.grad.data.abs(), dim=1)
        saliency_gender = torch.squeeze(saliency_gender.to("cpu"))

        # Age saliency map
        age_index = age_score.argmax().item()
        age_output = age_score[0, age_index]
        age_output.backward()
        saliency_age, _ = torch.max(X.grad.data.abs(), dim=1)
        saliency_age = torch.squeeze(saliency_age.to("cpu"))

        # denormalise image
        image = torch.squeeze(X).transpose(0, 2).to("cpu")
        image = (image * channel_stds) + channel_means
        image = image.to(torch.int64)

        # Save saliency maps
        plt.imshow(image)
        plt.imshow(saliency_gender, cmap=plt.cm.hot, alpha=0.5)
        plt.savefig(r"Data\Plots\saliency_maps\gender\{}.png".format(i))
        plt.close()

        plt.imshow(image)
        plt.imshow(saliency_age, cmap=plt.cm.hot, alpha=0.5)
        plt.savefig(r"Data\Plots\saliency_maps\age\{}.png".format(i))
        plt.close()


if __name__ == "__main__":
    create_saliency_maps()
    print("done")
