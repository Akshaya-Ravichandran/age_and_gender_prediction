from Scripts.Modelling.resnet import resnet18, resnet50
from Scripts.Modelling.final_model import MultiTaskNet
from Scripts.DataLoader.custom_dataset import IMDBDataset, MyCollate
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import math
import torch.optim as optim
from Scripts.utils import PytorchTraining
from Scripts.EDA.class_labels import get_age_class_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train config
train_batch_size = 64
train_dataset = IMDBDataset("train", "Data\imdb_crop_processed")
train_dataloader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    collate_fn=MyCollate(train_batch_size),
)

# Test config
test_batch_size = 128
test_dataset = IMDBDataset("test", "Data\imdb_crop_processed")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=True,
    collate_fn=MyCollate(test_batch_size),
)

# Validation config
val_batch_size = 64
val_dataset = IMDBDataset("test", "Data\imdb_crop_processed")
val_dataloader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    shuffle=True,
    collate_fn=MyCollate(val_batch_size),
)

model = MultiTaskNet(pretrained=True, freeze_mid_layers=False)
model = model.to(device)
# load model
# model.load_state_dict(torch.load("model.pt"))
# X, Y = next(iter(train_dataloader))
# X = X.to(device)
# a = model.predict(X)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.2,
    patience=7,
    threshold=0.0001,
    threshold_mode="rel",
    cooldown=0,
    min_lr=0.000000001,
    eps=1e-08,
    verbose=True,
)
class_weights = get_age_class_weights().to(device)
age_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

gender_loss_fn = nn.BCELoss()


trainer = PytorchTraining(
    model,
    train_dataloader,
    test_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    age_loss_fn,
    gender_loss_fn,
)
# trainer.calculate_metrics()

trainer.train(2, "model_new_ages.pt", 50, unfreeze_on_epoch=20)
trainer.calculate_metrics()


print("d")

# X, Y = next(iter(test_dataloader))
# model.predict(X)
