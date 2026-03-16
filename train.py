import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import yaml

from unet import DiffusionModel


# load config

params_filepath = os.path.dirname(os.path.abspath(__file__)) + "/" + "params.yml"

with open(params_filepath, "r") as f:
    params = yaml.safe_load(f)

# Define image transforms (normalize to roughly zero mean / unit variance)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Training dataset
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Val dataset
val_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print('Training set has {} instances'.format(len(train_dataset)))
print('Validation set has {} instances'.format(len(val_dataset)))

train_loader = DataLoader(
    train_dataset,
    batch_size= params["batch_size"],
    shuffle=True,
    num_workers= params["num_workers"]
    )
val_loader = DataLoader(
    val_dataset,
    batch_size=params["batch_size"],
    shuffle=False,
    num_workers=params["num_workers"]
    )

# Get number of classes from the dataset
n_classes = len(train_loader.dataset.classes)

# Create the diffusion model
dfm = DiffusionModel(
    device=params["device"],
    in_channels=params["in_channels"],
    time_const=params["time_const"],
    conditional=params["conditional"],
    n_classes=n_classes,
    time_embedding_norm=params["time_embedding_norm"],
    block_depth=params["block_depth"],
    block_sizes=params["block_sizes"],
    n_res_blocks=params["n_res_blocks"],
    noise_embed_dim=params["noise_embed_dim"],
    n_groups=params["n_groups"],
    n_attn_heads=params["n_attn_heads"],
    attn_levels=params["attn_levels"],
    norm_attn=params["norm_attn"],
    dropout=params["dropout"],
    beta_schedule=params["beta_schedule"],
    sched_min=params["sched_min"],
    sched_max=params["sched_max"],
    prediction_type=params["prediction_type"],
    ema_decay=params["ema_decay"],
)
# x = df.model(torch.zeros((10,3,32,32)),torch.zeros((10,1)),torch.zeros((10,),dtype=torch.int))
# print(x.shape)

label_names = train_dataset.classes

print(label_names)

# dfm.train(
#     steps = 200_000,
#     train_dataloader = train_loader,
#     val_dataloader = val_loader,
#     lr = params["lr"],
#     uncond_prob = params["uncond_prob"],
#     grad_norm = params["grad_norm"],
#     log_every = 10_000,
# )

dfm.load("./ckpts/bigger.pt")

x = dfm.generate(
    [6]*20,
    (32,32),
    5.0,
    seed = 0,
)

import matplotlib.pyplot as plt
import math

# x is (N, 3, H, W)
x = x.detach().cpu()
N = x.shape[0]

cols = min(N, 4)
rows = math.ceil(N / cols)

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

# Make axes always iterable
if N == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for i in range(N):
    img = x[i].permute(1, 2, 0)  # (H, W, 3)
    axes[i].imshow(img)
    axes[i].axis("off")

# Hide extra axes
for i in range(N, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.show()