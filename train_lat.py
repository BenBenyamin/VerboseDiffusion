import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import yaml

from diffusion import StableDiffusionModel


# load params

params_filepath = os.path.dirname(os.path.abspath(__file__)) + "/" + "params.yml"

with open(params_filepath, "r") as f:
    params = yaml.safe_load(f)

# Define image transforms (normalize to roughly zero mean / unit variance)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Training dataset
train_dataset = datasets.ImageFolder(
    root = "./dataset/afhq/train",
    transform=transform
)

# Val dataset
val_dataset = datasets.ImageFolder(
    root = "./dataset/afhq/val",
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
dfm = StableDiffusionModel(
    device=params["device"],
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
    pretrained_vae_name=params["pretrained_vae_name"],
    scaling_const=params["scaling_const"],
)

label_names = train_dataset.classes

print(label_names)

dfm.load("Stable_100k")

dfm.steps_cnt = 100_000

dfm.train(
    steps = 100_000,
    train_dataloader = train_loader,
    val_dataloader = val_loader,
    image_load_microbatch=params["image_load_microbatch"],
    lr = params["lr"],
    uncond_prob = params["uncond_prob"],
    grad_norm = params["grad_norm"],
    log_every = 5_000,
)

dfm.save("Stable_200k")