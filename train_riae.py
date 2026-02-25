import torch
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from save_load_model import save_checkpoint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def one_hot_encode(label):
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


image_size = 32


class OneHotMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(
            root='data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True
                ),
                transforms.ToTensor(),  # Converts to [C, H, W] in [0.0, 1.0]
            ])
        )

    def __getitem__(self, index):
        image, label = self.dataset[index]
        one_hot_label = one_hot_encode(label)
        return image, one_hot_label

    def __len__(self):
        return len(self.dataset)


train_dataset = OneHotMNIST(train=True)
test_dataset = OneHotMNIST(train=False)
num_epochs = 20
batch_size = 40
ema_decay = 0.9995
train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

from modules.r2id import RIAE
from modules.render_image import render_image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cuda is available: {torch.cuda.is_available()}")

model = RIAE(
    col_channels=1,
    lat_channels=16,
    embed_dim=128,
    reduction=4,
    pos_high_freq=8,
    pos_low_freq=3,
    num_heads=8,
    dropout=0.1,
).to(device)

import copy

ema_model = copy.deepcopy(model)
ema_model.eval()
for param in ema_model.parameters():
    param.requires_grad = False


@torch.no_grad()
def update_ema_model(model, ema_model, decay):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


import math
from torch.optim.lr_scheduler import LambdaLR


def make_cosine_with_warmup(optimizer, warmup_steps, total_steps, lr_end):
    peak_lr = float(optimizer.defaults['lr'])

    lr_end = float(lr_end)
    min_mult = lr_end / peak_lr

    def lr_lambda(step):
        step = float(step)
        if step <= 0:
            return max(min_mult, 0.0)
        if step < warmup_steps:
            return (step / float(max(1.0, warmup_steps)))
        # after warmup: cosine decay from 1.0 -> min_mult
        progress = (step - warmup_steps) / float(max(1.0, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # map cosine in [0,1] to multiplier in [min_mult, 1.0]
        return min_mult + (1.0 - min_mult) * cosine

    return LambdaLR(optimizer, lr_lambda, -1)


peak_lr = 1e-5
final_lr = 1e-6
total_steps = num_epochs * len(train_dloader)
warmup_steps = len(train_dloader)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = make_cosine_with_warmup(optimizer, warmup_steps, total_steps, final_lr)


def invert_image(image):
    return (image - 0.5) * 2.0


def uninvert_image(image):
    return (image / 2.0) + 0.5


from tqdm import tqdm

train_loss_sums = []
test_loss_sums = []
enc_frac = 0.5
train_losses = []

for e in range(num_epochs):
    model.train()
    train_loss_sum = 0.0
    for i, (image, label) in tqdm(enumerate(train_dloader), total=len(train_dloader), desc=f"TRAIN - E{e}"):
        image, label = invert_image(image).to(device), label.to(device)

        lat_img = model.encode(image, fraction=enc_frac)
        recon_img = model.decode(lat_img)

        loss = torch.nn.functional.mse_loss(recon_img, image)
        loss.backward()
        train_losses.append(loss.item())
        train_loss_sum += loss.item()
        optimizer.step()
        model.zero_grad()

        update_ema_model(model, ema_model, ema_decay)
    train_loss_sum /= len(train_dloader)
    train_loss_sums.append(train_loss_sum)

    plt.title("Loss")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.show()

    model.eval()
    ema_model.eval()
    test_loss_sum = 0.0
    for i, (image, label) in tqdm(enumerate(test_dloader), total=len(test_dloader), desc=f"TEST - E{e}"):
        with torch.no_grad():
            image, label = invert_image(image).to(device), label.to(device)
            lat_img = ema_model.encode(image, fraction=enc_frac)
            recon_img = ema_model.decode(lat_img)
            loss = torch.nn.functional.mse_loss(recon_img, image)
            test_loss_sum += loss.item()
            if i == 0:
                render_image(uninvert_image(image))
                render_image(uninvert_image(recon_img), f"LOSS: {loss}")
    test_loss_sum /= len(test_dloader)
    test_loss_sums.append(test_loss_sum)

    print(f"TRAIN: {train_loss_sum:.5f} | TEST: {test_loss_sum:.5f}")
    plt.title("Loss")
    plt.plot(train_loss_sums, label="train")
    plt.plot(test_loss_sums, label="test")
    plt.legend()
    plt.show()
