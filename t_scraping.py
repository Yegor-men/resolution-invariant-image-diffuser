import matplotlib.pyplot as plt
import torch
from torch import nn
# ======================================================================================================================
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 100
NUM_CLASSES = 10


def one_hot_encode(label):
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


image_size = 64


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

train_dloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================================================================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cuda is available: {torch.cuda.is_available()}")

from modules.r2id import RIID
from modules.dummy_textencoder import DummyTextCond

model = RIID(
    c_channels=1,
    d_channels=256,
    rescale_factor=8,
    enc_blocks=8,
    dec_blocks=8,
    num_heads=8,
    pos_high_freq=2,
    pos_low_freq=3,
    time_high_freq=7,
    time_low_freq=3,
    film_dim=256,
    axial_dropout=0.1,
    cross_dropout=0.1,
    ffn_dropout=0.2,
    share_weights=False,
)

text_encoder = DummyTextCond(
    token_sequence_length=1,
    d_channels=model.d_channels
)

from save_load_model import load_checkpoint_into

model = load_checkpoint_into(model, "models/E60_0.01078_diffusion_20251224_175855.pt", "cuda")
text_encoder = load_checkpoint_into(text_encoder, "models/E60_0.01078_text_embedding_20251224_175855.pt")

model.to(device)
model.eval()

text_encoder.to(device)
text_encoder.eval()

# ======================================================================================================================
from modules.alpha_bar import alpha_bar_cosine
from modules.corrupt_image import corrupt_image
from tqdm import tqdm
import numpy as np

max_num = 500

losses = []

with torch.no_grad():
    t_range = torch.linspace(0, 1, steps=500)
    t_scrape_null_losses = []
    t_scrape_pos_losses = []

    for t in tqdm(t_range, total=max_num, desc="Scraping"):
        image, label = next(iter(train_dloader))
        b, c, h, w = image.shape
        image, label = image.to(device), label.to(device)
        image = image * 2.0 - 1.0

        alpha_bar = alpha_bar_cosine(torch.ones(b) * t).to(device)
        noisy_image, eps = corrupt_image(image, alpha_bar)
        noisy_image, eps = noisy_image.to(device), eps.to(device)
        pos_cond = text_encoder(label).to(device)
        null_cond = text_encoder(torch.zeros_like(label)).to(device)
        cond_list = [pos_cond, null_cond]

        predicted_eps_list = model(noisy_image, alpha_bar, cond_list)
        eps_pos, eps_null = predicted_eps_list[0], predicted_eps_list[1]

        null_loss = nn.functional.mse_loss(eps_null, eps)
        pos_loss = nn.functional.mse_loss(eps_pos, eps)

        t_scrape_null_losses.append(null_loss.item())
        t_scrape_pos_losses.append(pos_loss.item())

    x = np.linspace(0, 1, len(t_scrape_null_losses))
    plt.plot(x, t_scrape_null_losses, label="Null")
    plt.plot(x, t_scrape_pos_losses, label="Pos")
    percentiles = [1, 25, 50, 75, 99]
    indices = [int(p / 100 * (len(t_scrape_null_losses) - 1)) for p in percentiles]
    percentile_x = [x[i] for i in indices]
    percentile_y = [t_scrape_null_losses[i] for i in indices]
    for px, py, p in zip(percentile_x, percentile_y, percentiles):
        plt.scatter(px, py, color='red')
        plt.text(px, py, f'{py}', fontsize=9, ha='center', va='bottom')
    plt.title('T scrape Losses')
    plt.legend()
    plt.show()
