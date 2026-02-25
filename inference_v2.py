import matplotlib.pyplot as plt
import torch
from modules.alpha_bar import alpha_bar_cosine
from modules.render_image import render_image
from modules.run_ddim_visualization import run_ddim_visualization

device = "cuda" if torch.cuda.is_available() else "cpu"

from modules.r2id import RIID, RIAE
from modules.dummy_textencoder import DummyTextCond

riae = RIAE(
    col_channels=1,
    lat_channels=64,
    embed_dim=256,
    reduction=4,
    pos_high_freq=8,
    pos_low_freq=3,
    num_heads=8,
    dropout=0.1,
).to(device)
riae.print_model_summary()

r2id = RIID(
    c_channels=riae.lat_channels,
    d_channels=256,
    enc_blocks=8,
    dec_blocks=8,
    num_heads=8,
    pos_high_freq=8,
    pos_low_freq=3,
    time_high_freq=7,
    time_low_freq=3,
    film_dim=128,
    cloud_dropout=0.1,
    cross_dropout=0.1,
    ffn_dropout=0.2,
).to(device)
r2id.print_model_summary()
train_num_clouds = 1

text_encoder = DummyTextCond(
    token_sequence_length=2,
    d_channels=r2id.d_channels
).to(device)

from save_load_model import load_checkpoint_into

riae = load_checkpoint_into(riae, "models/_E20_0.07765_autoencoder_20260226_071710.pt", "cuda")
text_encoder = load_checkpoint_into(text_encoder, "models/E40_0.27112_text_embedding_20260226_095918.pt")
r2id = load_checkpoint_into(r2id, "models/E40_0.27112_diffusion_20260226_095918.pt", "cuda")

riae.eval()
r2id.eval()
text_encoder.eval()


def invert_image(image):
    return (image - 0.5) * 2.0


def uninvert_image(image):
    return (image / 2.0) + 0.5


with torch.no_grad():
    # positive_label = torch.zeros(100, 10).to(device)
    # for i in range(10):
    #     positive_label[i * 10:(i + 1) * 10, i] = 1.0

    positive_label = torch.eye(10).to(device)

    pos_text_cond = text_encoder(positive_label)
    null_text_cond = text_encoder(torch.zeros_like(positive_label))

    sizes = [
        (16, 16, 1, "16px"),
        (28, 28, 1, "28px"),
        (32, 32, 1, "32px"),
        (24, 40, 1, "foo"),
        (40, 24, 1, "bar"),
        (18, 27, 1, "3_2"),
        (27, 18, 1, "2_3"),
        (32, 24, 1, "3_4"),
        (24, 32, 1, "4_3"),
        (24, 30, 1, "5_4"),
        (30, 24, 1, "4_5"),
        (18, 32, 1, "16_9"),
        (32, 18, 1, "9_16"),
        # (64, 64, 1, "32px"),  # Need to solve vram
        # (1024, 1024, 16, "1024px"),  # need to solve vram
    ]

    for (height, width, num_clouds, name) in sizes:
        height, width = height // riae.reduction, width // riae.reduction
        grid_noise = torch.randn(10, riae.lat_channels, height, width).to(device)
        # grid_noise = torch.randn(100, riae.lat_channels, height, width).to(device)

        final_x0_hat, final_x = run_ddim_visualization(
            model=r2id,
            num_clouds=num_clouds,
            initial_noise=grid_noise,
            pos_text_cond=pos_text_cond,
            null_text_cond=null_text_cond,
            alpha_bar_fn=alpha_bar_cosine,
            num_steps=100,
            cfg_scale=4.0,
            eta=2.0,
            device=torch.device("cuda"),
        )

        diffused_image = uninvert_image(riae.decode(final_x, fraction=1.0))
        render_image(diffused_image, f"{name} - Latent H:{height}, W:{width}", name, True)
