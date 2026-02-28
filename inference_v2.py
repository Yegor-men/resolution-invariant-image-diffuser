import matplotlib.pyplot as plt
import torch
from modules.alpha_bar import alpha_bar_cosine
from modules.render_image import render_image
from modules.run_ddim_visualization import run_ddim_visualization

device = "cuda" if torch.cuda.is_available() else "cpu"

from modules.r2id import R2ID, R2IR
from modules.dummy_textencoder import DummyTextCond

r2ir = R2IR(
    col_channels=3,
    lat_channels=768,
    embed_dim=1024,
    pos_high_freq=16,
    pos_low_freq=16,
    enc_blocks=1,
    dec_blocks=1,
    num_heads=16,
    mha_dropout=0.1,
    ffn_dropout=0.2,
).to(device)

r2id = R2ID(
    c_channels=r2ir.lat_channels,
    d_channels=1024,
    enc_blocks=8,
    dec_blocks=8,
    num_heads=16,
    pos_high_freq=16,
    pos_low_freq=16,
    time_high_freq=7,
    time_low_freq=3,
    film_dim=128,
    cloud_dropout=0.1,
    cross_dropout=0.1,
    ffn_dropout=0.2,
).to(device)

text_encoder = DummyTextCond(
    token_sequence_length=2,
    d_channels=r2id.d_channels
).to(device)

from save_load_model import load_checkpoint_into

r2ir = load_checkpoint_into(r2ir, "models/E40_0.01115_autoencoder_20260228_185931.pt", "cuda")
text_encoder = load_checkpoint_into(text_encoder, "models/E17_0.02337_text_embedding_20260228_201648.pt")
r2id = load_checkpoint_into(r2id, "models/E17_0.02337_diffusion_20260228_201647.pt", "cuda")

r2ir.eval()
r2id.eval()
text_encoder.eval()


def invert_image(image):
    return (image - 0.5) * 2.0


def uninvert_image(image):
    return (image / 2.0) + 0.5


with torch.no_grad():
    positive_label = torch.zeros(100, 10).to(device)
    for i in range(10):
        positive_label[i * 10:(i + 1) * 10, i] = 1.0

    # positive_label = torch.eye(10).to(device)

    pos_text_cond = text_encoder(positive_label)
    null_text_cond = text_encoder(torch.zeros_like(positive_label))

    lat_h, lat_w = 8, 8
    grid_noise = torch.randn(100, r2ir.lat_channels, lat_h, lat_w).to(device)

    sizes = [
        (10, 10, "10px"),
        (16, 16, "16px"),
        (28, 28, "28px"),
        (32, 32, "32px"),
        (24, 40, "foo"),
        (40, 24, "bar"),
        (18, 27, "3_2"),
        (27, 18, "2_3"),
        (32, 24, "3_4"),
        (24, 32, "4_3"),
        (24, 30, "5_4"),
        (30, 24, "4_5"),
        (18, 32, "16_9"),
        (32, 18, "9_16"),
        # (64, 64, "64px"), vram issue with 100 batch size
        # (1024, 1024, "1024px"),  # need to solve vram,  batch size 100 is too heavy, but then the model also weighs 270m lmao
    ]

    final_x0_hat, final_x = run_ddim_visualization(
        model=r2id,
        num_clouds=1,
        initial_noise=grid_noise,
        pos_text_cond=pos_text_cond,
        null_text_cond=null_text_cond,
        alpha_bar_fn=alpha_bar_cosine,
        num_steps=100,
        cfg_scale=4.0,
        eta=2.0,
        device=torch.device("cuda"),
    )

    for (height, width, name) in sizes:
        diffused_image = uninvert_image(r2ir.decode(final_x, height=height, width=width))
        render_image(diffused_image, f"{name} - Latent H:{lat_h}, W:{lat_w} - Image H:{height}, W:{width}", name, True)
