import torch
from modules.alpha_bar import alpha_bar_cosine
from modules.render_image import render_image
from modules.run_ddim_visualization import run_ddim_visualization

device = "cuda" if torch.cuda.is_available() else "cpu"

from modules.r2id import RIID
from modules.dummy_textencoder import DummyTextCond

model = RIID(
    c_channels=1,
    d_channels=128,
    enc_blocks=8,
    dec_blocks=4,
    num_heads=8,
    pos_high_freq=8,
    pos_low_freq=3,
    time_high_freq=7,
    time_low_freq=3,
    film_dim=128,
    cloud_dropout=0.1,
    cross_dropout=0.1,
    ffn_dropout=0.2,
)

text_encoder = DummyTextCond(
    token_sequence_length=2,
    d_channels=model.d_channels
)

from save_load_model import load_checkpoint_into

model = load_checkpoint_into(model, "models/E20_0.05264_diffusion_20260224_070413.pt", "cuda")
text_encoder = load_checkpoint_into(text_encoder, "models/E20_0.05264_text_embedding_20260224_070413.pt")

model.to(device)
model.eval()

text_encoder.to(device)
text_encoder.eval()

with torch.no_grad():
    positive_label = torch.zeros(100, 10).to(device)
    for i in range(10):
        positive_label[i * 10:(i + 1) * 10, i] = 1.0

    pos_text_cond = text_encoder(positive_label)
    null_text_cond = text_encoder(torch.zeros_like(positive_label))

    sizes = [
        # (28, 28, 1, "Base @ 1"),
        # (28, 28, 2, "Base @ 2"),
        # (28, 28, 4, "Base @ 4"),
        # (28, 28, 8, "Base @ 8"),
        # (28, 28, 16, "Base @ 16"),
        # (32, 32, 1, "1:1 @ 1"),
        # (32, 32, 2, "1:1 @ 2"),
        # (32, 32, 4, "1:1 @ 4"),
        # (32, 32, 8, "1:1 @ 8"),
        # (32, 32, 16, "1:1 @ 16"),
        # (24, 40, "foo"),
        # (40, 24, "bar"),
        # (18, 27, "3:2"),
        # (27, 18, "2:3"),
        # (32, 24, "3:4"),
        # (24, 32, "4:3"),
        # (24, 30, "5:4"),
        # (30, 24, "4:5"),
        # (18, 32, "16:9"),
        # (32, 18, "9:16"),
        # (64, 64, 1, "Double @ 1"),
        # (64, 64, 2, "Double @ 2"),
        # (64, 64, 4, "Double @ 4"),
        # (64, 64, 8, "Double @ 8"),
        # (64, 64, 16, "Double @ 16"),
    ]

    for (height, width, num_clouds, name) in sizes:
        grid_noise = torch.randn(100, 1, height, width).to(device)

        final_x0_hat, final_x = run_ddim_visualization(
            model=model,
            num_clouds=num_clouds,
            initial_noise=grid_noise,
            pos_text_cond=pos_text_cond,
            null_text_cond=null_text_cond,
            alpha_bar_fn=alpha_bar_cosine,
            render_image_fn=render_image,
            num_steps=100,
            cfg_scale=4.0,  # change to 1.0 for sdxl
            eta=2.0,  # change to 1.0 for sdxl
            render_every=1000,
            device=torch.device("cuda"),
            title=f"{name} - H:{height}, W:{width}"
        )
