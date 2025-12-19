import torch
from modules.alpha_bar import alpha_bar_cosine
from modules.render_image import render_image
from modules.run_ddim_visualization import run_ddim_visualization

device = "cuda" if torch.cuda.is_available() else "cpu"

from modules.s2id import SIID

model = SIID(
    c_channels=1,
    d_channels=320,
    rescale_factor=8,
    enc_blocks=6,
    dec_blocks=6,
    num_heads=5,
    pos_high_freq=2,
    pos_low_freq=3,
    time_high_freq=7,
    time_low_freq=3,
    film_dim=320,
    cross_dropout=0.1,
    axial_dropout=0.1,
    ffn_dropout=0.2,
    share_weights=False,
)

from save_load_model import load_checkpoint_into

model = load_checkpoint_into(model, "models/refinedE30_0.02163_20251219_181019.pt", "cuda")
model.to(device)
model.eval()

import time

with torch.no_grad():
    positive_text_conditioning = torch.zeros(100, 10).to(device)
    for i in range(10):
        positive_text_conditioning[i * 10:(i + 1) * 10, i] = 1.0

    noise = torch.randn(100, 1, 64, 64).to(device)

    start1 = time.time()
    final_x0_hat, final_x = run_ddim_visualization(
        model=model,
        initial_noise=noise,
        positive_text_conditioning=positive_text_conditioning,
        alpha_bar_fn=alpha_bar_cosine,
        render_image_fn=render_image,
        num_steps=50,
        cfg_scale=2.0,
        eta=2.0,
        render_every=1000,
        device=torch.device("cuda")
    )
    end = time.time()
    print(f"Time taken: {end - start1:.3f}s")
