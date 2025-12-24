import torch
from modules.alpha_bar import alpha_bar_cosine
from modules.render_image import render_image
from modules.run_ddim_visualization import run_ddim_visualization

device = "cuda" if torch.cuda.is_available() else "cpu"

from modules.s2id import SIID
from modules.dummy_textencoder import DummyTextCond

model = SIID(
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

import time

with torch.no_grad():
    positive_label = torch.zeros(100, 10).to(device)
    for i in range(10):
        positive_label[i * 10:(i + 1) * 10, i] = 1.0

    multi_pos_text_cond = text_encoder(positive_label)
    multi_null_text_cond = text_encoder(torch.zeros_like(positive_label))

    h, w = 128, 128
    multi_noise = torch.randn(100, 1, model.rescale_factor * h, model.rescale_factor * w).to(device)

    foo = torch.zeros(1, 10).to(device)
    foo[0][0] = 1.0
    single_pos_text_cond = text_encoder(foo)
    single_null_text_cond = text_encoder(torch.zeros_like(foo))
    single_noise = torch.randn(1, 1, model.rescale_factor * h, model.rescale_factor * w).to(device)

    thing = "single"

    start1 = time.time()
    if thing == "multi":
        final_x0_hat, final_x = run_ddim_visualization(
            model=model,
            initial_noise=multi_noise,
            pos_text_cond=multi_pos_text_cond,
            null_text_cond=multi_null_text_cond,
            alpha_bar_fn=alpha_bar_cosine,
            render_image_fn=render_image,
            num_steps=100,
            cfg_scale=5.0,
            eta=2.0,
            render_every=1,
            device=torch.device("cuda")
        )
    elif thing == "single":
        final_x0_hat, final_x = run_ddim_visualization(
            model=model,
            initial_noise=single_noise,
            pos_text_cond=single_pos_text_cond,
            null_text_cond=single_null_text_cond,
            alpha_bar_fn=alpha_bar_cosine,
            render_image_fn=render_image,
            num_steps=100,
            cfg_scale=5.0,
            eta=2.0,
            render_every=1,
            device=torch.device("cuda")
        )
    end = time.time()

    print(f"Time taken: {end - start1:.3f}s")
