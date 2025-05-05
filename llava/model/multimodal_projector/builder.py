import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "subobject_tokenization":
        # implementation from https://github.com/ChenDelong1999/subobjects-VLM/blob/main
        config.token_roi_resolution = 32  # token roi resolution from https://github.com/ChenDelong1999/subobjects-VLM/blob/main/configs/visual_embedding/clip_vit_l_14_336.json
        feature_channels = config.mm_hidden_size
        mlp_expansion = 4
        projector = build_subobject_vision_projector(config)
        return projector

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")


def build_subobject_vision_projector(config, delay_load=False, **kwargs):
    # implementation from https://github.com/ChenDelong1999/subobjects-VLM/blob/main
    config.token_roi_resolution = 32  # token roi resolution from https://github.com/ChenDelong1999/subobjects-VLM/blob/main/configs/visual_embedding/clip_vit_l_14_336.json
    feature_channels = config.mm_hidden_size
    mlp_expansion = 4
    projector = nn.Sequential(
        nn.Linear(
            4 + feature_channels + config.token_roi_resolution**2,
            config.hidden_size,
            bias=config.mlp_bias,
        ),
        nn.ReLU(),
        nn.Linear(
            config.hidden_size,
            config.hidden_size * mlp_expansion,
            bias=config.mlp_bias,
        ),
        nn.ReLU(),
        nn.Linear(
            config.hidden_size * mlp_expansion,
            config.hidden_size,
            bias=config.mlp_bias,
        ),
    )
    return projector
