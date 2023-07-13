# Reimplementation of Scalable Diffusion Models with Transformers
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

class AsymPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=(224,224),
            patch_size=(16,16),
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Embedding layers for timestep and condioning vector

class TimestepEmbedder(nn.Module):
    """
    Transforms scalar time steps into vector representations.
    """
    def __init__(self, vec_size, freq_vector_size=256):
        super().__init__()
        self.freq_vector_size = freq_vector_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_vector_size, vec_size, bias=True),
            nn.SiLU(),
            nn.Linear(vec_size, vec_size, bias=True),
        )

    @staticmethod
    def sin_cos_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        time_freq = self.sin_cos_embedding(t, self.freq_vector_size)
        time_embedding = self.mlp(time_freq)
        return time_embedding

# Implementation of DiT Model

class VisionTransformerBlock(nn.Module):
    """
    A Vision Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, feature_dim, attention_heads, mlp_factor=4.0, **additional_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(feature_dim, num_heads=attention_heads, qkv_bias=True, **additional_kwargs)
        self.norm2 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_size = int(feature_dim * mlp_factor)
        gelu_approximation = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=feature_dim, hidden_features=mlp_hidden_size, act_layer=gelu_approximation, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 6 * feature_dim, bias=True)
        )

    def forward(self, input_tensor, conditioning_tensor):
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning_tensor).chunk(6, dim=1)
        input_tensor = input_tensor + gate_attention.unsqueeze(1) * self.attn(modulate(self.norm1(input_tensor), shift_attention, scale_attention))
        input_tensor = input_tensor + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(input_tensor), shift_mlp, scale_mlp))
        return input_tensor



class VisionTransformerFinalLayer(nn.Module):
    """
    The final layer of Vision Transformer.
    """
    def __init__(self, feature_dim, image_patch_size, output_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(feature_dim, image_patch_size[0] * image_patch_size[1] * output_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 2 * feature_dim, bias=True)
        )

    def forward(self, input_tensor, conditioning_tensor):
        shift_value, scale_value = self.adaLN_modulation(conditioning_tensor).chunk(2, dim=1)
        input_tensor = modulate(self.norm_final(input_tensor), shift_value, scale_value)
        input_tensor = self.linear(input_tensor)
        return input_tensor



class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_dim=(32,32),
        patch_dim=(2,2),
        input_channels=4,
        hidden_dim=152,
        layers=6,
        attention_heads=16,
        mlp_ratio=4.0,
        cond_dropout_prob=0.1,
        sigma_learning=True,
    ):
        super().__init__()

        if isinstance(input_dim, int):
            input_dim = (input_dim,input_dim)

        if isinstance(patch_dim, int):
            patch_dim = (patch_dim,patch_dim)

        self.input_dim = input_dim
        self.sigma_learning = sigma_learning
        self.input_channels = input_channels
        self.output_channels = input_channels * 2 if sigma_learning else input_channels
        self.image_patch_dim = patch_dim
        self.attention_heads = attention_heads
        self.cond_dropout_prob = cond_dropout_prob

        self.x_embedder = AsymPatchEmbed(input_dim, patch_dim, input_channels, hidden_dim, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        num_patches = self.x_embedder.num_patches

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            VisionTransformerBlock(hidden_dim, attention_heads, mlp_factor=mlp_ratio) for _ in range(layers)
        ])
        self.final_layer = VisionTransformerFinalLayer(hidden_dim, patch_dim, self.output_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = generate_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        """
        c = self.output_channels
        patch_dim1, patch_dim2 = self.x_embedder.patch_size
        height, width = self.x_embedder.grid_size

        x = x.reshape(shape=(x.shape[0], height, width, patch_dim1, patch_dim2, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        unpatchified_xs = x.reshape(shape=(x.shape[0], c, height * patch_dim1, width * patch_dim2))
        return unpatchified_xs

    def forward(self, x, t, cond):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        cond: (N,C') tensor of conditioning vectors
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)

        # Apply conditioning dropout, to use in combination with classifier-free guidance:
        if self.cond_dropout_prob > 0:
            mask = torch.rand(cond.shape[0])<(1-self.cond_dropout_prob)
            mask = mask.unsqueeze(1).to(cond.device)
            cond = cond * mask
        c = t + cond                             # (N, D)

        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x

    def forward_with_cfg(self, x, t, cond, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, cond)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# Sine/Cosine Positional Embedding Functions
# Reference: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Reimplemented by: guillefix

def generate_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = generate_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def generate_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = generate_sincos_pos_embed(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = generate_sincos_pos_embed(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def generate_sincos_pos_embed(embed_dim, positions):
    assert embed_dim % 2 == 0

    positions = np.array(positions).reshape(-1, 1)
    omegas = 1. / 10000**(np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.))

    embeds = positions * omegas  # Broadcasting happens here

    emb_sin = np.sin(embeds)
    emb_cos = np.cos(embeds)

    embedding = np.concatenate((emb_sin, emb_cos), axis=1)

    return embedding
