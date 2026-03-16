"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNorm_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
    LoraLayer,
    LoraConfig
)


# BUG-1 FIX: use text_num_embeds instead of hardcoded sizes
# BUG-12 FIX: removed dead TextEmbedding_orig class

class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=0, conv_mult=2, tuning_config=None, ko=True):
        super().__init__()
        self.tuning_config = tuning_config
        self.mask_padding = mask_padding

        # BUG-1 FIX: use text_num_embeds (actual vocab size) instead of hardcoded 2546/158
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.text_embed_ko = nn.Embedding(text_num_embeds + 1, text_dim)

        # BUG-35: warn if embedding size differs from pretrained F5-TTS (2546)
        _pretrained_vocab = 2545
        if text_num_embeds != _pretrained_vocab:
            import warnings
            warnings.warn(
                f"TextEmbedding: text_num_embeds={text_num_embeds} differs from pretrained F5-TTS ({_pretrained_vocab}). "
                f"Pretrained embedding weights will be skipped during load_state_dict(strict=False)."
            )

        self.alpha = 1 if ko else 0

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)

            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult, tuning_config=tuning_config) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:
            text = torch.zeros_like(text)

        # Only call the embedding that is actually used (prevents index-out-of-range)
        if self.alpha == 0:
            text = self.text_embed(text)
        elif self.alpha == 1:
            text = self.text_embed_ko(text)
        else:
            text = (1 - self.alpha) * self.text_embed(text) + self.alpha * self.text_embed_ko(text)

        if self.extra_modeling:
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        return text


# BUG-8 FIX: scale now used in forward
class BottleneckAdapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, output_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.scale = 0.01

    def forward(self, x):
        adapter_output = self.up_proj(self.act(self.down_proj(x)))
        return self.layer_norm(x + self.scale * adapter_output)


## Drop Path
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, tuning_config=None):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

        self.tuning_config = tuning_config

        if self.tuning_config:
            self.lora_layers = nn.ModuleDict()
            self.lora_r = tuning_config.r
            self.lora_alpha = tuning_config.lora_alpha
            self.dropout = tuning_config.lora_dropout
            self.target_modules = set(tuning_config.target_modules)
            self.drop_path = DropPath(drop_prob=tuning_config.drop_path)

            if "proj" in self.target_modules:
                self.lora_layers["proj"] = LoraLayer(mel_dim * 2 + text_dim, out_dim, r=self.lora_r, alpha=self.lora_alpha, drop_out=self.dropout)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:
            cond = torch.zeros_like(cond)

        concat = torch.cat((x, cond, text_embed), dim=-1)
        x = self.proj(concat)

        if self.tuning_config and "proj" in self.target_modules:
            lora_out = self.lora_layers["proj"](concat)
            lora_out = self.drop_path(lora_out)  # BUG-6 FIX: DropPath handles prob=0 internally
            x = x + lora_out

        x = self.conv_pos_embed(x) + x
        return x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        long_skip_connection=False,
        checkpoint_activations=False,
        conditioning_adapter_config=None,
        prompt_adapter_config=None,
        dit_lora_adapter_config=None,
        ko=True,
    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim

        self.text_embed = TextEmbedding(text_num_embeds, text_dim, mask_padding=text_mask_padding, conv_layers=conv_layers, tuning_config=conditioning_adapter_config, ko=ko)
        self.text_cond, self.text_uncond = None, None
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim, tuning_config=prompt_adapter_config)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult,
                    dropout=dropout, qk_norm=qk_norm, pe_attn_head=pe_attn_head,
                    tuning_config=dit_lora_adapter_config,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.checkpoint_activations = checkpoint_activations
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)
        return ckpt_forward

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: float["b n d"],  # noqa: F722
        cond: float["b n d"],  # noqa: F722
        text: int["b nt"],  # noqa: F722
        time: float["b"] | float[""],  # noqa: F821 F722
        drop_audio_cond,
        drop_text,
        mask: bool["b n"] | None = None,  # noqa: F722
        cache=False,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)

        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, seq_len, drop_text=True)
                text_embed = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, seq_len, drop_text=False)
                text_embed = self.text_cond
        else:
            text_embed = self.text_embed(text, seq_len, drop_text=drop_text)

        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output
