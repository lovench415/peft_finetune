"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb


# raw wav to mel spec


mel_basis_cache = {}
hann_window_cache = {}


def get_bigvgan_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
    fmin=0,
    fmax=None,
    center=False,
):  # Copy from https://github.com/NVIDIA/BigVGAN/tree/main
    device = waveform.device
    key = f"{n_fft}_{n_mel_channels}_{target_sample_rate}_{hop_length}_{win_length}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(sr=target_sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=fmin, fmax=fmax)
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)  # TODO: why they need .float()?
        hann_window_cache[key] = torch.hann_window(win_length).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_length) // 2
    waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec


def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    ):
        super().__init__()
        assert mel_spec_type in ["vocos", "bigvgan"], print("We only support two extract mel backend: vocos or bigvgan")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        if mel_spec_type == "vocos":
            self.extractor = get_vocos_mel_spectrogram
        elif mel_spec_type == "bigvgan":
            self.extractor = get_bigvgan_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # 256

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# convolutional position embedding
class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        ) 
            
    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):  # noqa: F722
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        conv1d_out = self.conv1d(x)
        
        out = conv1d_out.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out

# rotary positional embedding related


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# Global Response Normalization layer (Instance Normalization ?)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108
from sqlite3 import adapters
from typing import Callable, List, Optional
import torch
from torch import Tensor
import torch.nn as nn

class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.
    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        out_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1 = torch.nn.Conv1d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv1d(squeeze_channels, out_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale


class ConvAdapter(nn.Module):
    def __init__(self, inplanes, outplanes, width, 
                kernel_size=3, padding=1, stride=1, groups=1, dilation=1, norm_layer=None, act_layer=None, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        if act_layer is None:
            act_layer = nn.Identity

        # self.act = nn.SiLU()
        print("Entering ConvAdapter")
        print(f"inplanes : {inplanes}\t width : {width}\t kernel_size : {kernel_size}\tgroups:{groups}")
        # depth-wise conv
        self.conv1 = nn.Conv1d(inplanes, width, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding, dilation=int(dilation))
        # self.norm = norm_layer(width)
        self.act = act_layer()

        # poise-wise conv
        self.conv2 = nn.Conv1d(width, outplanes, kernel_size=1, stride=1)

        # se 
        # self.se = SqueezeExcitation(inplanes, width, outplanes, activation=act_layer)
        self.se = nn.Parameter(1.0 * torch.ones((1, outplanes,1)), requires_grad=True) ## alpha (initialized with 1)
        # self.se = 4.0

    def forward(self, x):
        out = self.conv1(x)
        # out = self.norm(out)
        out = self.act(out)
        out = self.conv2(out)
        out = out * self.se
        # TODO: add norm layer

        return out

class ConvAdapterConfig:
    """
    Configuration class for ConvAdapter
    """
    def __init__(self, method, adapt_size, kernel_size, adapt_scale=1.0, init_scale=1.0):
        self.method = method
        self.adapt_size = adapt_size
        self.kernel_size = kernel_size

        self.adapt_scale = adapt_scale
        self.init_scale = init_scale


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        tuning_config=None
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2 # 3

        # Depthwise convolution
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )

        # Layer normalization and point-wise convolutions
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # Pointwise 1x1 convolution
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.tuning_config = tuning_config

        if self.tuning_config and 'conv_adapt' in self.tuning_config.method:
            self.conv_adapter = ConvAdapter(
                dim, dim,
                kernel_size = int(self.tuning_config.kernel_size),
                padding= int(dilation * (self.tuning_config.kernel_size - 1) // 2),
                width= int(dim // self.tuning_config.adapt_size),
                stride= 1,
                groups= int(dim // self.tuning_config.adapt_size) if self.tuning_config.adapt_size >=1 else int(dim), # incase 'compression factor' is small than 1 
                dilation= 1,
                act_layer= nn.GELU
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Transpose for depthwise convolution
        x = x.transpose(1, 2)  # (b, n, d) -> (b, d, n)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (b, d, n) -> (b, n, d)
  
        #if self.tuning_config and 'conv_adapt' in self.tuning_config['method']:
        if self.tuning_config and 'conv_adapt' in self.tuning_config.method:
            x_adapt = self.conv_adapter(residual.transpose(1, 2)).transpose(1, 2)
            x = x + x_adapt
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        return residual + x


# RMSNorm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x):
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight

        return x

    


# AdaLayerNorm
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    

class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
    

# FeedForward

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


        

class LoraConfig:
    """
    Configuration class for LoRA (Low-Rank Adaptation).
    """
    def __init__(self, r, lora_alpha, target_modules, lora_dropout=0.05, bias="none",scale=1.0, drop_path = 0.0, lora_adapter_name= "lora", use_sparse=False, very_sparse=False):
        self.r = r  # Rank of LoRA
        self.lora_alpha = lora_alpha  # Scaling factor
        self.target_modules = target_modules  # Target layers to apply LoRA
        self.lora_dropout = lora_dropout  # Dropout rate for LoRA layers
        self.bias = bias  # Whether to use bias in LoRA layers
        self.scaling = scale
        self.drop_path = drop_path
        self.lora_adapter_name = lora_adapter_name # "lora", "randlora"
        self.use_sparse = use_sparse
        self.very_sparse = very_sparse
        self.drop_out = lora_dropout


# Modified from peft/tuners/randlora/layers.py

class UniqueBaseGrad(torch.autograd.Function):
    @staticmethod    
    def forward(ctx, randbasis_A, randlora_lambda, randlora_gamma):
        out = randlora_lambda[:, :, None] * randbasis_A * randlora_gamma[None, :]
        ctx.save_for_backward(randbasis_A, randlora_lambda, randlora_gamma)
        return out 
    
    @staticmethod
    def backward(ctx, grad_output):
        randbasis_A, randlora_lambda, randlora_gamma = ctx.saved_tensors
        randbasis_A = randbasis_A.to(grad_output.dtype)
        randlora_lambda = randlora_lambda.to(grad_output.dtype)
        randlora_gamma = randlora_gamma.to(grad_output.dtype)

        grad_lambda = torch.einsum('kbj,kvj,bj->kb', grad_output, randbasis_A, randlora_gamma)
        grad_gamma = torch.einsum('kbj,kvj,kb->bj', grad_output, randbasis_A, randlora_lambda)
        return None, grad_lambda, grad_gamma


class RandLoraLayer(nn.Module):
    def __init__(self, in_features, out_features, r, dropout=0.05, use_sparse=False, very_sparse=False, seed=42, scale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.dropout = nn.Dropout(p=dropout)
        self.n = math.ceil(min(in_features, out_features) / r)  # full-rank 보장

        torch.manual_seed(seed)
        self._init_randbasis(use_sparse=use_sparse, very_sparse=very_sparse)

        # Trainable parameters (scaling diagonal matrix)
        self.Lambda = nn.Parameter(torch.zeros(r, self.n))
        self.Gamma = nn.Parameter(torch.ones(self.n, min(in_features, out_features)) / max(in_features, out_features))

        # scaling factor
        self.scale = scale if scale is not None else (1.0 / (r * math.sqrt(self.n)))

        self.reset_parameters()

    def _init_randbasis(self, use_sparse=False, very_sparse=False):
        s = math.sqrt(min(self.out_features, self.in_features)) if very_sparse else 3

        rand_A = torch.rand(self.r, self.n, self.in_features)
        rand_B = torch.rand(self.out_features, self.n, self.r)

        if use_sparse or very_sparse:
            A_sparse = torch.zeros_like(rand_A)
            B_sparse = torch.zeros_like(rand_B)
            A_sparse[rand_A < 1/(2*s)] = -1
            A_sparse[rand_A > 1 - 1/(2*s)] = 1
            B_sparse[rand_B < 1/(2*s)] = -1
            B_sparse[rand_B > 1 - 1/(2*s)] = 1
            rand_A = A_sparse / A_sparse.std()
            rand_B = B_sparse / B_sparse.std()
        else:
            rand_A = rand_A / rand_A.std()
            rand_B = rand_B / rand_B.std()

        self.register_buffer("randbasis_A", rand_A)
        self.register_buffer("randbasis_B", rand_B)

    def reset_parameters(self):
        nn.init.zeros_(self.Lambda)
        nn.init.constant_(self.Gamma, 1.0 / max(self.Gamma.shape))

    def get_scaled_bases(self):
        min_dim = min(self.in_features, self.out_features)
        max_dim = max(self.in_features, self.out_features)

        sliced_A = self.randbasis_A[:, :, :min_dim]
        sliced_B = self.randbasis_B[:max_dim, :, :]

        # Flatten
        update_A = UniqueBaseGrad.apply(sliced_A, self.Lambda, self.Gamma).flatten(end_dim=1)
        update_B = sliced_B.flatten(start_dim=1)

        if min_dim == self.in_features:
            return update_A, update_B
        else:
            return update_B.T, update_A.T

    def forward(self, x):
        update_A, update_B = self.get_scaled_bases()
        lora_result = F.linear(F.linear(self.dropout(x), update_B), update_A) * self.scaling
        return lora_result
        #return F.linear(F.linear(x, B), A) * self.scale

class LoraLayer(nn.Module):
    """
    LoRA Layer for injecting low-rank adaptations into specific layers.
    """
    def __init__(self, in_features, out_features, r, alpha, drop_out=0.05, init_zero_all=False):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.in_features = in_features
        self.out_features = out_features
        # initialization
        self.init_zero_all = init_zero_all
        
        self.lora_A = nn.Parameter(torch.randn(in_features, r))
        self.lora_B = nn.Parameter(torch.randn(r, out_features))
        self.dropout = nn.Dropout(p=drop_out)
        #self.drop_path = DropPath(drop_prob=self.drop_path_prob)
        self.reset_parameters() # 초기화


    def reset_parameters(self):
        if self.init_zero_all :
            nn.init.zeros_(self.lora_A)
        else:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # drop out
        x = self.dropout(x)
        # Perform the low-rank adaptation transformation
        lora_output = torch.matmul(x, self.lora_A).matmul(self.lora_B)
        lora_output = lora_output * self.scaling 

        return lora_output






class Attention(nn.Module):
    def __init__(
        self,
        processor: JointAttnProcessor | AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
        context_pre_only=None,
        qk_norm: Optional[str] = None,
        tuning_config= None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor
        
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        
        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else :
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")


        if self.context_dim is not None:
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if qk_norm is None:
                self.c_q_norm = None
                self.c_k_norm = None
            elif qk_norm == "rms_norm":
                self.c_q_norm = RMSNorm(dim_head, eps=1e-6)
                self.c_k_norm = RMSNorm(dim_head, eps=1e-6)
        

        self.tuning_config = tuning_config

        if self.tuning_config:
            # DiT LoRA Adapters

            self.lora_layers = nn.ModuleDict()
            self.lora_r = tuning_config.r
            self.lora_alpha = tuning_config.lora_alpha
            self.target_modules = set(tuning_config.target_modules)
            self.use_sparse = tuning_config.use_sparse
            self.very_sparse = tuning_config.very_sparse
            self.lora_adapter_name = tuning_config.lora_adapter_name
            self.dropout = tuning_config.drop_out

            if "to_q" in self.target_modules:
                if self.lora_adapter_name == "lora":
                    self.lora_layers["to_q"] = LoraLayer(dim, self.inner_dim, self.lora_r, self.lora_alpha)
                elif self.lora_adapter_name == "randlora":
                    self.lora_layers["to_q"] = RandLoraLayer(dim, self.inner_dim, self.lora_r, use_sparse=self.use_sparse, very_sparse=self.very_sparse)
            if "to_v" in self.target_modules:
                if self.lora_adapter_name == "lora":
                    self.lora_layers["to_v"] = LoraLayer(dim, self.inner_dim, self.lora_r, self.lora_alpha)
                elif self.lora_adapter_name == "randlora":
                    self.lora_layers["to_v"] = RandLoraLayer(dim, self.inner_dim, self.lora_r, use_sparse=self.use_sparse, very_sparse=self.very_sparse)
            if "to_k" in self.target_modules:
                if self.lora_adapter_name == "lora":
                    self.lora_layers["to_k"] = LoraLayer(dim, self.inner_dim, self.lora_r, self.lora_alpha)
                elif self.lora_adapter_name == "randlora":
                    self.lora_layers["to_k"] = RandLoraLayer(dim, self.inner_dim, self.lora_r, use_sparse=self.use_sparse, very_sparse=self.very_sparse)


        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, dim)

    def forward(
        self,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b n d"] = None,  # context c  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.Tensor:

        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self,  x, mask=mask, rope=rope)


# Attention processor


class AttnProcessor:
    def __init__(
        self,
        pe_attn_head: int | None = None, # number of attention head to apply rope, None for all
    ):
        self.pe_attn_head = pe_attn_head

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections.
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # LoRA 적용
        if attn.tuning_config:
            if "to_q" in attn.lora_layers:
                query += attn.lora_layers["to_q"](x)
            if "to_k" in attn.lora_layers:
                key += attn.lora_layers["to_k"](x)
            if "to_v" in attn.lora_layers:
                value += attn.lora_layers["to_v"](x)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)


        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
            else :
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)


        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = mask
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class JointAttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b nt d"] = None,  # context c, here text # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.FloatTensor:
        residual = x

        batch_size = c.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # `context` projections
        c_query = attn.to_q_c(c)
        c_key = attn.to_k_c(c)
        c_value = attn.to_v_c(c)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_query = c_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_key = c_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_value = c_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)
        if attn.c_q_norm is not None:
            c_query = attn.c_q_norm(c_query)
        if attn.c_k_norm is not None:
            c_key = attn.c_k_norm(c_key)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        # joint attention
        query = torch.cat([query, c_query], dim=2)
        key = torch.cat([key, c_key], dim=2)
        value = torch.cat([value, c_value], dim=2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = F.pad(mask, (0, c.shape[1]), value=True)  # no mask for c (text)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # Split the attention outputs.
        x, c = (
            x[:, : residual.shape[1]],
            x[:, residual.shape[1] :],
        )

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)
        if not attn.context_pre_only:
            c = attn.to_out_c(c)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
            # c = c.masked_fill(~mask, 0.)  # no mask for c (text)

        return x, c



# DiT Block

class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, qk_norm=None, pe_attn_head=None, tuning_config= None):
        super().__init__()

        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(pe_attn_head=pe_attn_head),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
            tuning_config= tuning_config,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x

# MMDiT Block https://arxiv.org/abs/2403.03206


class MMDiTBlock(nn.Module):
    r"""
    modified from diffusers/src/diffusers/models/attention.py

    notes.
    _c: context related. text, cond, etc. (left part in sd3 fig2.b)
    _x: noised input related. (right part)
    context_pre_only: last layer only do prenorm + modulation cuz no more ffn
    """

    def __init__(
        self, dim, heads, dim_head, ff_mult=4, dropout=0.1, context_dim=None, context_pre_only=False, qk_norm=None
    ):
        super().__init__()
        if context_dim is None:
            context_dim = dim
        self.context_pre_only = context_pre_only

        self.attn_norm_c = AdaLayerNorm_Final(context_dim) if context_pre_only else AdaLayerNorm(context_dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=JointAttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=context_dim,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
        )

        if not context_pre_only:
            self.ff_norm_c = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
            self.ff_c = FeedForward(dim=context_dim, mult=ff_mult, dropout=dropout, approximate="tanh")
        else:
            self.ff_norm_c = None
            self.ff_c = None
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, c, t, mask=None, rope=None, c_rope=None):  # x: noised input, c: context, t: time embedding
        # pre-norm & modulation for attention input
        if self.context_pre_only:
            norm_c = self.attn_norm_c(c, t)
        else:
            norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)

        # attention
        x_attn_output, c_attn_output = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope)

        # process attention output for context c
        if self.context_pre_only:
            c = None
        else:  # if not last layer
            c = c + c_gate_msa.unsqueeze(1) * c_attn_output

            norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            c_ff_output = self.ff_c(norm_c)
            c = c + c_gate_mlp.unsqueeze(1) * c_ff_output

        # process attention output for input x
        x = x + x_gate_msa.unsqueeze(1) * x_attn_output

        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x_ff_output = self.ff_x(norm_x)
        x = x + x_gate_mlp.unsqueeze(1) * x_ff_output

        return c, x


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float["b"]):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time