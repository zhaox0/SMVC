import torch
import torch.nn.functional as F
from contextlib import contextmanager
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from typing import Optional          # DurationPredictor 
from utils import init_weights, get_padding
import numpy as np
import random
from stft import TorchSTFT
from transformers import BertModel, BertConfig
from face_encoder import FaceEncoder, Repara
from flow_recoulping import ResidualCouplingBlock
from TVTR1 import TimeVaryingTimbreModule

LRELU_SLOPE = 0.1


@torch.jit.script
def fused_tanh_sigmoid_multiply(input_a, n_channels):
    n_channels_int = n_channels[0]
    t_act = torch.tanh(input_a[:, :n_channels_int, :])
    s_act = torch.sigmoid(input_a[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2).contiguous()

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s.transpose(1,2))
        h = h.transpose(1,2)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups, bias=False)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
      

class LinearNorm_mixln(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm_mixln, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

def normalize_f0(f0, eps=1e-5):
    mean = f0.mean(dim=-1, keepdim=True)
    std = f0.std(dim=-1, keepdim=True)
    f0_norm = (f0 - mean) / (std + eps)
    return f0_norm

class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, hidden_size=256):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self._activated = True
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm_mixln(hidden_size, 2 * hidden_size)

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})'

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x, spk_embed):
        if not self.training or not self._activated:
            return x
        if random.random() > self.p:
            return x
        x = x.transpose(1, 2)
        spk_embed = spk_embed.transpose(1, 2)
        B = x.size(0)
        mu, sig = torch.mean(x, dim=1, keepdim=True), torch.std(x, dim=1, keepdim=True)
        x_normed = (x - mu) / (sig + 1e-6)
        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)
        mu1, sig1 = torch.split(self.affine_layer(spk_embed), self.hidden_size, dim=-1)
        perm = torch.randperm(B)
        mu2, sig2 = mu1[perm], sig1[perm]
        mu_mix = mu1 * lmda + mu2 * (1 - lmda)
        sig_mix = sig1 * lmda + sig2 * (1 - lmda)
        out = sig_mix * x_normed + mu_mix
        out = out.transpose(1, 2)
        return out

class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        self.affine = nn.Linear(in_dim, out_dim)

    def forward(self, input):
        return self.affine(input)


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channel, gin_channels):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channel = in_channel
        self.norm = nn.LayerNorm(in_channel, elementwise_affine=False)
        self.style = AffineLinear(gin_channels, in_channel * 2)
        self.style.affine.bias.data[:in_channel] = 1
        self.style.affine.bias.data[in_channel:] = 0

    def forward(self, x_in, g_l):
        style = self.style(g_l)
        gamma, beta = style.chunk(2, dim=-1)
        x_in = x_in.permute(0, 2, 1)
        out = self.norm(x_in)
        out = gamma * out + beta
        return out


def extract_prosodic_features(mel, F0_model):
    batch_size, n_mels, time_frames = mel.shape
    with torch.no_grad():
        f0_cont, _, _ = F0_model(mel.unsqueeze(1))
        energy_cont = torch.sqrt(torch.mean(mel ** 2, dim=1))
    return f0_cont, energy_cont


class WN(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert (kernel_size % 2 == 1)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        self.conv = nn.Conv1d(512, 256, 1)
        self.style_norms = torch.nn.ModuleList()
        if gin_channels != 0:
            for i in range(n_layers):
                style_norm = StyleAdaptiveLayerNorm(hidden_channels * 2, gin_channels)
                self.style_norms.append(style_norm)
            cond_layer = torch.nn.Conv1d(gin_channels, hidden_channels * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None and hasattr(self, 'style_norms'):
                cond_offset = i * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + self.hidden_channels, :]
                g_l = g_l.permute(0, 2, 1)
                x_in = self.style_norms[i](x_in, g_l)
                x_in = x_in.permute(0, 2, 1)
            acts = fused_tanh_sigmoid_multiply(x_in, n_channels_tensor)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


class Encoder1(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels,
                 kernel_size, dilation_rate, n_layers, gin_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.mixstyle = MixStyle()
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, g=None):
        x_mask = 1
        x = self.pre(x) * x_mask
        # x = self.mixstyle(x, g)
        x = self.enc(x, x_mask, g=g)
        # x = self.mixstyle(x, g)
        x = self.proj(x) * x_mask
        return x


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.alpha1, self.alpha2):
            xt = x + (1 / a1) * (torch.sin(a1 * x) ** 2)
            xt = c1(xt)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock1_old(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class SineGen(torch.nn.Module):
    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0, flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        if not self.flag_for_pulse:
            rad_values = torch.nn.functional.interpolate(rad_values.transpose(1, 2),
                                                         scale_factor=1 / self.upsample_scale,
                                                         mode="linear").transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = torch.nn.functional.interpolate(phase.transpose(1, 2) * self.upsample_scale,
                                                    scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
            sines = torch.sin(phase)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


def padDiff(x):
    return F.pad(F.pad(x, (0, 0, -1, 1), 'constant', 0) - x, (0, 0, 0, -1), 'constant', 0)

class LinearNorm(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, spectral_norm=False):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias)
        if spectral_norm:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, input):
        return self.fc(input)

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, spectral_norm=False):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, input):
        return self.conv(input)

class Conv1dGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = ConvNorm(in_channels, 2 * out_channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x

class MelStyleEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.kernel_size = 5
        self.n_head = 8
        self.dropout = 0.1
        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout)
        )
        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )
        self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim,
                                           self.hidden_dim // self.n_head,
                                           self.hidden_dim // self.n_head, self.dropout)
        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def forward(self, x, mask=None):
        max_len = x.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        x = self.spectral(x)
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        x = self.fc(x)
        return x   # [B, T, 256]

class ASP(nn.Module):
    def __init__(self, in_channels):
        super(ASP, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=in_channels, kernel_size=1),
            nn.Softmax(dim=2)
        )
        self.conv1 = torch.nn.Conv1d(512, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 256, 1)

    def forward(self, x):  # [B, F, T]
        w = self.attention(x)
        w_mean = torch.sum(x * w, dim=2)
        w_std = torch.sqrt((torch.sum((x**2) * w, dim=2) - w_mean**2).clamp(min=1e-5))
        x = torch.cat((w_mean, w_std), dim=1)  # [B, 2F]
        x = x.unsqueeze(-1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x   # [B, 256, 1]

class LayerNorm_dp(torch.nn.Module):
    def __init__(self, nout: int):
        super(LayerNorm_dp, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(nout, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x.transpose(1, -1))
        x = x.transpose(1, -1)
        return x

class DurationPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=4, n_chans=512, kernel_size=3,
                 dropout_rate=0.1, offset=1.0, max_duration=100):
        """
        idim: dep_c_dim + g_end_dim。
              请根据实际特征维度设置，例如 dep_c 为 768-dim（mHuBERT base）
              时 idim = 768 + 256 = 1024。
        """
        super(DurationPredictor, self).__init__()
        self.offset = float(offset)
        self.max_duration = int(max_duration)
        if self.max_duration < 1:
            raise ValueError("max_duration must be >= 1.")
        self.max_log_duration = float(
            np.log(self.max_duration + self.offset)
        )
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1,
                                    padding=(kernel_size - 1) // 2),
                    torch.nn.ReLU(),
                    LayerNorm_dp(n_chans),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(self, x0: torch.Tensor, x1: torch.Tensor,
                 x_masks: Optional[torch.Tensor] = None,
                 is_inference: bool = False):
        x0 = x0.detach()
        x1 = x1.detach()
       
        if x1.dim() == 3 and x1.shape[2] != 1:
            x1 = x1.permute(0, 2, 1)  # [B, 1, C] → [B, C, 1]

        x1_pad = x1.expand(-1, -1, x0.shape[2])
        xs = torch.cat([x0, x1_pad], dim=1)
        for f in self.conv:
            xs = f(xs)
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)   # [B, T]
        if is_inference:
            # Clamp in log space before exp() to prevent a single abnormal
            # prediction from creating an unbounded output sequence.
            xs = xs.clamp(
                min=0.0,
                max=self.max_log_duration,
            )
            xs = torch.round(xs.exp() - self.offset)
            xs = xs.clamp(min=0, max=self.max_duration).long()
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0)
        return xs

    def forward(self, x0: torch.Tensor, x1: torch.Tensor,
                x_masks: Optional[torch.Tensor] = None):
    
        return self._forward(x0, x1, x_masks, False)

    def inference(self, x0: torch.Tensor, x1: torch.Tensor,
                  x_masks: Optional[torch.Tensor] = None):
      
        return self._forward(x0, x1, x_masks, True)

class TriD_Pb(nn.Module):
    def __init__(self, hidden_size=256, p=0.5, eps=1e-6, alpha=0.5):
        super().__init__()
        self.p = p
        self.eps = eps
        self.hidden_size = hidden_size
        self._activated = True
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.affine_layer = LinearNorm_mixln(hidden_size, 2 * hidden_size)

    def pert_no_mu(self, std):
        epsilon = torch.randn_like(std) * 0.3
        return epsilon * std


    def sqrtvar(self, x):
        t = (x.std(dim=0, keepdim=True) + self.eps)
        t = t.repeat(x.shape[0], 1, 1)
        return t

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x, spk):
        if not self.training or not self._activated:
            return x
        if random.random() > self.p:
            return x
        B, C, T = x.shape
        mu = x.mean(dim=[2], keepdim=True)
        var = x.var(dim=[2], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()   
        x_normed = (x - mu) / sig
        spk = spk.permute(0, 2, 1)  #[B, T, C]
        mu1, sig1 = torch.split(self.affine_layer(spk), self.hidden_size, dim=-1)
        mu1 = mu1.transpose(1, 2)  #[B, C, T]
        sig1 = sig1.transpose(1, 2)  #[B, C, T]
        # sig1 = F.softplus(sig1) + self.eps
        perm = torch.randperm(B, device=x.device)  #shuffle
        mu_spk = mu1[perm]
        sig_spk = sig1[perm]
        lmda = self.beta.sample((B, C, 1)).to(x.device)  #Beta
        # bernoulli = torch.bernoulli(lmda).to(x.device)  #[B, C, 1]
        mu_mix = mu1 * lmda + mu_spk * (1. - lmda)
        sig_mix = sig1 * lmda + sig_spk * (1. - lmda)
        mu1_sq = self.sqrtvar(mu_mix)
        sig1_sq = self.sqrtvar(sig_mix)
        mu_end = mu_mix + self.pert_no_mu(mu1_sq)
        sig_end = sig_mix + self.pert_no_mu(sig1_sq)
        return x_normed * sig_end + mu_end

class WeightedSum(nn.Module):
    def __init__(self, init_alpha=0.5):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.tensor([init_alpha]))

    def forward(self, x, y):
        alpha = torch.sigmoid(self.raw_alpha)
        return alpha * x + (1 - alpha) * y


class Generator(torch.nn.Module):
    def __init__(self, h, F0_model):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.m_source = SourceModuleHnNSF(
            sampling_rate=h.sampling_rate,
            upsample_scale=np.prod(h.upsample_rates) * h.gen_istft_hop_size,
            harmonic_num=8, voiced_threshod=10)
        self.f0_upsamp = torch.nn.Upsample(
            scale_factor=np.prod(h.upsample_rates) * h.gen_istft_hop_size)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()

        self.F0_model = F0_model

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))
            c_cur = h.upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(h.upsample_rates):
                stride_f0 = np.prod(h.upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(
                    h.gen_istft_n_fft + 2, c_cur,
                    kernel_size=stride_f0 * 2, stride=stride_f0,
                    padding=(stride_f0 + 1) // 2))
                self.noise_res.append(resblock(h, c_cur, 7, [1, 3, 5]))
            else:
                self.noise_convs.append(Conv1d(h.gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(resblock(h, c_cur, 11, [1, 3, 5]))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.post_n_fft = h.gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(filter_length=h.gen_istft_n_fft,
                              hop_length=h.gen_istft_hop_size,
                              win_length=h.gen_istft_n_fft)

        gin_channels = 256
        inter_channels = hidden_channels = h.upsample_initial_channel - gin_channels

        self.enc = Encoder1(256, inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
        self.TVTR = TimeVaryingTimbreModule( d_g=gin_channels, d_c=inter_channels, d_model=hidden_channels, K=48, num_heads=4, n_templates=6, max_topk=48, style_bias_template_init_std=0.03, style_bias_scale_init=1.2, style_bias_scale_max=2.5,)
        self.conv1d = Conv1d(257, 256, 1)
        self.dec = Encoder(inter_channels, inter_channels, hidden_channels, 5, 1, 20, gin_channels=gin_channels)
        self.lf1 = nn.Linear(512, 256)
        self.face = FaceEncoder(dim_emb=256)
        self.reparam = Repara(256, 320, 256)
        self.flow_trans = ResidualCouplingBlock(256, 256, 5, 1, 2, gin_channels=256)
        self.TriD_Pb = TriD_Pb()
        self.enc_spk = MelStyleEncoder(in_dim=80, hidden_dim=256, out_dim=256)
        self.linear_c = nn.Linear(768, 256)
        self.dur_pre = DurationPredictor(idim=1024)   # idim = dep_c_dim + g_end_dim，请按实际维度调整
        self.ASP = ASP(in_channels=256)
        self.ASP1 = ASP(in_channels=256)
        self.weight = WeightedSum()

    def repeat_content(
        self,
        content,
        durations,
        max_output_frames=None,
    ):
        """
        Expand deduplicated content according to integer durations.

        Supports arbitrary batch size. Variable-length outputs are padded to the
        longest sample in the batch. Padding frames are zeros.
        """
        if content.ndim != 3 or durations.ndim != 2:
            raise ValueError(
                "Expected content [B, D, T] and durations [B, T], got {} and {}."
                .format(tuple(content.shape), tuple(durations.shape))
            )
        if content.size(0) != durations.size(0):
            raise ValueError("Content/duration batch sizes do not match.")
        if content.size(2) != durations.size(1):
            raise ValueError("Content/duration token lengths do not match.")

        if max_output_frames is None:
            max_output_frames = int(
                getattr(self.h, 'max_output_frames', 12000)
            )
        max_output_frames = int(max_output_frames)
        if max_output_frames < 1:
            raise ValueError("max_output_frames must be >= 1.")

        outputs = []
        lengths = []
        for batch_index in range(content.size(0)):
            sample_content = content[batch_index]
            sample_durations = durations[batch_index].long().clamp(
                min=0,
                max=self.dur_pre.max_duration,
            )

            repeated = torch.repeat_interleave(
                sample_content,
                sample_durations,
                dim=1,
            )
            if repeated.size(1) == 0:
                repeated = sample_content[:, :1]

            if repeated.size(1) > max_output_frames:
                repeated = repeated[:, :max_output_frames]

            outputs.append(repeated)
            lengths.append(repeated.size(1))

        padded_length = max(lengths)
        padded_outputs = []
        for repeated in outputs:
            if repeated.size(1) < padded_length:
                repeated = F.pad(
                    repeated,
                    (0, padded_length - repeated.size(1)),
                    mode='constant',
                    value=0.0,
                )
            padded_outputs.append(repeated)

        return torch.stack(padded_outputs, dim=0)

    def _make_dur_masks(self, dep_c, dep_c_len=None):
        """
        dep_c     : [B, D, T_dep]
        dep_c_len : [B] long，每个样本实际的去重 token 数；None 则用零检测
        Returns   : [B, T_dep] bool，True = padding 位置（需屏蔽）
        """
        B, D, T_dep = dep_c.shape
        if dep_c_len is not None:
            # 精确掩码：超出实际长度的位置为 True
            idx = torch.arange(T_dep, device=dep_c.device).unsqueeze(0)  # [1, T_dep]
            masks = idx >= dep_c_len.unsqueeze(1)  # [B, T_dep]
        else:
            # 回退方案：全零行判定为 padding
            masks = (dep_c.abs().sum(dim=1) == 0)  # [B, T_dep]
        return masks


    def extract_prosodic_features(self, mel):
        return extract_prosodic_features(mel, self.F0_model)

    def forward(
        self,
        x,
        mel,
        face_emb,
        dep_c,
        dur_gt,
        return_aux: bool = False,
        dep_c_len=None,
    ):
        g_t = self.enc_spk(mel.transpose(1, 2))               
        mel_tr = torch.flip(mel, dims=[2])  # [B, 50, 80]
        g_t_tr = self.enc_spk(mel_tr.transpose(1, 2))         
        g_tr = self.ASP1(g_t_tr.transpose(1, 2))             
        g_t_and_tr = self.weight(g_t, g_tr.transpose(1, 2))  
        g_end = self.ASP(g_t_and_tr.transpose(1, 2))          
        g_end = g_end.permute(0, 2, 1)                      
        g_spk, _, _ = self.reparam(g_end)                     
        g_spk = g_spk.permute(0, 2, 1)
        # g = g_spk# [B, 256, 1]

     
        face_emb = self.lf1(face_emb)                        
        g_face = self.face(face_emb)                         
        g_flow = self.flow_trans(g_face)                    
        # g_flow = g_face
        g = random.choices([g_spk, g_flow], weights=[5, 5])[0] 

       
        # f0, _, _ = self.F0_model(mel.unsqueeze(1))            
        # f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)      
        # har_source, _, _ = self.m_source(f0)                 
        # har_source = har_source.transpose(1, 2).squeeze(1)     
        # har_spec, har_phase = self.stft.transform(har_source) 
        # har = torch.cat([har_spec, har_phase], dim=1)          

        # Use the explicit token length whenever available. Inferring padding
        # from duration==0 is ambiguous after segment-boundary correction.
        dur_masks = self._make_dur_masks(dep_c, dep_c_len)


        dur_pre = self.dur_pre.forward(dep_c, g, dur_masks)
        x_content = self.linear_c(x.transpose(1, 2)).transpose(1, 2)
        x_content = self.TriD_Pb(x_content, g)

        x = self.enc(x_content, g=g)     
        g, aux = self.TVTR(g, x, return_aux=True)  
        x = self.dec(x, g=g)             
        x = torch.cat([x, g], dim=1)     

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            # x_source = self.noise_convs[i](har)
            # x_source = self.noise_res[i](x_source)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            # x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec  = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        g_re1 = g_spk.squeeze(2)    # [B, 256]
        g_re2 = g_flow.squeeze(2)   # [B, 256]

        if return_aux:
            return spec, phase, g_re1, g_re2, dur_pre, g_end, aux

        return spec, phase, g_re1, g_re2, dur_pre, g_end

    def get_f0(self, mel, f0_mean_tgt, voiced_threshold=10):
        f0, _, _ = self.F0_model(mel.unsqueeze(1))
        voiced = f0 > voiced_threshold
        lf0 = torch.log(f0)
        lf0_ = lf0 * voiced.float()
        lf0_mean = lf0_.sum(1) / voiced.float().sum(1)
        lf0_mean = lf0_mean.unsqueeze(1)
        lf0_adj = lf0 - lf0_mean + torch.log(f0_mean_tgt)
        f0_adj = torch.exp(lf0_adj)
        energy = mel.sum(1)
        unsilent = energy > -700
        unsilent = unsilent | voiced
        f0_adj = f0_adj * unsilent.float()
        return f0_adj

    def get_x(self, mel, dep_c, face_emb, dep_c_len=None):
        face_emb = self.lf1(face_emb)
        g_face = self.face(face_emb)
        g_flow = self.flow_trans(g_face)
        g = g_flow
        if g.dim() == 3 and g.shape[-1] != 1:
            g = g.transpose(1, 2)

        dur_masks = self._make_dur_masks(dep_c, dep_c_len)
        dur_pre_int = self.dur_pre.inference(dep_c, g, dur_masks)
        dur_pre = dur_pre_int.float()

        dep_c_rep = self.repeat_content(dep_c, dur_pre)
        x_content = self.linear_c(dep_c_rep.transpose(1, 2)).transpose(1, 2)

        x = self.enc(x_content, g=g)
        g = self.TVTR(g, x, return_aux=False)
        x = self.dec(x, g=g)
        x = torch.cat([x, g], dim=1)
        return x

    def get_x_face(self, mel, dep_c, face_emb, dep_c_len=None):
        face_emb = self.lf1(face_emb)
        g_face = self.face(face_emb)
        g_flow = self.flow_trans(g_face)
        g = g_flow
     
        if g.dim() == 3 and g.shape[-1] != 1:
            g = g.transpose(1, 2)

        dur_masks = self._make_dur_masks(dep_c, dep_c_len)
        dur_pre_int = self.dur_pre.inference(dep_c, g, dur_masks)
        dur_pre = dur_pre_int.float()
        dep_c_rep = self.repeat_content(dep_c, dur_pre)
        x_content = self.linear_c(dep_c_rep.transpose(1, 2)).transpose(1, 2)

      
        x = self.enc(x_content, g=g)
        g = self.TVTR(g, x, return_aux=False)
        x = self.dec(x, g=g)
        x = torch.cat([x, g], dim=1)
        return x

    def infer(self, x):
        # f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        # har_source, _, _ = self.m_source(f0)
        # har_source = har_source.transpose(1, 2).squeeze(1)
        # har_spec, har_phase = self.stft.transform(har_source)
        # har = torch.cat([har_spec, har_phase], dim=1)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            # x_source = self.noise_convs[i](har)
            # x_source = self.noise_res[i](x_source)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            # x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec  = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        y = self.stft.inverse(spec, phase)
        return y

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_post)


def stft(x, fft_size, hop_size, win_length, window):
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    return torch.abs(x_stft).transpose(2, 1)


class SpecDiscriminator(nn.Module):
    def __init__(self, fft_size=1024, shift_size=120, win_length=600,
                 window="hann_window", use_spectral_norm=False):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.discriminators = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ])
        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):
        fmap = []
        y = y.squeeze(1)
        y = stft(y, self.fft_size, self.shift_size, self.win_length,
                 self.window.to(y.get_device()))
        y = y.unsqueeze(1)
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)
        y = self.out(y)
        fmap.append(y)
        return torch.flatten(y, 1, -1), fmap


class MultiResSpecDiscriminator(torch.nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240], window="hann_window"):
        super(MultiResSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
            SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
            SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window)
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2), DiscriminatorP(3), DiscriminatorP(5),
            DiscriminatorP(7), DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses


def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        delta = dr - dg
        m_DG = torch.median(delta)
        error = (delta - m_DG) ** 2
        selected = error[dr < dg + m_DG]
        L_rel = selected.mean() if selected.numel() > 0 else error.new_zeros(())
        loss = loss + tau - F.relu(tau - L_rel)
    return loss


def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        delta = dr - dg
        m_DG = torch.median(delta)
        error = (delta - m_DG) ** 2
        selected = error[dr < dg + m_DG]
        L_rel = selected.mean() if selected.numel() > 0 else error.new_zeros(())
        loss = loss + tau - F.relu(tau - L_rel)
    return loss


