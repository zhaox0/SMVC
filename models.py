import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding
import numpy as np
import random
from stft import TorchSTFT
from face_encoder import FaceEncoder, Repara
from flow_recoulping import ResidualCouplingBlock

LRELU_SLOPE = 0.1


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts


class WN(torch.nn.Module):
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    super(WN, self).__init__()
    assert(kernel_size % 2 == 1)
    self.hidden_channels =hidden_channels
    self.kernel_size = kernel_size,
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout

    self.in_layers = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    self.drop = nn.Dropout(p_dropout)

    if gin_channels != 0:
      cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
      self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
      in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
      self.in_layers.append(in_layer)

      # last one is not necessary
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
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
      else:
        g_l = torch.zeros_like(x_in)

      acts = fused_add_tanh_sigmoid_multiply(
          x_in,
          g_l,
          n_channels_tensor)
      acts = self.drop(acts)

      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:,:self.hidden_channels,:]
        x = (x + res_acts) * x_mask
        output = output + res_skip_acts[:,self.hidden_channels:,:]
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


class Encoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
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
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

  def forward(self, x, x_mask=1, g=None):
    # x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
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
            xt = x + (1 / a1) * (torch.sin(a1 * x) ** 2)  # Snake1D
            xt = c1(xt)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
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
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
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
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], \
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
#             # for normal case

#             # To prevent torch.cumsum numerical overflow,
#             # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
#             # Buffer tmp_over_one_idx indicates the time step to add -1.
#             # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
#             tmp_over_one = torch.cumsum(rad_values, 1) % 1
#             tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
#             cumsum_shift = torch.zeros_like(rad_values)
#             cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

#             phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            rad_values = torch.nn.functional.interpolate(rad_values.transpose(1, 2), 
                                                         scale_factor=1/self.upsample_scale, 
                                                         mode="linear").transpose(1, 2)
    
#             tmp_over_one = torch.cumsum(rad_values, 1) % 1
#             tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
#             cumsum_shift = torch.zeros_like(rad_values)
#             cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
    
            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = torch.nn.functional.interpolate(phase.transpose(1, 2) * self.upsample_scale, 
                                                    scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
            sines = torch.sin(phase)
            
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim,
                             device=f0.device)
        # fundamental component
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # generate uv signal
        # uv = torch.ones(f0.shape)
        # uv = uv * (f0 > self.voiced_threshold)
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv
def padDiff(x):
    return F.pad(F.pad(x, (0,0,-1,1), 'constant', 0) - x, (0,0,0,-1), 'constant', 0)

            

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
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(h.upsample_rates) * h.gen_istft_hop_size)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        
        self.F0_model = F0_model
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

            c_cur = h.upsample_initial_channel // (2 ** (i + 1))
            
            if i + 1 < len(h.upsample_rates):  #
                stride_f0 = np.prod(h.upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(
                    h.gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
                self.noise_res.append(resblock(h, c_cur, 7, [1,3,5]))
            else:
                self.noise_convs.append(Conv1d(h.gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(resblock(h, c_cur, 11, [1,3,5]))
            
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.post_n_fft = h.gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(filter_length=h.gen_istft_n_fft, hop_length=h.gen_istft_hop_size, win_length=h.gen_istft_n_fft)

        gin_channels = 256
        inter_channels = hidden_channels = h.upsample_initial_channel - gin_channels

        self.enc = Encoder(768, inter_channels, hidden_channels, 5, 1, 4) 
        self.dec = Encoder(inter_channels, inter_channels, hidden_channels, 5, 1, 20, gin_channels=gin_channels)
        self.lf1 = nn.Linear(512, 256)
        self.face = FaceEncoder(dim_emb=256)
        self.reparam = Repara(256, 320, 256)
        self.flow_trans = ResidualCouplingBlock(256, 256, 5, 1, 2, gin_channels=256)

    def forward(self, x, mel, spk_emb, face_emb):
        # spk_emb
        g_spk, _, _ = self.reparam(spk_emb.unsqueeze(1))
        g_spk = g_spk.permute(0, 2, 1)
        # face_emb
        face_emb = self.lf1(face_emb)
        g_face = self.face(face_emb)
        g_flow= self.flow_trans(g_face)
        g = random.choices([g_spk, g_flow], weights=[5, 5])[0]

        f0, _, _ = self.F0_model(mel.unsqueeze(1))
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t

        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)

        x = self.enc(x)
        x = self.dec(x, g=g)
        g = g.repeat(1, 1, x.shape[-1])
        x = torch.cat([x, g], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source)
            
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        g_re1 = g_spk.squeeze(2)
        g_re2 = g_flow.squeeze(2)

        return spec, phase, g_re1, g_re2 

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
        unsilent = unsilent | voiced    # simple vad
        f0_adj = f0_adj * unsilent.float()

        return f0_adj
    
    def get_x(self, x, face_emb):
        face_emb = self.lf1(face_emb)
        g_face = self.face(face_emb)
        g = g_face.permute(0, 2, 1)
        # g = g + spk_emb.unsqueeze(-1)

        x = self.enc(x)
        x = self.dec(x, g=g)
        g = g.repeat(1, 1, x.shape[-1])
        x = torch.cat([x, g], dim=1)

        return x

    def get_x_face(self, x, face_emb):
        # g_spk, _, _ = self.reparam(spk_emb.unsqueeze(1))
        face_emb = self.lf1(face_emb)
        g_face = self.face(face_emb)
        g_flow= self.flow_trans(g_face)
        g = g_flow

        x = self.enc(x)
        x = self.dec(x, g=g)
        g = g.repeat(1, 1, x.shape[-1])
        x = torch.cat([x, g], dim=1)

        return x
    
    def get_x_spk(self, x, spk_emb):
        g_spk, _, _ = self.reparam(spk_emb.unsqueeze(1))
        # g = g + spk_emb.unsqueeze(-1)
        g = g_spk.permute(0, 2, 1)
        x = self.enc(x)
        x = self.dec(x, g=g)
        g = g.repeat(1, 1, x.shape[-1])
        x = torch.cat([x, g], dim=1)

        return x
    
    def infer(self, x, f0):
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source)
            
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
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
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window,
            return_complex=True)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.abs(x_stft).transpose(2, 1)

class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", use_spectral_norm=False):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.discriminators = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))),
        ])

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):

        fmap = []
        y = y.squeeze(1)
        y = stft(y, self.fft_size, self.shift_size, self.win_length, self.window.to(y.get_device()))
        y = y.unsqueeze(1)
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap

class MultiResSpecDiscriminator(torch.nn.Module):

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window"):

        super(MultiResSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
            SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
            SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window)
            ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
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

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
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
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
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
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
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

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

if __name__ == "__main__":
    import argparse
    import json
    from env import AttrDict
    from Utils.JDC.model import JDCNet
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='config_v1_16k.json')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    
    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(h.F0_path)['model']
    F0_model.load_state_dict(params)

    x = torch.randn(4, 768, 60)
    mel = torch.randn(4, 80, 60)
    spk_emb = torch.randn(4, 256)
    face_emb = torch.randn(4, 1, 512)

    generator = Generator(h, F0_model)
    spec, phase = generator(x, mel, spk_emb, face_emb)