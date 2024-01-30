import math
from typing import List
from typing import Union

import librosa
import numpy as np
import soundfile
import torch
from torch import nn

from .quantize import VectorQuantize
from .layers import Snake1d
from .layers import WNConv1d
from .layers import WNConvTranspose1d
import torch.distributions as D
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


mel_transform = None
def get_mel(wav):
    global mel_transform
    if mel_transform is None:
        from .spectrogram import LogMelSpectrogram
        mel_transform = LogMelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            f_min=40,
            f_max=16000,
            n_mels=128,
        ).to(wav.device).to(wav.dtype)
    return mel_transform(wav)

class PostEncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim, dilation=1),
            ResidualUnit(dim, dilation=3),
            ResidualUnit(dim, dilation=9),
            Snake1d(dim),
            WNConv1d(
                dim,
                dim,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
        )

    def forward(self, x):
        return self.block(x)

class DAV(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        vae_latent_channels: int = 64,
        vq_reg_codebook_size: int = 4096,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        # self.quantizer = ResidualVectorQuantize(
        #     input_dim=latent_dim,
        #     n_codebooks=n_codebooks,
        #     codebook_size=codebook_size,
        #     codebook_dim=codebook_dim,
        #     quantizer_dropout=quantizer_dropout,
        # )

        self.quantizer = VectorQuantize(
            input_dim=latent_dim,
            codebook_size=128,
            codebook_dim=32
        )

        mel_dim = 128

        self.mel_proj = nn.Conv1d(mel_dim, latent_dim, 3, padding=1)
        self.post_encoder = PostEncoderBlock(latent_dim, 1)

        self.mean_proj = nn.Conv1d(latent_dim, vae_latent_channels, 1)
        self.logs_proj = nn.Conv1d(latent_dim, vae_latent_channels, 1)

        self.dec_in_proj = nn.Conv1d(vae_latent_channels, latent_dim, 1)

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)


    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
    ):
        mel = get_mel(audio_data)

        z_audio = self.encoder(audio_data)
        z_audio_q, commitment_loss, codebook_loss, indices, z_e = self.quantizer(z_audio)

        z_mel = self.mel_proj(mel)
        z = z_audio_q + z_mel
        z = self.post_encoder(z)

        mean = self.mean_proj(z)  # [B, C, T]
        logs = self.logs_proj(z)  # [B, C, T]
        logs = torch.clamp(logs, min=-12, max=12)

        posterior = D.Normal(mean, torch.exp(logs))
        prior = D.Normal(torch.zeros_like(mean), torch.ones_like(logs))
        kl_loss = D.kl_divergence(posterior, prior).mean()

        return posterior, kl_loss, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        z = self.dec_in_proj(z)
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        posterior, kl_loss, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers
        )
        z = posterior.rsample()
        assert not torch.isnan(z).any(), "z is nan"

        x = self.decode(z)
        assert not torch.isnan(x).any(), "x decode is nan"

        return {
            "audio": x[..., :length],
            "latent": z,
            "kl_loss": kl_loss,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }

    @torch.no_grad()
    @torch.inference_mode()
    def encode_from_wav44k_tensor(self, x):
        audio_data = self.preprocess(x, 44100)
        posterior, kl_loss, commitment_loss, codebook_loss = self.encode(
            audio_data, None
        )
        out = posterior.mode
        return out / self.norm_ratio

    @torch.no_grad()
    @torch.inference_mode()
    def decode_to_wav44k_tensor(self, z):
        x = self.decode(z * self.norm_ratio)
        return x

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = DAV(**ckpt['metadata']['kwargs'])
    model = model.to(device)
    model = model.eval()
    model.load_state_dict(ckpt['state_dict'])
    return model


def encode_from_wav44k_numpy(model, wav44k_numpy):
    device = model.parameters().__next__().device
    x = torch.FloatTensor(wav44k_numpy).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        audio_data = model.preprocess(x, 44100)
        posterior, kl_loss, commitment_loss, codebook_loss = model.encode(
            audio_data, None
        )
        out = posterior.mode
    return out.squeeze(0).cpu()


def encode_from_file(model, wav_path):
    wav44k_numpy, _ = librosa.load(wav_path, sr=44100, mono=True)
    return encode_from_wav44k_numpy(model, wav44k_numpy)

def decode_to_wav44k_numpy(model, z):
    device = model.parameters().__next__().device
    z = z.unsqueeze(0).to(device)
    with torch.no_grad():
        x = model.decode(z)
    return x.squeeze(0).squeeze(0).cpu().numpy()

def decode_to_file(model, z, wav_path):
    wav44k_numpy = decode_to_wav44k_numpy(model, z)
    soundfile.write(wav_path, wav44k_numpy, 44100)

