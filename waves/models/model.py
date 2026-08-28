"""
File: model.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Runtime WAVES model with sparse Mixture-of-Experts routing.

License: MIT License
"""

from typing import Any, cast

import torch
from torch import Tensor, nn

from waves.models.config import WAVESConfig
from waves.models.noise_conditioning import build_noise_source
from waves.models.transformer import TransformerBlock


class LearnableSigmoid2d(nn.Module):
    """Learnable sigmoid used by the magnitude mask decoder."""

    def __init__(
        self,
        in_features: int,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.beta = beta

        self.slope = nn.Parameter(
            torch.ones(
                in_features,
                1,
            )
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Apply the learnable sigmoid."""

        return self.beta * torch.sigmoid(self.slope * x)


class SPConvTranspose2d(nn.Module):
    """Sub-pixel convolution used by the WAVES decoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        r: int = 1,
    ) -> None:
        super().__init__()

        self.pad1 = nn.ConstantPad2d(
            (1, 1, 0, 0),
            value=0.0,
        )

        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * r,
            kernel_size=kernel_size,
            stride=(1, 1),
        )

        self.r = r

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Apply sub-pixel convolution."""

        x = cast(
            Tensor,
            self.pad1(x),
        )

        output = cast(
            Tensor,
            self.conv(x),
        )

        (
            batch_size,
            num_channels,
            height,
            width,
        ) = output.shape

        output = output.view(
            (
                batch_size,
                self.r,
                num_channels // self.r,
                height,
                width,
            )
        )

        output = output.permute(
            0,
            2,
            3,
            4,
            1,
        )

        return output.contiguous().view(
            (
                batch_size,
                num_channels // self.r,
                height,
                -1,
            )
        )


class DenseBlock(nn.Module):
    """Dilated dense convolution block."""

    def __init__(
        self,
        config: WAVESConfig,
        kernel_size: tuple[int, int] = (2, 3),
        depth: int = 4,
    ) -> None:
        super().__init__()

        self.depth = depth

        self.dense_block = nn.ModuleList()

        for index in range(depth):
            dilation = 2**index
            pad_length = dilation

            dense_conv = nn.Sequential(
                nn.ConstantPad2d(
                    (
                        1,
                        1,
                        pad_length,
                        0,
                    ),
                    value=0.0,
                ),
                nn.Conv2d(
                    config.dense_channel * (index + 1),
                    config.dense_channel,
                    kernel_size,
                    dilation=(
                        dilation,
                        1,
                    ),
                ),
                nn.InstanceNorm2d(
                    config.dense_channel,
                    affine=True,
                ),
                nn.PReLU(config.dense_channel),
            )

            self.dense_block.append(dense_conv)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Apply the dense block."""

        skip = x

        for index in range(self.depth):
            layer = self.dense_block[index]

            x = cast(
                Tensor,
                layer(skip),
            )

            skip = torch.cat(
                [
                    x,
                    skip,
                ],
                dim=1,
            )

        return x


class DenseEncoder(nn.Module):
    """Dense encoder used by WAVES."""

    def __init__(
        self,
        config: WAVESConfig,
        in_channel: int,
    ) -> None:
        super().__init__()

        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channel,
                config.dense_channel,
                (1, 1),
            ),
            nn.InstanceNorm2d(
                config.dense_channel,
                affine=True,
            ),
            nn.PReLU(config.dense_channel),
        )

        self.dense_block = DenseBlock(
            config,
            depth=config.dense_depth,
        )

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(
                config.dense_channel,
                config.dense_channel,
                (1, 3),
                (1, 2),
                padding=(0, 1),
            ),
            nn.InstanceNorm2d(
                config.dense_channel,
                affine=True,
            ),
            nn.PReLU(config.dense_channel),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Encode magnitude and phase features."""

        x = cast(
            Tensor,
            self.dense_conv_1(x),
        )

        x = cast(
            Tensor,
            self.dense_block(x),
        )

        return cast(
            Tensor,
            self.dense_conv_2(x),
        )


class MaskDecoder(nn.Module):
    """Magnitude mask decoder."""

    def __init__(
        self,
        config: WAVESConfig,
        out_channel: int = 1,
    ) -> None:
        super().__init__()

        self.dense_block = DenseBlock(
            config,
            depth=config.dense_depth,
        )

        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(
                config.dense_channel,
                config.dense_channel,
                (1, 3),
                2,
            ),
            nn.InstanceNorm2d(
                config.dense_channel,
                affine=True,
            ),
            nn.PReLU(config.dense_channel),
            nn.Conv2d(
                config.dense_channel,
                out_channel,
                (1, 2),
            ),
        )

        self.lsigmoid = LearnableSigmoid2d(
            config.n_fft // 2 + 1,
            beta=config.beta,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Decode the magnitude mask."""

        x = cast(
            Tensor,
            self.dense_block(x),
        )

        x = cast(
            Tensor,
            self.mask_conv(x),
        )

        x = x.permute(
            0,
            3,
            2,
            1,
        ).squeeze(-1)

        return cast(
            Tensor,
            self.lsigmoid(x),
        )


class PhaseDecoder(nn.Module):
    """Phase decoder."""

    def __init__(
        self,
        config: WAVESConfig,
        out_channel: int = 1,
    ) -> None:
        super().__init__()

        self.dense_block = DenseBlock(
            config,
            depth=config.dense_depth,
        )

        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(
                config.dense_channel,
                config.dense_channel,
                (1, 3),
                2,
            ),
            nn.InstanceNorm2d(
                config.dense_channel,
                affine=True,
            ),
            nn.PReLU(config.dense_channel),
        )

        self.phase_conv_r = nn.Conv2d(
            config.dense_channel,
            out_channel,
            (1, 2),
        )

        self.phase_conv_i = nn.Conv2d(
            config.dense_channel,
            out_channel,
            (1, 2),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Decode phase estimates."""

        x = cast(
            Tensor,
            self.dense_block(x),
        )

        x = cast(
            Tensor,
            self.phase_conv(x),
        )

        real = cast(
            Tensor,
            self.phase_conv_r(x),
        )

        imaginary = cast(
            Tensor,
            self.phase_conv_i(x),
        )

        phase = torch.atan2(
            imaginary,
            real,
        )

        return phase.permute(
            0,
            3,
            2,
            1,
        ).squeeze(-1)


class TSTransformerBlock(nn.Module):
    """Time-frequency transformer block."""

    def __init__(
        self,
        config: WAVESConfig,
        moe_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        separate_mag_phase = (
            bool(
                moe_config.get(
                    "separate_mag_phase",
                    False,
                )
            )
            if moe_config is not None
            else False
        )

        self.time_transformer = TransformerBlock(
            d_model=config.dense_channel,
            n_heads=config.n_heads,
            moe_config=moe_config,
        )

        self.freq_transformer = TransformerBlock(
            d_model=config.dense_channel,
            n_heads=config.n_heads,
            moe_config=moe_config,
        )

        if not separate_mag_phase and moe_config is not None:
            self.freq_transformer.ffn = self.time_transformer.ffn

    def forward(
        self,
        x: Tensor,
        noise_embed: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply time and frequency transformer paths."""

        (
            batch_size,
            channels,
            time_steps,
            frequency_bins,
        ) = x.size()

        x = (
            x.permute(
                0,
                3,
                2,
                1,
            )
            .contiguous()
            .view(
                batch_size * frequency_bins,
                time_steps,
                channels,
            )
        )

        time_noise_context = None

        if noise_embed is not None:
            time_noise_context = (
                noise_embed.unsqueeze(1)
                .expand(
                    batch_size,
                    frequency_bins,
                    -1,
                )
                .contiguous()
                .view(
                    batch_size * frequency_bins,
                    -1,
                )
            )

        time_result = cast(
            tuple[Tensor, Tensor],
            self.time_transformer(
                x,
                noise_ctx=time_noise_context,
            ),
        )

        transformed = time_result[0]
        time_auxiliary_loss = time_result[1]

        x = transformed + x

        x = (
            x.view(
                batch_size,
                frequency_bins,
                time_steps,
                channels,
            )
            .permute(
                0,
                2,
                1,
                3,
            )
            .contiguous()
            .view(
                batch_size * time_steps,
                frequency_bins,
                channels,
            )
        )

        frequency_noise_context = None

        if noise_embed is not None:
            frequency_noise_context = (
                noise_embed.unsqueeze(1)
                .expand(
                    batch_size,
                    time_steps,
                    -1,
                )
                .contiguous()
                .view(
                    batch_size * time_steps,
                    -1,
                )
            )

        frequency_result = cast(
            tuple[Tensor, Tensor],
            self.freq_transformer(
                x,
                noise_ctx=frequency_noise_context,
            ),
        )

        transformed = frequency_result[0]
        frequency_auxiliary_loss = frequency_result[1]

        x = transformed + x

        x = x.view(
            batch_size,
            time_steps,
            frequency_bins,
            channels,
        ).permute(
            0,
            3,
            1,
            2,
        )

        return (
            x,
            time_auxiliary_loss + frequency_auxiliary_loss,
        )


class WAVESModel(nn.Module):
    """WAVES generator with sparse Mixture-of-Experts routing."""

    def __init__(
        self,
        config: WAVESConfig,
    ) -> None:
        super().__init__()

        self.dense_encoder = DenseEncoder(
            config,
            in_channel=2,
        )

        moe_config = config.moe

        moe_layers = (
            {
                int(index)
                for index in moe_config.get(
                    "apply_to",
                    [],
                )
            }
            if moe_config is not None
            else set()
        )

        self.TSTransformer = nn.ModuleList()

        for index in range(config.num_tsblocks):
            layer_moe_config = moe_config if index in moe_layers else None

            self.TSTransformer.append(
                TSTransformerBlock(
                    config,
                    moe_config=layer_moe_config,
                )
            )

        self.mask_decoder = MaskDecoder(
            config,
            out_channel=1,
        )

        self.phase_decoder = PhaseDecoder(
            config,
            out_channel=1,
        )

        noise_source = (
            str(
                moe_config.get(
                    "noise_source",
                    "mean_spectrum",
                )
            )
            if moe_config is not None
            else "mean_spectrum"
        )

        noise_context_dim = (
            int(
                moe_config.get(
                    "noise_ctx_dim",
                    0,
                )
            )
            if moe_config is not None
            else 0
        )

        self._noise_source_type = noise_source

        if moe_layers and noise_context_dim > 0:
            num_frequency_bins = config.n_fft // 2 + 1

            self.noise_proj: nn.Module | None = build_noise_source(
                noise_source=noise_source,
                n_freq=num_frequency_bins,
                noise_ctx_dim=noise_context_dim,
                encoder_channels=config.dense_channel,
            )

        else:
            self.noise_proj = None

    def forward(
        self,
        noisy_amp: Tensor,
        noisy_pha: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Enhance magnitude and phase spectra."""

        x = torch.stack(
            (
                noisy_amp,
                noisy_pha,
            ),
            dim=-1,
        ).permute(
            0,
            3,
            2,
            1,
        )

        x = cast(
            Tensor,
            self.dense_encoder(x),
        )

        noise_embed: Tensor | None = None

        if self.noise_proj is not None:
            if self._noise_source_type == "encoder":
                noise_embed = cast(
                    Tensor,
                    self.noise_proj(
                        noisy_amp,
                        encoder_out=x,
                    ),
                )

            else:
                noise_embed = cast(
                    Tensor,
                    self.noise_proj(noisy_amp),
                )

        total_auxiliary_loss = noisy_amp.new_zeros(())

        if self.noise_proj is not None:
            commitment_loss = getattr(
                self.noise_proj,
                "_last_commitment_loss",
                None,
            )

            if isinstance(
                commitment_loss,
                Tensor,
            ):
                total_auxiliary_loss = total_auxiliary_loss + commitment_loss

            elif isinstance(
                commitment_loss,
                int | float,
            ):
                total_auxiliary_loss = total_auxiliary_loss + float(commitment_loss)

        for transformer_module in self.TSTransformer:
            transformer_block = cast(
                TSTransformerBlock,
                transformer_module,
            )

            transformer_result = cast(
                tuple[Tensor, Tensor],
                transformer_block(
                    x,
                    noise_embed=noise_embed,
                ),
            )

            x = transformer_result[0]

            total_auxiliary_loss = total_auxiliary_loss + transformer_result[1]

        mask = cast(
            Tensor,
            self.mask_decoder(x),
        )

        denoised_amp = noisy_amp * mask

        denoised_pha = cast(
            Tensor,
            self.phase_decoder(x),
        )

        denoised_com = torch.stack(
            (
                denoised_amp * torch.cos(denoised_pha),
                denoised_amp * torch.sin(denoised_pha),
            ),
            dim=-1,
        )

        return (
            denoised_amp,
            denoised_pha,
            denoised_com,
            total_auxiliary_loss,
        )
