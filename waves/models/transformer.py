"""
File: transformer.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Transformer blocks used by WAVES.

License: MIT License
"""

from typing import Any, cast

from torch import Tensor, nn
import torch.nn.functional as F

from waves.models.moe import MoEFFN


def build_moe_ffn(
    moe_config: dict[str, Any],
    d_model: int,
    bidirectional: bool,
    dropout: float,
) -> MoEFFN:
    """Build the token-choice MoE feed-forward module."""

    routing = moe_config.get(
        "routing",
        "token_choice",
    )

    if routing not in {
        "token_choice",
        None,
    }:
        msg = f"WAVES currently supports only token-choice MoE routing, got: {routing!r}."
        raise ValueError(msg)

    return MoEFFN(
        d_model=d_model,
        num_experts=int(
            moe_config.get(
                "num_experts",
                4,
            )
        ),
        top_k=int(
            moe_config.get(
                "top_k",
                2,
            )
        ),
        expert_ffn_dim=int(
            moe_config.get(
                "expert_ffn_dim",
                256,
            )
        ),
        balance_loss_weight=float(
            moe_config.get(
                "balance_loss_weight",
                0.01,
            )
        ),
        z_loss_weight=float(
            moe_config.get(
                "z_loss_weight",
                0.001,
            )
        ),
        bias_update_speed=float(
            moe_config.get(
                "bias_update_speed",
                0.001,
            )
        ),
        noise_ctx_dim=int(
            moe_config.get(
                "noise_ctx_dim",
                0,
            )
        ),
        noise_injection=str(
            moe_config.get(
                "noise_injection",
                "concat",
            )
        ),
        expert_type=str(
            moe_config.get(
                "expert_type",
                "shared_gru",
            )
        ),
        use_expert_bias=bool(
            moe_config.get(
                "use_expert_bias",
                True,
            )
        ),
        bidirectional=bidirectional,
        dropout=dropout,
    )


class FFN(nn.Module):
    """GRU-based FFN used by non-MoE transformer blocks."""

    def __init__(
        self,
        d_model: int,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            d_model,
            d_model * 2,
            1,
            bidirectional=bidirectional,
            batch_first=True,
        )

        linear_input_dim = d_model * 4 if bidirectional else d_model * 2

        self.linear = nn.Linear(
            linear_input_dim,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Apply the GRU feed-forward network."""

        self.gru.flatten_parameters()

        x = cast(
            Tensor,
            self.gru(x)[0],
        )

        x = F.leaky_relu(x)

        x = cast(
            Tensor,
            self.dropout(x),
        )

        return cast(
            Tensor,
            self.linear(x),
        )


class TransformerBlock(nn.Module):
    """Attention and FFN transformer block used by WAVES."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bidirectional: bool = True,
        dropout: float = 0.0,
        moe_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)

        self.use_moe = moe_config is not None

        if moe_config is not None:
            self.ffn: nn.Module = build_moe_ffn(
                moe_config=moe_config,
                d_model=d_model,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        else:
            self.ffn = FFN(
                d_model=d_model,
                bidirectional=bidirectional,
                dropout=dropout,
            )

        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        noise_ctx: Tensor | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply attention and the configured feed-forward network."""

        transformed = cast(
            Tensor,
            self.norm1(x),
        )

        attention_result = self.attention(
            transformed,
            transformed,
            transformed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        transformed = cast(
            Tensor,
            attention_result[0],
        )

        transformed = cast(
            Tensor,
            self.dropout1(transformed),
        )

        x = x + transformed

        transformed = cast(
            Tensor,
            self.norm2(x),
        )

        if self.use_moe:
            moe_ffn = cast(
                MoEFFN,
                self.ffn,
            )

            moe_result = cast(
                tuple[Tensor, Tensor],
                moe_ffn(
                    transformed,
                    noise_ctx=noise_ctx,
                ),
            )

            transformed = moe_result[0]
            auxiliary_loss = moe_result[1]

        else:
            transformed = cast(
                Tensor,
                self.ffn(transformed),
            )

            auxiliary_loss = transformed.new_zeros(())

        transformed = cast(
            Tensor,
            self.dropout2(transformed),
        )

        x = x + transformed

        x = cast(
            Tensor,
            self.norm3(x),
        )

        return (
            x,
            auxiliary_loss,
        )
