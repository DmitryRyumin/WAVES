"""
File: moe.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Sparse token-choice Mixture-of-Experts feed-forward module for WAVES.

License: MIT License
"""

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F

RoutingModifier = Callable[..., Tensor]


def trunc_normal_init(
    tensor: Tensor,
    std: float = 0.02,
    a: float = -0.06,
    b: float = 0.06,
) -> None:
    """Initialize a tensor with a truncated normal distribution."""

    nn.init.trunc_normal_(
        tensor,
        mean=0.0,
        std=std,
        a=a,
        b=b,
    )


class MoEFFN(nn.Module):
    """Token-choice top-k Mixture-of-Experts feed-forward module."""

    expert_bias: Tensor

    def __init__(
        self,
        d_model: int = 64,
        num_experts: int = 4,
        top_k: int = 2,
        expert_ffn_dim: int = 256,
        balance_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        bias_update_speed: float = 0.001,
        noise_ctx_dim: int = 0,
        noise_injection: str = "concat",
        expert_type: str = "shared_gru",
        use_expert_bias: bool = True,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_experts <= 0:
            msg = "num_experts must be greater than zero."
            raise ValueError(msg)

        if top_k <= 0 or top_k > num_experts:
            msg = "top_k must be in the range [1, num_experts]."
            raise ValueError(msg)

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.balance_loss_weight = balance_loss_weight
        self.z_loss_weight = z_loss_weight
        self.bias_update_speed = bias_update_speed

        self.noise_ctx_dim = noise_ctx_dim
        self.noise_injection = noise_injection
        self.expert_type = expert_type
        self.use_expert_bias = use_expert_bias

        self.gru: nn.GRU | None
        self._router_proj: nn.Linear | None = None

        if expert_type == "shared_gru":
            self.gru = nn.GRU(
                d_model,
                d_model * 2,
                1,
                bidirectional=bidirectional,
                batch_first=True,
            )

            self.gru_out_dim = d_model * 4 if bidirectional else d_model * 2

            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            self.gru_out_dim,
                            expert_ffn_dim,
                        ),
                        nn.GELU(),
                        nn.Linear(
                            expert_ffn_dim,
                            d_model,
                        ),
                    )
                    for _ in range(num_experts)
                ]
            )

            self._expert_input_dim = self.gru_out_dim

        elif expert_type == "per_expert_gru":
            self.gru = None

            self.gru_out_dim = d_model * 4 if bidirectional else d_model * 2

            self.experts = nn.ModuleList(
                [
                    GRUExpert(
                        d_model=d_model,
                        gru_out_dim=self.gru_out_dim,
                        expert_ffn_dim=expert_ffn_dim,
                        bidirectional=bidirectional,
                        dropout=dropout,
                    )
                    for _ in range(num_experts)
                ]
            )

            self._router_proj = nn.Linear(
                d_model,
                self.gru_out_dim,
            )

            self._expert_input_dim = self.gru_out_dim

        elif expert_type == "mlp_only":
            self.gru = None

            self.gru_out_dim = d_model

            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            d_model,
                            expert_ffn_dim,
                        ),
                        nn.GELU(),
                        nn.Linear(
                            expert_ffn_dim,
                            d_model,
                        ),
                    )
                    for _ in range(num_experts)
                ]
            )

            self._expert_input_dim = d_model

        else:
            msg = f"Unknown expert type: {expert_type}"
            raise ValueError(msg)

        gate_input_dim = (
            self._expert_input_dim + noise_ctx_dim if noise_injection == "concat" else self._expert_input_dim
        )

        self.gate = nn.Linear(
            gate_input_dim,
            num_experts,
        )

        self.film_gamma: nn.Linear | None = None
        self.film_beta: nn.Linear | None = None
        self.noise_bias_proj: nn.Linear | None = None

        if noise_injection == "film" and noise_ctx_dim > 0:
            self.film_gamma = nn.Linear(
                noise_ctx_dim,
                self._expert_input_dim,
            )

            self.film_beta = nn.Linear(
                noise_ctx_dim,
                self._expert_input_dim,
            )

        elif noise_injection == "additive_bias" and noise_ctx_dim > 0:
            self.noise_bias_proj = nn.Linear(
                noise_ctx_dim,
                num_experts,
            )

        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts),
            persistent=True,
        )

        self.dropout = nn.Dropout(dropout)

        self._last_top_idx: Tensor | None = None
        self._last_input_shape: tuple[int, int] | None = None

        self.routing_modifier: RoutingModifier | None = None

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize router and expert linear layers."""

        trunc_normal_init(self.gate.weight)

        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

        for expert in self.experts:
            for module in expert.modules():
                if isinstance(
                    module,
                    nn.Linear,
                ):
                    trunc_normal_init(module.weight)

                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def compute_router_input(
        self,
        x: Tensor,
    ) -> Tensor:
        """Compute the representation used by the router."""

        (
            batch_size,
            sequence_length,
            _,
        ) = x.shape

        num_tokens = batch_size * sequence_length

        if self.expert_type == "shared_gru":
            if self.gru is None:
                msg = "Shared GRU expert backbone is unavailable."
                raise RuntimeError(msg)

            self.gru.flatten_parameters()

            gru_output = cast(
                Tensor,
                self.gru(x)[0],
            )

            gru_output = F.leaky_relu(gru_output)

            gru_output = cast(
                Tensor,
                self.dropout(gru_output),
            )

            return gru_output.reshape(
                num_tokens,
                self.gru_out_dim,
            )

        if self.expert_type == "per_expert_gru":
            if self._router_proj is None:
                msg = "Per-expert GRU router projection is unavailable."
                raise RuntimeError(msg)

            return cast(
                Tensor,
                self._router_proj(
                    x.reshape(
                        num_tokens,
                        self.d_model,
                    )
                ),
            )

        return x.reshape(
            num_tokens,
            self.d_model,
        )

    def forward(
        self,
        x: Tensor,
        noise_ctx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Route tokens through the selected experts."""

        (
            batch_size,
            sequence_length,
            _,
        ) = x.shape

        num_tokens = batch_size * sequence_length

        num_experts = self.num_experts

        flattened = self.compute_router_input(x)

        per_expert_gru_output: list[Tensor] | None = None

        if self.expert_type == "per_expert_gru":
            per_expert_gru_output = []

            for expert in self.experts:
                gru_expert = cast(
                    GRUExpert,
                    expert,
                )

                per_expert_gru_output.append(gru_expert.gru_forward(x))

        with torch.amp.autocast(
            "cuda",
            enabled=False,
        ):
            gate_input = flattened.float()

            if noise_ctx is not None and self.noise_ctx_dim > 0:
                context = (
                    noise_ctx.unsqueeze(1)
                    .expand(
                        batch_size,
                        sequence_length,
                        -1,
                    )
                    .reshape(
                        num_tokens,
                        -1,
                    )
                    .float()
                )

                if self.noise_injection == "concat":
                    gate_input = torch.cat(
                        [
                            gate_input,
                            context,
                        ],
                        dim=-1,
                    )

                elif self.noise_injection == "film":
                    if self.film_gamma is None or self.film_beta is None:
                        msg = "FiLM noise injection layers are unavailable."
                        raise RuntimeError(msg)

                    gamma = cast(
                        Tensor,
                        self.film_gamma(context),
                    )

                    beta = cast(
                        Tensor,
                        self.film_beta(context),
                    )

                    gate_input = gamma * gate_input + beta

            logits = cast(
                Tensor,
                self.gate(gate_input),
            )

            if noise_ctx is not None and self.noise_ctx_dim > 0 and self.noise_injection == "additive_bias":
                if self.noise_bias_proj is None:
                    msg = "Additive noise-bias projection is unavailable."
                    raise RuntimeError(msg)

                context = (
                    noise_ctx.unsqueeze(1)
                    .expand(
                        batch_size,
                        sequence_length,
                        -1,
                    )
                    .reshape(
                        num_tokens,
                        -1,
                    )
                    .float()
                )

                noise_bias = cast(
                    Tensor,
                    self.noise_bias_proj(context),
                )

                logits = logits + noise_bias

            if self.use_expert_bias:
                logits = logits + self.expert_bias

            z_loss = (
                self.z_loss_weight
                * torch.logsumexp(
                    logits,
                    dim=-1,
                )
                .square()
                .mean()
            )

            if self.routing_modifier is not None:
                logits = self.routing_modifier(
                    logits=logits,
                    x=x,
                    gate_input=gate_input,
                    module=self,
                )

            scores = F.softmax(
                logits,
                dim=-1,
            )

            (
                top_scores,
                top_indices,
            ) = scores.topk(
                self.top_k,
                dim=-1,
            )

            weights = top_scores / top_scores.sum(
                dim=-1,
                keepdim=True,
            )

            with torch.no_grad():
                self._last_top_idx = top_indices.detach().cpu()

                self._last_input_shape = (
                    batch_size,
                    sequence_length,
                )

            expert_mask = torch.zeros(
                num_tokens,
                num_experts,
                device=x.device,
                dtype=torch.float32,
            )

            expert_mask.scatter_(
                1,
                top_indices,
                1.0,
            )

            expert_load = expert_mask.detach().mean(dim=0)

            mean_scores = scores.mean(dim=0)

            balance_loss = self.balance_loss_weight * num_experts * (expert_load * mean_scores).sum()

            auxiliary_loss = balance_loss + z_loss

        output = torch.zeros(
            num_tokens,
            self.d_model,
            device=x.device,
            dtype=torch.float32,
        )

        for rank in range(self.top_k):
            for expert_index in range(num_experts):
                mask = (
                    top_indices[
                        :,
                        rank,
                    ]
                    == expert_index
                )

                if not bool(mask.any().item()):
                    continue

                selected_indices = mask.nonzero(as_tuple=True)[0]

                expert = self.experts[expert_index]

                if self.expert_type == "per_expert_gru":
                    if per_expert_gru_output is None:
                        msg = "Per-expert GRU output is unavailable."
                        raise RuntimeError(msg)

                    gru_expert = cast(
                        GRUExpert,
                        expert,
                    )

                    expert_output = gru_expert.mlp_forward(per_expert_gru_output[expert_index][selected_indices])

                else:
                    expert_output = cast(
                        Tensor,
                        expert(flattened[selected_indices]),
                    )

                output[selected_indices] += (
                    weights[
                        selected_indices,
                        rank : rank + 1,
                    ]
                    * expert_output.float()
                )

        with torch.no_grad():
            if self.training and torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    expert_load,
                    op=torch.distributed.ReduceOp.AVG,
                )

            if self.training and self.use_expert_bias:
                target_load = self.top_k / num_experts

                self.expert_bias.add_(self.bias_update_speed * (target_load - expert_load))

        return (
            output.to(x.dtype).reshape(
                batch_size,
                sequence_length,
                self.d_model,
            ),
            auxiliary_loss,
        )


class GRUExpert(nn.Module):
    """Single expert with a GRU backbone and MLP head."""

    def __init__(
        self,
        d_model: int,
        gru_out_dim: int,
        expert_ffn_dim: int,
        bidirectional: bool,
        dropout: float,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            d_model,
            d_model * 2,
            1,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.gru_out_dim = gru_out_dim

        self.act = nn.LeakyReLU()

        self.drop = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(
                gru_out_dim,
                expert_ffn_dim,
            ),
            nn.GELU(),
            nn.Linear(
                expert_ffn_dim,
                d_model,
            ),
        )

    def gru_forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Run the GRU over the full sequence."""

        self.gru.flatten_parameters()

        output = cast(
            Tensor,
            self.gru(x)[0],
        )

        output = cast(
            Tensor,
            self.act(output),
        )

        output = cast(
            Tensor,
            self.drop(output),
        )

        (
            batch_size,
            sequence_length,
            hidden_size,
        ) = output.shape

        return output.reshape(
            batch_size * sequence_length,
            hidden_size,
        )

    def mlp_forward(
        self,
        hidden: Tensor,
    ) -> Tensor:
        """Apply the expert MLP to GRU features."""

        return cast(
            Tensor,
            self.mlp(hidden),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Apply the full expert to a sequence."""

        self.gru.flatten_parameters()

        output = cast(
            Tensor,
            self.gru(x)[0],
        )

        output = cast(
            Tensor,
            self.act(output),
        )

        output = cast(
            Tensor,
            self.drop(output),
        )

        return cast(
            Tensor,
            self.mlp(output),
        )
