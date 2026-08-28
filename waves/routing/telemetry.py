"""
File: telemetry.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Mixture-of-Experts routing telemetry collection for WAVES.

License: MIT License
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any, Literal

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

from waves.models.moe import MoEFFN
from waves.models.transformer import TransformerBlock

RoutingAxis = Literal[
    "time",
    "frequency",
]

_LAYER_PATTERN = re.compile(r"(?:^|\.)TSTransformer\.(\d+)\.")


@dataclass(frozen=True, slots=True)
class RoutingObservation:
    """Aggregated routing statistics from one MoE Transformer invocation."""

    window_index: int
    start_sample: int
    end_sample: int

    layer_index: int
    axis: RoutingAxis

    assignment_count: int

    expert_load: Tensor
    position_load: Tensor


@dataclass(frozen=True, slots=True)
class RoutingTelemetry:
    """Collected routing observations for one enhanced audio sample."""

    observations: tuple[RoutingObservation, ...]

    @property
    def is_empty(self) -> bool:
        """Return whether routing telemetry is empty."""

        return not self.observations

    @property
    def num_experts(self) -> int:
        """Return the number of routed experts."""

        if not self.observations:
            return 0

        first_load = self.observations[0].expert_load

        if first_load.ndim != 1:
            msg = "Routing expert load must be one-dimensional."
            raise ValueError(msg)

        num_experts = int(first_load.shape[0])

        if any(
            observation.expert_load.ndim != 1 or observation.expert_load.shape[0] != num_experts
            for observation in self.observations
        ):
            msg = "Routing telemetry contains observations with different expert counts."
            raise ValueError(msg)

        return num_experts

    @property
    def window_indices(self) -> tuple[int, ...]:
        """Return inference-window indices."""

        return tuple(sorted({observation.window_index for observation in self.observations}))

    def expert_occupancy(self) -> Tensor:
        """Return global selected-assignment share for each expert."""

        num_experts = self.num_experts

        if num_experts == 0:
            return torch.empty(
                0,
                dtype=torch.float32,
            )

        weighted_load = torch.zeros(
            num_experts,
            dtype=torch.float64,
        )

        total_assignments = 0

        for observation in self.observations:
            weighted_load += (
                observation.expert_load.to(
                    dtype=torch.float64,
                )
                * observation.assignment_count
            )

            total_assignments += observation.assignment_count

        if total_assignments <= 0:
            return torch.zeros(
                num_experts,
                dtype=torch.float32,
            )

        return (weighted_load / total_assignments).to(
            dtype=torch.float32,
        )

    def window_expert_load(
        self,
    ) -> dict[int, Tensor]:
        """Return expert load aggregated for each inference window."""

        grouped: dict[
            int,
            list[RoutingObservation],
        ] = defaultdict(list)

        for observation in self.observations:
            grouped[observation.window_index].append(observation)

        return {window_index: _aggregate_expert_load(observations) for window_index, observations in grouped.items()}

    def frequency_expert_load(
        self,
    ) -> Tensor | None:
        """Return expert load by encoded frequency position."""

        observations = [observation for observation in self.observations if observation.axis == "frequency"]

        if not observations:
            return None

        first_position_load = observations[0].position_load

        if first_position_load.ndim != 2:
            msg = "Frequency position load must have two dimensions."
            raise ValueError(msg)

        num_positions = int(first_position_load.shape[0])
        num_experts = self.num_experts

        if num_positions <= 0:
            msg = "Frequency routing must contain at least one position."
            raise ValueError(msg)

        weighted_load = torch.zeros(
            (
                num_positions,
                num_experts,
            ),
            dtype=torch.float64,
        )

        total_weight = 0

        for observation in observations:
            position_load = observation.position_load

            if position_load.shape != (
                num_positions,
                num_experts,
            ):
                msg = "Frequency-routing observations contain incompatible position-load shapes."
                raise ValueError(msg)

            if observation.assignment_count % num_positions != 0:
                msg = "Frequency-routing assignment count is incompatible with the number of positions."
                raise ValueError(msg)

            weight = observation.assignment_count // num_positions

            weighted_load += (
                position_load.to(
                    dtype=torch.float64,
                )
                * weight
            )

            total_weight += weight

        if total_weight <= 0:
            return torch.zeros(
                (
                    num_positions,
                    num_experts,
                ),
                dtype=torch.float32,
            )

        return (weighted_load / total_weight).to(
            dtype=torch.float32,
        )


@dataclass(frozen=True, slots=True)
class _WindowContext:
    """Current sliding-window context."""

    index: int
    start_sample: int
    end_sample: int


def _aggregate_expert_load(
    observations: list[RoutingObservation],
) -> Tensor:
    """Aggregate expert loads using assignment-count weighting."""

    if not observations:
        return torch.empty(
            0,
            dtype=torch.float32,
        )

    first_load = observations[0].expert_load

    if first_load.ndim != 1:
        msg = "Routing expert load must be one-dimensional."
        raise ValueError(msg)

    num_experts = int(first_load.shape[0])

    weighted_load = torch.zeros(
        num_experts,
        dtype=torch.float64,
    )

    total_assignments = 0

    for observation in observations:
        if observation.expert_load.ndim != 1 or observation.expert_load.shape[0] != num_experts:
            msg = "Routing observations contain different expert counts."
            raise ValueError(msg)

        weighted_load += (
            observation.expert_load.to(
                dtype=torch.float64,
            )
            * observation.assignment_count
        )

        total_assignments += observation.assignment_count

    if total_assignments <= 0:
        return torch.zeros(
            num_experts,
            dtype=torch.float32,
        )

    return (weighted_load / total_assignments).to(
        dtype=torch.float32,
    )


def _get_layer_index(
    module_name: str,
) -> int:
    """Extract the zero-based TS-layer index from a module name."""

    match = _LAYER_PATTERN.search(module_name)

    if match is None:
        msg = f"Unable to determine the TS-layer index from module name '{module_name}'."
        raise ValueError(msg)

    return int(match.group(1))


def _get_routing_axis(
    module_name: str,
) -> RoutingAxis:
    """Return routing axis from a Transformer module name."""

    if module_name.endswith("time_transformer"):
        return "time"

    if module_name.endswith("freq_transformer"):
        return "frequency"

    msg = f"Unable to determine the routing axis from module name '{module_name}'."
    raise ValueError(msg)


def _create_observation(
    *,
    module: TransformerBlock,
    layer_index: int,
    axis: RoutingAxis,
    window: _WindowContext,
) -> RoutingObservation | None:
    """Create one aggregated routing observation."""

    if not isinstance(module.ffn, MoEFFN):
        return None

    ffn = module.ffn

    top_idx = ffn._last_top_idx
    input_shape = ffn._last_input_shape

    if top_idx is None or input_shape is None:
        return None

    if top_idx.ndim != 2:
        msg = f"MoE top-index tensor must have two dimensions, got shape={tuple(top_idx.shape)}."
        raise ValueError(msg)

    sequence_count = int(input_shape[0])
    sequence_length = int(input_shape[1])

    num_experts = ffn.num_experts
    top_k = ffn.top_k

    expected_tokens = sequence_count * sequence_length

    if top_idx.shape[0] != expected_tokens:
        msg = (
            "MoE routing token count does not match "
            "the recorded input shape: "
            f"{top_idx.shape[0]} != "
            f"{sequence_count} * {sequence_length}."
        )
        raise ValueError(msg)

    if top_idx.shape[1] != top_k:
        msg = f"MoE top-index width does not match top_k: {top_idx.shape[1]} != {top_k}."
        raise ValueError(msg)

    top_idx = top_idx.to(
        device="cpu",
        dtype=torch.int64,
    ).contiguous()

    assignment_count = int(top_idx.numel())

    flat_indices = top_idx.reshape(-1)

    expert_counts = torch.bincount(
        flat_indices,
        minlength=num_experts,
    ).to(
        dtype=torch.float32,
    )

    expert_load = expert_counts / max(
        assignment_count,
        1,
    )

    routed_indices = top_idx.reshape(
        sequence_count,
        sequence_length,
        top_k,
    )

    position_counts = (
        F.one_hot(
            routed_indices,
            num_classes=num_experts,
        )
        .to(
            dtype=torch.float32,
        )
        .sum(
            dim=(
                0,
                2,
            )
        )
    )

    position_denominator = max(
        sequence_count * top_k,
        1,
    )

    position_load = position_counts / position_denominator

    return RoutingObservation(
        window_index=window.index,
        start_sample=window.start_sample,
        end_sample=window.end_sample,
        layer_index=layer_index,
        axis=axis,
        assignment_count=assignment_count,
        expert_load=expert_load,
        position_load=position_load,
    )


class RoutingTelemetryCollector:
    """Collect routing telemetry from WAVES MoE Transformer blocks."""

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        self._model = model

        self._handles: list[RemovableHandle] = []

        self._observations: list[RoutingObservation] = []

        self._window: _WindowContext | None = None

    @property
    def is_attached(self) -> bool:
        """Return whether routing hooks are currently attached."""

        return bool(self._handles)

    def begin_window(
        self,
        *,
        index: int,
        start_sample: int,
        end_sample: int,
    ) -> None:
        """Set metadata for the next model forward."""

        if index < 0:
            msg = "Routing window index must be non-negative."
            raise ValueError(msg)

        if start_sample < 0:
            msg = "Routing window start sample must be non-negative."
            raise ValueError(msg)

        if end_sample <= start_sample:
            msg = "Routing window end sample must be greater than its start sample."
            raise ValueError(msg)

        self._window = _WindowContext(
            index=index,
            start_sample=start_sample,
            end_sample=end_sample,
        )

    def attach(self) -> None:
        """Attach forward hooks to all routed WAVES Transformer blocks."""

        if self.is_attached:
            return

        for (
            module_name,
            module,
        ) in self._model.named_modules():
            if not isinstance(
                module,
                TransformerBlock,
            ):
                continue

            if not module.use_moe:
                continue

            if not isinstance(module.ffn, MoEFFN):
                continue

            layer_index = _get_layer_index(module_name)

            axis = _get_routing_axis(module_name)

            hook = self._create_hook(
                layer_index=layer_index,
                axis=axis,
            )

            self._handles.append(module.register_forward_hook(hook))

        if not self._handles:
            msg = "No MoE Transformer blocks were found in the loaded WAVES model."
            raise RuntimeError(msg)

    def close(self) -> None:
        """Remove all routing hooks."""

        for handle in self._handles:
            handle.remove()

        self._handles.clear()

        self._window = None

    def snapshot(self) -> RoutingTelemetry:
        """Return an immutable telemetry snapshot."""

        return RoutingTelemetry(
            observations=tuple(self._observations),
        )

    def _create_hook(
        self,
        *,
        layer_index: int,
        axis: RoutingAxis,
    ) -> Callable[
        [
            nn.Module,
            tuple[Any, ...],
            Any,
        ],
        None,
    ]:
        """Create a routing-capture forward hook."""

        def hook(
            module: nn.Module,
            _inputs: tuple[Any, ...],
            _output: Any,
        ) -> None:
            window = self._window

            if window is None:
                return

            if not isinstance(
                module,
                TransformerBlock,
            ):
                return

            observation = _create_observation(
                module=module,
                layer_index=layer_index,
                axis=axis,
                window=window,
            )

            if observation is not None:
                self._observations.append(observation)

        return hook

    def __enter__(
        self,
    ) -> RoutingTelemetryCollector:
        """Attach hooks when entering the context."""

        self.attach()

        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        """Remove hooks when leaving the context."""

        self.close()
