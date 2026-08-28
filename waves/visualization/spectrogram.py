"""
File: spectrogram.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Interactive Plotly spectrogram visualizations for WAVES.

License: MIT License
"""

from typing import cast

import librosa
import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import Tensor

from waves.config import (
    get_config_float,
    get_config_int,
    get_config_str,
)
from waves.localization import get_localized_text

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


def tensor_to_mono_array(
    waveform: Tensor,
) -> FloatArray:
    """Convert a waveform tensor to a contiguous mono NumPy array."""

    waveform = (
        waveform.detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )

    if waveform.ndim == 2:
        if waveform.shape[0] != 1:
            msg = f"Spectrogram visualization requires mono audio, got {waveform.shape[0]} channels."
            raise ValueError(msg)

        waveform = waveform.squeeze(0)

    elif waveform.ndim != 1:
        msg = f"Spectrogram waveform must have one or two dimensions, got shape={tuple(waveform.shape)}."
        raise ValueError(msg)

    if waveform.numel() == 0:
        msg = "Spectrogram waveform is empty."
        raise ValueError(msg)

    if not bool(torch.isfinite(waveform).all().item()):
        msg = "Spectrogram waveform contains non-finite values."
        raise ValueError(msg)

    return np.asarray(
        waveform.numpy(),
        dtype=np.float32,
    )


def compute_mel_power(
    waveform: FloatArray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    num_mels: int,
    max_frequency: float,
) -> FloatArray:
    """Compute a Mel power spectrogram."""

    mel_power = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=num_mels,
        fmin=0.0,
        fmax=max_frequency,
        power=2.0,
        center=True,
        pad_mode="reflect",
    )

    return np.asarray(
        mel_power,
        dtype=np.float32,
    )


def convert_power_pair_to_db(
    noisy_power: FloatArray,
    enhanced_power: FloatArray,
    top_db: float,
) -> tuple[FloatArray, FloatArray]:
    """Convert power spectrograms using one shared reference."""

    noisy_max = float(np.max(noisy_power))

    enhanced_max = float(np.max(enhanced_power))

    minimum_power = float(np.finfo(np.float32).tiny)

    reference_power: float = max(
        noisy_max,
        enhanced_max,
        minimum_power,
    )

    noisy_db = librosa.power_to_db(
        noisy_power,
        ref=reference_power,
        top_db=top_db,
    )

    enhanced_db = librosa.power_to_db(
        enhanced_power,
        ref=reference_power,
        top_db=top_db,
    )

    return (
        np.asarray(
            noisy_db,
            dtype=np.float32,
        ),
        np.asarray(
            enhanced_db,
            dtype=np.float32,
        ),
    )


def compute_power_change_db(
    noisy_power: FloatArray,
    enhanced_power: FloatArray,
    max_delta_db: float,
) -> FloatArray:
    """Compute clipped spectral power change in decibels."""

    minimum_power = float(np.finfo(np.float32).tiny)

    noisy_safe = np.maximum(
        noisy_power,
        minimum_power,
    )

    enhanced_safe = np.maximum(
        enhanced_power,
        minimum_power,
    )

    delta_db = 10.0 * np.log10(enhanced_safe / noisy_safe)

    return np.asarray(
        np.clip(
            delta_db,
            -max_delta_db,
            max_delta_db,
        ),
        dtype=np.float32,
    )


def build_time_bin_indices(
    frame_count: int,
    max_time_bins: int,
) -> IntArray:
    """Return spectrogram frame indices with optional temporal reduction."""

    if frame_count <= 0:
        msg = "Spectrogram does not contain any time frames."
        raise ValueError(msg)

    if max_time_bins <= 0:
        msg = "Visualization maximum time-bin count must be greater than zero."
        raise ValueError(msg)

    if frame_count <= max_time_bins:
        return np.arange(
            frame_count,
            dtype=np.int64,
        )

    indices = np.linspace(
        0,
        frame_count - 1,
        num=max_time_bins,
    )

    return np.unique(np.rint(indices).astype(np.int64))


def apply_axis_style(
    figure: go.Figure,
) -> None:
    """Apply consistent scientific axis styling."""

    axis_line_color = "rgba(128, 128, 128, 0.50)"
    grid_color = "rgba(128, 128, 128, 0.12)"
    spike_color = "rgba(128, 128, 128, 0.70)"

    figure.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=axis_line_color,
        linewidth=1,
        ticks="outside",
        ticklen=5,
        tickcolor=axis_line_color,
        automargin=True,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor=spike_color,
        spikethickness=1,
        title_standoff=10,
    )

    figure.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=axis_line_color,
        linewidth=1,
        ticks="outside",
        ticklen=5,
        tickcolor=axis_line_color,
        automargin=True,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor=spike_color,
        spikethickness=1,
        title_standoff=10,
    )


def create_spectrogram_comparison_figure(
    noisy_waveform: Tensor,
    enhanced_waveform: Tensor,
    sample_rate: int,
    language_index: int,
) -> go.Figure:
    """Create an interactive scientific speech-enhancement spectrogram figure."""

    if sample_rate <= 0:
        msg = "Visualization sample rate must be greater than zero."
        raise ValueError(msg)

    noisy = tensor_to_mono_array(noisy_waveform)

    enhanced = tensor_to_mono_array(enhanced_waveform)

    common_num_samples = min(
        noisy.shape[0],
        enhanced.shape[0],
    )

    if common_num_samples <= 0:
        msg = "No common audio samples are available for visualization."
        raise ValueError(msg)

    noisy = noisy[:common_num_samples]

    enhanced = enhanced[:common_num_samples]

    n_fft = get_config_int(
        "Visualization_SPECTROGRAM_N_FFT",
        2048,
    )

    hop_length = get_config_int(
        "Visualization_SPECTROGRAM_HOP_LENGTH",
        512,
    )

    num_mels = get_config_int(
        "Visualization_SPECTROGRAM_NUM_MELS",
        80,
    )

    configured_max_frequency = get_config_float(
        "Visualization_SPECTROGRAM_MAX_FREQUENCY",
        8000.0,
    )

    top_db = get_config_float(
        "Visualization_SPECTROGRAM_TOP_DB",
        80.0,
    )

    max_time_bins = get_config_int(
        "Visualization_SPECTROGRAM_MAX_TIME_BINS",
        2000,
    )

    max_delta_db = get_config_float(
        "Visualization_SPECTROGRAM_DELTA_MAX_DB",
        20.0,
    )

    colorscale = (
        get_config_str(
            "Visualization_SPECTROGRAM_COLORSCALE",
            "Cividis",
        ).strip()
        or "Cividis"
    )

    if n_fft <= 0:
        msg = "Spectrogram FFT size must be greater than zero."
        raise ValueError(msg)

    if hop_length <= 0:
        msg = "Spectrogram hop length must be greater than zero."
        raise ValueError(msg)

    if num_mels <= 0:
        msg = "Spectrogram Mel-band count must be greater than zero."
        raise ValueError(msg)

    if top_db <= 0.0:
        msg = "Spectrogram dynamic range must be greater than zero."
        raise ValueError(msg)

    if max_delta_db <= 0.0:
        msg = "Spectrogram power-change range must be greater than zero."
        raise ValueError(msg)

    nyquist_frequency = sample_rate / 2.0

    max_frequency = min(
        configured_max_frequency,
        nyquist_frequency,
    )

    if max_frequency <= 0.0:
        msg = "Spectrogram maximum frequency must be greater than zero."
        raise ValueError(msg)

    noisy_power = compute_mel_power(
        waveform=noisy,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        num_mels=num_mels,
        max_frequency=max_frequency,
    )

    enhanced_power = compute_mel_power(
        waveform=enhanced,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        num_mels=num_mels,
        max_frequency=max_frequency,
    )

    frame_count = min(
        noisy_power.shape[-1],
        enhanced_power.shape[-1],
    )

    noisy_power = noisy_power[
        :,
        :frame_count,
    ]

    enhanced_power = enhanced_power[
        :,
        :frame_count,
    ]

    (
        noisy_db,
        enhanced_db,
    ) = convert_power_pair_to_db(
        noisy_power=noisy_power,
        enhanced_power=enhanced_power,
        top_db=top_db,
    )

    delta_db = compute_power_change_db(
        noisy_power=noisy_power,
        enhanced_power=enhanced_power,
        max_delta_db=max_delta_db,
    )

    time_indices = build_time_bin_indices(
        frame_count=frame_count,
        max_time_bins=max_time_bins,
    )

    noisy_db = noisy_db[
        :,
        time_indices,
    ]

    enhanced_db = enhanced_db[
        :,
        time_indices,
    ]

    delta_db = delta_db[
        :,
        time_indices,
    ]

    time_seconds = np.asarray(
        librosa.frames_to_time(
            time_indices,
            sr=sample_rate,
            hop_length=hop_length,
        ),
        dtype=np.float32,
    )

    mel_frequencies_khz = np.asarray(
        librosa.mel_frequencies(
            n_mels=num_mels,
            fmin=0.0,
            fmax=max_frequency,
        )
        / 1000.0,
        dtype=np.float32,
    )

    noisy_title = get_localized_text(
        "Labels_NOISY_AUDIO",
        language_index,
    )

    enhanced_title = get_localized_text(
        "Labels_ENHANCED_AUDIO",
        language_index,
    )

    comparison_title = get_localized_text(
        "Labels_SPECTROGRAM",
        language_index,
    )

    effect_title = get_localized_text(
        "Labels_SPECTROGRAM_EFFECT",
        language_index,
    )

    reduced_label = get_localized_text(
        "Labels_SPECTROGRAM_CHANGE_REDUCED",
        language_index,
    )

    no_change_label = get_localized_text(
        "Labels_SPECTROGRAM_CHANGE_NONE",
        language_index,
    )

    increased_label = get_localized_text(
        "Labels_SPECTROGRAM_CHANGE_INCREASED",
        language_index,
    )

    time_axis_title = get_localized_text(
        "Labels_SPECTROGRAM_TIME",
        language_index,
    )

    frequency_axis_title = get_localized_text(
        "Labels_SPECTROGRAM_FREQUENCY",
        language_index,
    )

    power_title = get_localized_text(
        "Labels_SPECTROGRAM_POWER",
        language_index,
    )

    delta_title = get_localized_text(
        "Labels_SPECTROGRAM_DELTA",
        language_index,
    )

    hover_text = [
        [
            (
                f"{time_axis_title}: "
                f"{float(time_seconds[time_index]):.2f}"
                f"<br>{frequency_axis_title}: "
                f"{float(mel_frequencies_khz[frequency_index]):.2f}"
                f"<br>{noisy_title}: "
                f"{float(noisy_db[frequency_index, time_index]):.2f} dB"
                f"<br>{enhanced_title}: "
                f"{float(enhanced_db[frequency_index, time_index]):.2f} dB"
                f"<br>{delta_title}: "
                f"{float(delta_db[frequency_index, time_index]):+.2f} dB"
            )
            for time_index in range(time_seconds.shape[0])
        ]
        for frequency_index in range(mel_frequencies_khz.shape[0])
    ]

    hover_template = "%{text}<extra></extra>"

    figure = cast(
        go.Figure,
        make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                noisy_title,
                enhanced_title,
                effect_title,
            ],
            shared_yaxes=True,
            horizontal_spacing=0.025,
        ),
    )

    figure.add_trace(
        go.Heatmap(
            x=time_seconds,
            y=mel_frequencies_khz,
            z=noisy_db,
            text=hover_text,
            coloraxis="coloraxis",
            hoverongaps=False,
            hovertemplate=hover_template,
            zsmooth=False,
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Heatmap(
            x=time_seconds,
            y=mel_frequencies_khz,
            z=enhanced_db,
            text=hover_text,
            coloraxis="coloraxis",
            hoverongaps=False,
            hovertemplate=hover_template,
            zsmooth=False,
        ),
        row=1,
        col=2,
    )

    figure.add_trace(
        go.Heatmap(
            x=time_seconds,
            y=mel_frequencies_khz,
            z=delta_db,
            text=hover_text,
            coloraxis="coloraxis2",
            hoverongaps=False,
            hovertemplate=hover_template,
            zsmooth=False,
        ),
        row=1,
        col=3,
    )

    apply_axis_style(figure)

    max_frequency_khz = max_frequency / 1000.0

    figure.update_yaxes(
        title_text=frequency_axis_title,
        range=[
            0.0,
            max_frequency_khz,
        ],
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="",
        showticklabels=False,
        range=[
            0.0,
            max_frequency_khz,
        ],
        row=1,
        col=2,
    )

    figure.update_yaxes(
        title_text="",
        showticklabels=False,
        range=[
            0.0,
            max_frequency_khz,
        ],
        row=1,
        col=3,
    )

    figure.update_xaxes(
        title_text=time_axis_title,
        row=1,
        col=1,
    )

    figure.update_xaxes(
        title_text=time_axis_title,
        row=1,
        col=2,
    )

    figure.update_xaxes(
        title_text=time_axis_title,
        row=1,
        col=3,
    )

    figure.update_annotations(
        font={
            "size": 15,
        },
        yshift=5,
    )

    figure.update_layout(
        title={
            "text": comparison_title,
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "size": 20,
            },
        },
        autosize=True,
        height=500,
        margin={
            "l": 70,
            "r": 34,
            "t": 76,
            "b": 145,
        },
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        hovermode="closest",
        hoverlabel={
            "bgcolor": "rgba(17, 24, 39, 0.96)",
            "bordercolor": "rgba(255, 255, 255, 0.18)",
            "font": {
                "color": "#f9fafb",
                "size": 12,
            },
        },
        dragmode="zoom",
        modebar_remove=[
            "select2d",
            "lasso2d",
            "autoScale2d",
        ],
        uirevision=(f"spectrogram:{sample_rate}:{common_num_samples}"),
        showlegend=False,
        coloraxis={
            "colorscale": colorscale,
            "cmin": -top_db,
            "cmax": 0.0,
            "colorbar": {
                "orientation": "h",
                "title": {
                    "text": power_title,
                    "side": "top",
                },
                "x": 0.32,
                "xanchor": "center",
                "y": -0.18,
                "yanchor": "top",
                "len": 0.48,
                "thickness": 11,
                "outlinewidth": 0,
                "ticks": "outside",
                "ticklen": 4,
                "tickmode": "array",
                "tickvals": [
                    -top_db,
                    -0.75 * top_db,
                    -0.5 * top_db,
                    -0.25 * top_db,
                    0.0,
                ],
            },
        },
        coloraxis2={
            "colorscale": "RdBu_r",
            "cmin": -max_delta_db,
            "cmax": max_delta_db,
            "cmid": 0.0,
            "colorbar": {
                "orientation": "h",
                "title": {
                    "text": delta_title,
                    "side": "top",
                },
                "x": 0.835,
                "xanchor": "center",
                "y": -0.18,
                "yanchor": "top",
                "len": 0.26,
                "thickness": 11,
                "outlinewidth": 0,
                "ticks": "outside",
                "ticklen": 4,
                "tickmode": "array",
                "tickvals": [
                    -max_delta_db,
                    0.0,
                    max_delta_db,
                ],
                "ticktext": [
                    f"{reduced_label}<br>-{max_delta_db:.0f}",
                    f"{no_change_label}<br>0",
                    f"{increased_label}<br>+{max_delta_db:.0f}",
                ],
            },
        },
    )

    return figure
