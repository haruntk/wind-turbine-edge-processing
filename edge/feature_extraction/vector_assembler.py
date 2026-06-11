"""426-dimensional model-input feature vector assembly.

Orchestrates the full per-file processing flow:
  1. Preprocess each channel (detrend, optional bandstop)
  2. Window each group at its native rate with 50 % overlap
  3. Extract features (FFT spectral or time-domain)
  4. Concatenate groups by matching window_id, without hold/resampling

The output is a sequence of ``np.ndarray`` of shape ``(426,)``
in deterministic column order:
  ``[bearing_96 | nacelle_45 | tower_195 | slow_90]``
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from edge.config.channel_groups import ChannelGroup
from edge.feature_extraction.feature_extractor import (
    extract_spectral_features,
    extract_time_features,
)
from edge.signal_processing.fft_processor import apply_hann_window, compute_fft
from edge.signal_processing.preprocessor import preprocess_signal
from edge.signal_processing.windowing import WindowSegment, sliding_window
from edge.utils.logging import get_logger

_LOG = get_logger(__name__)

# Expected total feature count
EXPECTED_FEATURE_DIM = 426
EXPECTED_MODEL_GROUP_ORDER = ("bearing", "nacelle", "tower", "slow")


def _model_group_name(group_name: str) -> str:
    """Return the model/training name for a configured sensor group."""
    if group_name == "tower_tach":
        return "tower"
    return group_name


def _assert_model_feature_order(
    groups: list[ChannelGroup],
    config: dict[str, Any],
) -> None:
    """Assert DB vector order matches the training/model group order."""
    expected_order = tuple(
        config.get("model_input", {}).get(
            "expected_group_order",
            EXPECTED_MODEL_GROUP_ORDER,
        )
    )
    actual_order = tuple(_model_group_name(group.name) for group in groups)
    assert actual_order == expected_order, (
        "Feature group order must match training columns: "
        f"expected {expected_order}, got {actual_order}."
    )

    actual_dim = sum(group.total_features for group in groups)
    assert actual_dim == EXPECTED_FEATURE_DIM, (
        f"Feature dimension must remain {EXPECTED_FEATURE_DIM}, got {actual_dim}."
    )


def _extract_windows_for_group(
    signals: dict[str, np.ndarray],
    group: ChannelGroup,
    config: dict[str, Any],
) -> list[list[WindowSegment]]:
    """Preprocess and window every channel in *group*.

    Returns a list (per channel) of WindowSegment lists.  Channels that
    are missing from *signals* are skipped with a warning.
    """
    overlap_ratio: float = config.get("windowing", {}).get(
        "overlap_ratio", 0.5
    )
    drop_last: bool = config.get("windowing", {}).get("drop_last", True)

    per_channel_windows: list[list[WindowSegment]] = []

    for channel_name in group.channels:
        raw = signals.get(channel_name)
        if raw is None:
            _LOG.warning(
                "Channel '%s' not found in file — skipping.",
                channel_name,
            )
            per_channel_windows.append([])
            continue

        processed = preprocess_signal(
            raw, config, sampling_rate_hz=group.sampling_rate_hz
        )
        windows = sliding_window(
            processed,
            window_size_samples=group.window_size_samples,
            overlap_ratio=overlap_ratio,
            drop_last=drop_last,
        )
        per_channel_windows.append(windows)

    return per_channel_windows


def _features_for_window(
    window: WindowSegment,
    group: ChannelGroup,
) -> OrderedDict[str, float]:
    """Extract features from a single window for a given group type."""
    if group.representation == "fft":
        windowed = apply_hann_window(window.values)
        freqs, amplitudes, power = compute_fft(
            windowed, group.sampling_rate_hz
        )
        return extract_spectral_features(
            freqs=freqs,
            amplitudes=amplitudes,
            power=power,
            band_edges_hz=list(group.band_energy_edges_hz),
            sampling_rate_hz=group.sampling_rate_hz,
            n_signal_samples=len(window.values),
        )
    else:
        return extract_time_features(window.values)


def _compute_group_feature_table(
    signals: dict[str, np.ndarray],
    group: ChannelGroup,
    config: dict[str, Any],
) -> list[np.ndarray]:
    """Compute the feature table for an entire group.

    Returns a list of 1-D arrays, one per window position.  Each array
    has ``group.total_features`` elements (channels × features_per_ch).
    """
    per_channel_windows = _extract_windows_for_group(signals, group, config)

    # Determine number of window positions from channels that have data
    n_windows = 0
    for ch_windows in per_channel_windows:
        if ch_windows:
            n_windows = max(n_windows, len(ch_windows))

    if n_windows == 0:
        _LOG.warning(
            "Group '%s' produced no windows — returning zeros.",
            group.name,
        )
        return []

    n_feats = group.features_per_channel
    n_channels = len(group.channels)

    table: list[np.ndarray] = []
    for wid in range(n_windows):
        row = np.zeros(n_channels * n_feats, dtype=np.float64)
        for ch_idx, ch_windows in enumerate(per_channel_windows):
            if wid < len(ch_windows):
                feats = _features_for_window(ch_windows[wid], group)
                values = list(feats.values())
                offset = ch_idx * n_feats
                row[offset : offset + len(values)] = values
        table.append(row)

    return table


def get_feature_step_seconds(
    groups: list[ChannelGroup],
    config: dict[str, Any],
) -> float:
    """Return the shared model-vector timestamp step in seconds."""
    overlap_ratio: float = config.get("windowing", {}).get(
        "overlap_ratio", 0.5
    )
    steps = [
        group.window_size_seconds * (1.0 - overlap_ratio)
        for group in groups
    ]
    rounded_steps = {round(step, 9) for step in steps}
    assert len(rounded_steps) == 1, (
        "All groups must use the same window step before DB insertion; "
        f"got {steps}."
    )
    configured_step = config.get("processing", {}).get("output_step_seconds")
    if configured_step is not None:
        assert round(float(configured_step), 9) == round(float(steps[0]), 9), (
            "processing.output_step_seconds must match the configured "
            f"window step: expected {steps[0]}, got {configured_step}."
        )
    return float(steps[0])


def _get_scaler_path(config: dict[str, Any]) -> str:
    """Read the saved training scaler path from config or environment."""
    model_cfg = config.get("model_input", {})
    scaler_path = str(model_cfg.get("scaler_path") or "").strip()
    if scaler_path:
        return scaler_path
    return os.environ.get("FEATURE_SCALER_PATH", "").strip()


@lru_cache(maxsize=4)
def _load_training_scaler(scaler_path: str) -> Any:
    """Load the fitted RobustScaler without fitting anything at the edge."""
    path = Path(scaler_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Training scaler file not found: {path}")

    try:
        import joblib

        return joblib.load(path)
    except ImportError:
        with path.open("rb") as fh:
            return pickle.load(fh)


def _fill_missing_like_training(matrix: np.ndarray) -> np.ndarray:
    """Apply inf->nan, ffill, bfill, fillna(0) column-wise."""
    filled = np.asarray(matrix, dtype=np.float64).copy()
    filled[~np.isfinite(filled)] = np.nan

    if filled.size == 0:
        return filled

    row_idx = np.arange(filled.shape[0])[:, None]
    col_idx = np.arange(filled.shape[1])

    # Same intent as pandas ffill: each NaN takes the previous valid row.
    valid = ~np.isnan(filled)
    forward_idx = np.where(valid, row_idx, 0)
    np.maximum.accumulate(forward_idx, axis=0, out=forward_idx)
    filled = filled[forward_idx, col_idx]

    # Same intent as pandas bfill: remaining leading NaNs take next valid row.
    valid = ~np.isnan(filled)
    backward_idx = np.where(valid, row_idx, filled.shape[0] - 1)
    backward_idx = np.minimum.accumulate(backward_idx[::-1], axis=0)[::-1]
    filled = filled[backward_idx, col_idx]

    return np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_vectors_for_model_db(
    vectors: list[np.ndarray],
    config: dict[str, Any],
) -> list[np.ndarray]:
    """Make raw 426-dim vectors model-ready before they are written to DB.

    This mirrors training preprocessing exactly at the edge boundary:
    missing-value fill -> saved RobustScaler.transform -> clip. The scaler is
    never fit here; inference can therefore read DB rows without extra work.
    """
    if not vectors:
        return []

    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(
            "Expected a 2-D feature matrix with "
            f"{EXPECTED_FEATURE_DIM} columns, got shape {matrix.shape}."
        )

    scaler_path = _get_scaler_path(config)
    if not scaler_path:
        raise ValueError(
            "Missing training scaler path. Set model_input.scaler_path "
            "or FEATURE_SCALER_PATH."
        )

    scaler = _load_training_scaler(scaler_path)
    n_features = getattr(scaler, "n_features_in_", EXPECTED_FEATURE_DIM)
    assert n_features == EXPECTED_FEATURE_DIM, (
        "Training scaler feature count does not match DB vector width: "
        f"expected {EXPECTED_FEATURE_DIM}, got {n_features}."
    )

    matrix = _fill_missing_like_training(matrix)
    scaled = scaler.transform(matrix)

    model_cfg = config.get("model_input", {})
    clip_min = float(model_cfg.get("clip_min", -10.0))
    clip_max = float(model_cfg.get("clip_max", 10.0))
    clipped = np.clip(scaled, clip_min, clip_max)

    return [row.astype(np.float64, copy=False) for row in clipped]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def assemble_feature_vectors(
    signals: dict[str, np.ndarray],
    groups: list[ChannelGroup],
    config: dict[str, Any],
) -> list[np.ndarray]:
    """Assemble one 426-dim feature vector per analysis window.

    Processing steps:
      1. For each sensor group: preprocess → window (50 % overlap) →
         extract features → produce per-window feature rows.
      2. Assert group order is the model/training order.
      3. Concatenate rows with the same window_id. No 1 Hz hold/resampling is
         performed, so each DB row remains an original 5 s / 50 % window.

    Parameters
    ----------
    signals:
        Channel-name → 1-D signal array mapping from a single .mat file.
    groups:
        Ordered list of :class:`ChannelGroup` objects.
    config:
        Full pipeline configuration dict.

    Returns
    -------
    list[np.ndarray]
        List of 1-D float64 arrays, each of length 426.
        One vector per window step.
    """
    _assert_model_feature_order(groups, config)

    # --- Step 1: Compute per-group feature tables ---
    group_tables: list[list[np.ndarray]] = []

    for group in groups:
        table = _compute_group_feature_table(signals, group, config)
        group_tables.append(table)

        _LOG.debug(
            "Group '%s': %d windows, %d features/window, "
            "step %.3f s",
            group.name,
            len(table),
            group.total_features if table else 0,
            get_feature_step_seconds([group], config),
        )

    # --- Step 2: Validate common window_id range ---
    if not all(group_tables):
        _LOG.error("No features computed for any group.")
        return []

    table_lengths = [len(table) for table in group_tables]
    assert len(set(table_lengths)) == 1, (
        "All groups must produce the same number of windows; "
        f"got {dict(zip([group.name for group in groups], table_lengths))}."
    )
    n_windows = table_lengths[0]

    _LOG.info(
        "Assembling %d feature vectors at %.3f s/window step.",
        n_windows,
        get_feature_step_seconds(groups, config),
    )

    # --- Step 3: Concatenate matching window_id rows ---
    vectors: list[np.ndarray] = []
    for window_id in range(n_windows):
        parts = [table[window_id] for table in group_tables]
        vector = np.concatenate(parts)

        if len(vector) != EXPECTED_FEATURE_DIM:
            _LOG.error(
                "Feature vector dimension mismatch at window_id=%d: "
                "expected %d, got %d.",
                window_id,
                EXPECTED_FEATURE_DIM,
                len(vector),
            )

        vectors.append(vector)

    return vectors
