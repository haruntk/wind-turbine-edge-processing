"""Sensor group definitions for multi-rate feature extraction.

Each ``ChannelGroup`` represents a set of sensors that share the same
native sampling rate and are processed with a common windowing strategy.
The four groups match the Fraunhofer LBF dataset structure:

- **bearing** (6 ch @ 74 kHz, 5 s window, FFT → 16 features/ch)
- **nacelle** (3 ch @ 37 kHz, 5 s window, FFT → 15 features/ch)
- **tower_tach** (13 ch @ 2.96 kHz, 5 s window, FFT → 15 features/ch)
- **slow** (6 ch @ 1.48 kHz, 5 s window, Time-domain → 15 features/ch)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChannelGroup:
    """A set of channels sharing the same native sampling rate.

    Attributes
    ----------
    name:
        Human-readable group identifier (e.g. ``"bearing"``).
    sampling_rate_hz:
        Native sampling frequency in Hz.
    channels:
        Tuple of channel names belonging to this group.
    representation:
        Feature extraction domain: ``"fft"`` or ``"time"``.
    window_size_seconds:
        Duration of each analysis window in seconds.
    band_energy_edges_hz:
        Frequency band boundaries for band-energy features.
        Only relevant when ``representation == "fft"``.
    """

    name: str
    sampling_rate_hz: float
    channels: tuple[str, ...]
    representation: str
    window_size_seconds: float
    band_energy_edges_hz: tuple[float, ...] = ()

    @property
    def window_size_samples(self) -> int:
        """Window size in samples for this group's sampling rate."""
        return int(round(self.window_size_seconds * self.sampling_rate_hz))

    @property
    def features_per_channel(self) -> int:
        """Number of features extracted per channel.

        - FFT groups: 9 spectral statistics + len(band_edges) - 1 bands
        - Time groups: 15 time-domain statistics
        """
        if self.representation == "time":
            return 15
        return 9 + max(0, len(self.band_energy_edges_hz) - 1)

    @property
    def total_features(self) -> int:
        """Total feature count for this group (channels × features_per_ch)."""
        return len(self.channels) * self.features_per_channel


def load_groups_from_config(config: dict[str, Any]) -> list[ChannelGroup]:
    """Build a list of :class:`ChannelGroup` from the config dict.

    Parameters
    ----------
    config:
        Full pipeline configuration with a ``sensor_groups`` section.

    Returns
    -------
    list[ChannelGroup]
        Ordered list of sensor groups.
    """
    groups_cfg = config.get("sensor_groups", {})
    if not groups_cfg:
        raise ValueError("No sensor_groups defined in configuration.")

    groups: list[ChannelGroup] = []
    for name, gcfg in groups_cfg.items():
        groups.append(
            ChannelGroup(
                name=name,
                sampling_rate_hz=float(gcfg["sampling_rate_hz"]),
                channels=tuple(gcfg["channels"]),
                representation=str(gcfg.get("representation", "fft")),
                window_size_seconds=float(gcfg.get("window_size_seconds", 1.0)),
                band_energy_edges_hz=tuple(
                    gcfg.get("band_energy_edges_hz", [])
                ),
            )
        )
    return groups
