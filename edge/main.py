"""Edge processing pipeline orchestrator.

Entry point for the Jetson edge device.  Processes raw ``.mat`` files
through the full pipeline:

    RAW DATA → PREPROCESS → WINDOW (50 % overlap) → FFT →
    FEATURE EXTRACTION → SCALED 426-dim VECTOR (2.5 s step) → TimescaleDB

Usage::

    python -m edge.main --config edge/config/config.yaml
    python -m edge.main --config edge/config/config.yaml --mode batch
    python -m edge.main --config edge/config/config.yaml --dry-run

Architecture note:
    The database is the **single source of truth**.  This pipeline only
    computes and sends.  All downstream processing (LSTM Autoencoder,
    Dashboard) reads exclusively from TimescaleDB.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from edge.config.channel_groups import load_groups_from_config
from edge.config.settings import load_config, validate_config
from edge.data_ingestion.data_loader import load_batch
from edge.data_sender.db_client import TimescaleDBClient
from edge.data_sender.mqtt_publisher import MQTTPublisher
from edge.feature_extraction.vector_assembler import (
    EXPECTED_FEATURE_DIM,
    assemble_feature_vectors,
    get_feature_step_seconds,
    prepare_vectors_for_model_db,
)
from edge.utils.logging import get_logger, setup_logging

_LOG = get_logger(__name__)

# Graceful shutdown flag
_SHUTDOWN_REQUESTED = False


def _handle_signal(signum: int, frame: Any) -> None:
    """Set shutdown flag on SIGINT / SIGTERM."""
    global _SHUTDOWN_REQUESTED  # noqa: PLW0603
    _SHUTDOWN_REQUESTED = True
    _LOG.info("Shutdown signal received (sig=%d). Finishing current file…", signum)


def _infer_scenario_label(file_path: Path) -> str:
    """Infer a scenario label from the file's directory structure.

    Maps Fraunhofer LBF directory names to human-readable labels.
    Falls back to ``"unknown"`` if no match is found.

    Real folder structures observed:
      data/Bearing/Bearing/InnerRace/*.mat
      data/Bearing/Bearing/OutterRace/*.mat
      data/Bearing/Bearing/RollerElement/*.mat
      data/Bearing/Bearing/InnerRace_MassImbalance/*.mat
      data/Healthy/Healthy/*.mat
      data/Imbalance_6g/Imbalance_6g/*.mat
      data/Aerodynamic_5_Degrees/5_Degrees/*.mat
    """
    # Use the full path string for matching (case-insensitive)
    path_str = str(file_path).lower().replace("\\", "/")

    # Combined fault — must check before individual bearing/imbalance
    if "innerrace_massimbalance" in path_str or "inner_race_massimbalance" in path_str:
        return "combined_fault"

    # Bearing faults
    if "innerrace" in path_str or "inner_race" in path_str:
        return "bearing_inner_race"
    if "outterrace" in path_str or "outer_race" in path_str:
        return "bearing_outer_race"
    if "rollerelement" in path_str or "roller_element" in path_str:
        return "bearing_roller_element"

    # Aerodynamic imbalance
    if "aerodynamic" in path_str or "5_degrees" in path_str or "fivedegree" in path_str:
        return "aerodynamic"

    # Mass imbalance (various weights)
    if "imbalance" in path_str:
        # Try to extract weight/level info from path
        for marker in ("6g", "10g", "15g", "20g", "30g"):
            if marker in path_str:
                return f"imbalance_{marker}"
        return "imbalance"

    # Healthy baseline
    if "healthy" in path_str:
        return "healthy"

    return "unknown"


def _process_file(
    file_path: Path,
    signals: dict[str, np.ndarray],
    groups: list,
    config: dict[str, Any],
    sender: TimescaleDBClient | MQTTPublisher | None,
    dry_run: bool,
    file_index: int = 0,
) -> int:
    """Process a single .mat file and send vectors via the active transport.

    Returns the number of feature vectors produced.
    """
    scenario_label = _infer_scenario_label(file_path)

    _LOG.info(
        "Processing %s (%d channels, scenario=%s)",
        file_path.name,
        len(signals),
        scenario_label,
    )

    vectors = assemble_feature_vectors(signals, groups, config)

    if not vectors:
        _LOG.warning("No feature vectors produced for %s.", file_path.name)
        return 0

    bad_dims = [v for v in vectors if len(v) != EXPECTED_FEATURE_DIM]
    if bad_dims:
        _LOG.error(
            "%d vector(s) have wrong dimension (expected %d). Skipping file.",
            len(bad_dims),
            EXPECTED_FEATURE_DIM,
        )
        return 0

    _LOG.info(
        "Produced %d × %d feature vectors for %s.",
        len(vectors),
        EXPECTED_FEATURE_DIM,
        file_path.name,
    )

    if dry_run:
        _LOG.info("[DRY RUN] Skipping send.")
        return len(vectors)

    # Persist only model-ready values: train-time fill -> saved scaler -> clip.
    # Inference can then read DB rows directly without scaler/clip/dedup logic.
    vectors = prepare_vectors_for_model_db(vectors, config)

    # Build records — each file gets a unique base time to avoid timestamp collisions
    feature_step_seconds = get_feature_step_seconds(groups, config)
    base_time = datetime.now(tz=timezone.utc).replace(microsecond=0) - timedelta(
        seconds=len(vectors) * feature_step_seconds
    ) + timedelta(seconds=file_index * 1000)

    transport_cfg = config.get("mqtt", config.get("database", {}))
    batch_size: int = int(transport_cfg.get("batch_size", 50))

    use_mqtt = isinstance(sender, MQTTPublisher)
    mqtt_records: list[tuple[datetime, np.ndarray, str]] = []
    db_records: list[tuple[datetime, np.ndarray]] = []

    for idx, vector in enumerate(vectors):
        ts = base_time + timedelta(seconds=idx * feature_step_seconds)
        if use_mqtt:
            mqtt_records.append((ts, vector, scenario_label))
            if len(mqtt_records) >= batch_size:
                sender.publish_feature_vectors_batch(mqtt_records)
                mqtt_records.clear()
        else:
            db_records.append((ts, vector))
            if len(db_records) >= batch_size:
                sender.insert_feature_vectors_batch(db_records)
                db_records.clear()

    # Flush remaining
    if sender is not None:
        if use_mqtt and mqtt_records:
            sender.publish_feature_vectors_batch(mqtt_records)
        elif not use_mqtt and db_records:
            sender.insert_feature_vectors_batch(db_records)

    transport = "MQTT" if use_mqtt else "TimescaleDB"
    _LOG.info("Sent %d vectors via %s for %s.", len(vectors), transport, file_path.name)

    return len(vectors)


def run_pipeline(config_path: str, mode: str = "batch", dry_run: bool = False) -> None:
    """Run the edge processing pipeline.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    mode:
        Processing mode: ``"batch"`` (process all files once).
    dry_run:
        If ``True``, compute features but skip DB insertion.
    """
    # --- Load and validate config ---
    config = load_config(config_path)
    validate_config(config)

    # --- Setup logging ---
    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file"),
    )

    _LOG.info("=" * 60)
    _LOG.info("Wind Turbine Edge Processing Pipeline")
    _LOG.info("=" * 60)
    _LOG.info("Mode: %s | Dry run: %s", mode, dry_run)
    _LOG.info("Config: %s", config.get("_config_path", config_path))

    # --- Load sensor groups ---
    groups = load_groups_from_config(config)
    total_features = sum(g.total_features for g in groups)
    _LOG.info(
        "Sensor groups: %s → %d total features",
        [g.name for g in groups],
        total_features,
    )
    if total_features != EXPECTED_FEATURE_DIM:
        _LOG.warning(
            "Total features (%d) ≠ expected (%d). Check config!",
            total_features,
            EXPECTED_FEATURE_DIM,
        )

    turbine_id: str = config.get("turbine", {}).get("id", "WT-001")
    data_dir: str = config.get("data_source", {}).get(
        "mat_directory", "./data"
    )
    file_pattern: str = config.get("data_source", {}).get(
        "file_pattern", "*.mat"
    )

    # --- Setup transport (MQTT preferred, direct DB as fallback) ---
    sender: MQTTPublisher | TimescaleDBClient | None = None
    if not dry_run:
        if "mqtt" in config:
            try:
                sender = MQTTPublisher.from_config(config, turbine_id=turbine_id)
                sender.connect()
                _LOG.info("Transport: MQTT → %s", config["mqtt"]["host"])
            except Exception:
                _LOG.error("Cannot connect to MQTT broker. Use --dry-run to skip.", exc_info=True)
                return
        elif "database" in config:
            try:
                sender = TimescaleDBClient.from_config(config)
                sender.connect()
                sender.ensure_schema()
                _LOG.info("Transport: direct TimescaleDB → %s", config["database"]["host"])
            except Exception:
                _LOG.error("Cannot connect to TimescaleDB. Use --dry-run to skip.", exc_info=True)
                return

    # --- Process files ---
    total_files = 0
    total_vectors = 0
    t_start = time.perf_counter()

    try:
        for file_path, signals in load_batch(data_dir, file_pattern):
            if _SHUTDOWN_REQUESTED:
                _LOG.info("Shutdown requested — stopping.")
                break

            try:
                n = _process_file(
                    file_path=file_path,
                    signals=signals,
                    groups=groups,
                    config=config,
                    sender=sender,
                    dry_run=dry_run,
                    file_index=total_files,
                )
                total_files += 1
                total_vectors += n
            except Exception:
                _LOG.error(
                    "Error processing %s — skipping.",
                    file_path.name,
                    exc_info=True,
                )

    finally:
        if sender is not None:
            sender.disconnect()

    elapsed = time.perf_counter() - t_start
    _LOG.info("=" * 60)
    _LOG.info("Pipeline complete.")
    _LOG.info(
        "Files: %d | Vectors: %d | Time: %.1f s",
        total_files,
        total_vectors,
        elapsed,
    )
    _LOG.info("=" * 60)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Wind Turbine Edge Processing Pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="edge/config/config.yaml",
        help="Path to YAML config file (default: edge/config/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch"],
        default="batch",
        help="Processing mode (default: batch)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute features without inserting into database.",
    )
    args = parser.parse_args()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    run_pipeline(
        config_path=args.config,
        mode=args.mode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
