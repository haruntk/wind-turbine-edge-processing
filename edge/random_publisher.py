#!/usr/bin/env python3
"""Random feature vector simulator publisher.

Runs on the Jetson edge device. Scans the raw data directory,
randomly selects .mat files, computes the scaled 426-dim feature
vectors, and streams them one-by-one via MQTT with a configurable
delay to simulate real-time sensor ingestion.

Usage:
    python edge/random_publisher.py --config edge/config/config.yaml --interval 2.5
"""

from __future__ import annotations

import argparse
import random
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# Set project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from edge.config.channel_groups import load_groups_from_config
from edge.config.settings import load_config, validate_config
from edge.data_ingestion.mat_loader import load_mat_file
from edge.data_sender.mqtt_publisher import MQTTPublisher
from edge.feature_extraction.vector_assembler import (
    EXPECTED_FEATURE_DIM,
    assemble_feature_vectors,
    prepare_vectors_for_model_db,
    get_feature_step_seconds,
)
from edge.main import _infer_scenario_label
from edge.utils.logging import get_logger, setup_logging

_LOG = get_logger("random_publisher")
_SHUTDOWN_REQUESTED = False


def _handle_signal(signum: int, frame: Any) -> None:
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    _LOG.info("Shutdown requested (sig=%d). Stopping simulator...", signum)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wind Turbine Feature Vector Simulator")
    parser.add_argument(
        "--config",
        type=str,
        default="edge/config/config.yaml",
        help="Path to YAML config file (default: edge/config/config.yaml)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Delay between publishing different files in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    # Register signal handlers for clean exit
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    # Load and validate config
    config = load_config(args.config)
    validate_config(config)

    # Setup logging
    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file"),
    )

    _LOG.info("=" * 60)
    _LOG.info("Wind Turbine Feature Vector Random Publisher Simulator (Instant Batch)")
    _LOG.info("=" * 60)
    _LOG.info(f"Delay between file loads: {args.interval} seconds")

    groups = load_groups_from_config(config)
    turbine_id = config.get("turbine", {}).get("id", "WT-001")
    data_dir = config.get("data_source", {}).get("mat_directory", "./data")
    file_pattern = config.get("data_source", {}).get("file_pattern", "*.mat")

    # Locate all mat files
    root_path = Path(data_dir).resolve()
    if not root_path.is_dir():
        _LOG.error(f"Data directory does not exist: {root_path}")
        sys.exit(1)

    mat_files = list(root_path.rglob(file_pattern))
    if not mat_files:
        _LOG.error(f"No files matching '{file_pattern}' found in {root_path}")
        sys.exit(1)

    _LOG.info(f"Found {len(mat_files)} source files in {root_path} for random simulation.")

    # Initialize MQTT publisher
    if "mqtt" not in config:
        _LOG.error("MQTT section missing in configuration config.yaml. Cannot stream.")
        sys.exit(1)

    try:
        sender = MQTTPublisher.from_config(config, turbine_id=turbine_id)
        sender.connect()
        _LOG.info(f"Connected to MQTT broker at {config['mqtt']['host']}:{config['mqtt'].get('port', 8883)}")
    except Exception as e:
        _LOG.error(f"Failed to connect to MQTT broker: {e}")
        sys.exit(1)

    # Initialize rolling simulation clock at current time
    current_sim_time = datetime.now(tz=timezone.utc).replace(microsecond=0)

    try:
        while not _SHUTDOWN_REQUESTED:
            # Select a random file
            file_path = random.choice(mat_files)
            scenario_label = _infer_scenario_label(file_path)
            
            _LOG.info(f"Processing and publishing file: {file_path.name} | Scenario label: {scenario_label}")
            
            try:
                signals = load_mat_file(file_path)
                vectors = assemble_feature_vectors(signals, groups, config)
                
                if not vectors:
                    _LOG.warning(f"No vectors produced from {file_path.name}. Skipping file.")
                    continue
                
                # Scaler & Clip
                vectors = prepare_vectors_for_model_db(vectors, config)
                
                feature_step_seconds = get_feature_step_seconds(groups, config)
                
                # Generate chronological records starting from current_sim_time
                records = []
                for idx, vector in enumerate(vectors):
                    ts = current_sim_time + timedelta(seconds=idx * feature_step_seconds)
                    records.append((ts, vector, scenario_label))
                
                # Publish as fast as possible in batches
                batch_size = int(config.get("mqtt", {}).get("batch_size", 50))
                for i in range(0, len(records), batch_size):
                    if _SHUTDOWN_REQUESTED:
                        break
                    batch = records[i : i + batch_size]
                    sender.publish_feature_vectors_batch(batch)
                
                _LOG.info(f"Successfully published {len(vectors)} vectors | Time Range: {records[0][0].strftime('%H:%M:%S')} - {records[-1][0].strftime('%H:%M:%S')} | Scenario: {scenario_label}")
                
                # Advance simulation clock so next file follows chronologically
                file_duration = len(vectors) * feature_step_seconds
                current_sim_time += timedelta(seconds=file_duration)
                
                # Small delay between files to avoid saturating CPU
                time.sleep(args.interval)
                    
            except Exception as e:
                _LOG.error(f"Error processing simulated file {file_path.name}: {e}", exc_info=True)
                time.sleep(5)  # Back off before selecting next file
                
    finally:
        sender.disconnect()
        _LOG.info("Simulator publisher stopped.")


if __name__ == "__main__":
    main()
