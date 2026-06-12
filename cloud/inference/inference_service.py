#!/usr/bin/env python3
"""LSTM-AE Anomaly Detection Cloud Inference Service.

Reads feature vectors from TimescaleDB (feature_vectors table),
constructs sequences of length 20 and stride 5, and performs
unsupervised inference using a trained LSTM Autoencoder model.
Writes prediction results back to TimescaleDB (anomaly_results table).

Can be run in:
  * batch mode: Process all historical features and exit.
  * stream mode: Continuously poll database for new feature vectors.

Usage:
  python cloud/inference/inference_service.py --mode stream
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch

# Dynamically add train_model/src to sys.path to import LSTMAutoencoder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "train_model" / "src"))

try:
    from fraunhofer_ad.models.lstm_autoencoder import LSTMAutoencoder
except ImportError:
    # Fallback definition in case paths or python environment are restricted
    import torch.nn as nn

    class LSTMAutoencoder(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            latent_size: int,
            num_layers: int,
            dropout: float = 0.0,
            bidirectional: bool = False,
        ) -> None:
            super().__init__()
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.latent_size = latent_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional

            encoder_out_size = hidden_size * 2 if bidirectional else hidden_size
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.to_latent = nn.Linear(encoder_out_size, latent_size)
            self.latent_norm = nn.LayerNorm(latent_size)
            self.latent_dropout = nn.Dropout(p=dropout)
            self.latent_to_h0 = nn.Linear(latent_size, num_layers * hidden_size)
            self.latent_to_c0 = nn.Linear(latent_size, num_layers * hidden_size)
            self.decoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout,
                batch_first=True,
            )
            self.output_layer = nn.Linear(hidden_size, input_size)

        def encode(self, inputs: torch.Tensor) -> torch.Tensor:
            encoded_sequence, _ = self.encoder(inputs)
            if self.bidirectional:
                fwd = encoded_sequence[:, :, : self.hidden_size]
                bwd = encoded_sequence[:, :, self.hidden_size :]
                pooled = torch.cat([fwd.mean(dim=1), bwd.mean(dim=1)], dim=-1)
            else:
                pooled = encoded_sequence.mean(dim=1)
            latent = self.to_latent(pooled)
            latent = self.latent_norm(latent)
            latent = self.latent_dropout(latent)
            return latent

        def decode(self, latent: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len, _ = inputs.size()
            h0 = self.latent_to_h0(latent).view(batch_size, self.num_layers, self.hidden_size)
            h0 = h0.permute(1, 0, 2).contiguous()
            c0 = self.latent_to_c0(latent).view(batch_size, self.num_layers, self.hidden_size)
            c0 = c0.permute(1, 0, 2).contiguous()

            if self.training:
                reversed_inputs = torch.flip(inputs, dims=[1])
                decoded_sequence, _ = self.decoder(reversed_inputs, (h0, c0))
                reversed_output = self.output_layer(decoded_sequence)
                return torch.flip(reversed_output, dims=[1])
            else:
                hidden = (h0, c0)
                current_input = torch.zeros(
                    batch_size, 1, self.input_size, device=inputs.device, dtype=inputs.dtype
                )
                outputs: list[torch.Tensor] = []
                for _ in range(seq_len):
                    out, hidden = self.decoder(current_input, hidden)
                    pred = self.output_layer(out)
                    outputs.append(pred)
                    current_input = pred
                return torch.flip(torch.cat(outputs, dim=1), dims=[1])

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            latent = self.encode(inputs)
            return self.decode(latent, inputs)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference_service")

# Consts
SEQUENCE_LENGTH = 20
SEQUENCE_STRIDE = 5


class InferenceService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.model = None
        self.threshold = None
        self.conn = None
        self.running = True

        # Load threshold and model
        self.load_metadata_and_threshold()
        self.load_model()

    def load_metadata_and_threshold(self) -> None:
        """Loads static threshold from metrics.json or fallback settings."""
        metrics_path = Path(self.args.metrics_path)
        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    data = json.load(f)
                # Check if threshold is dict or single float
                val = data.get("threshold")
                if isinstance(val, dict):
                    self.threshold = float(val.get("value", 2.55979299545288))
                else:
                    self.threshold = float(val) if val is not None else 2.55979299545288
                logger.info(f"Loaded threshold from {metrics_path}: {self.threshold:.6f}")
                return
            except Exception as e:
                logger.warning(f"Error loading {metrics_path}: {e}. Trying fallback.")

        # Fallback to run_metadata.json
        metadata_path = PROJECT_ROOT / "bucket" / "run_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    data = json.load(f)
                self.threshold = float(data.get("threshold_value", 2.55979299545288))
                logger.info(f"Loaded threshold from run_metadata.json: {self.threshold:.6f}")
                return
            except Exception as e:
                logger.warning(f"Error loading {metadata_path}: {e}")

        # Static default fallback
        self.threshold = 2.55979299545288
        logger.info(f"Using default hardcoded threshold: {self.threshold:.6f}")

    def load_model(self) -> None:
        """Loads the trained LSTM-AE model configuration and state dict."""
        model_path = Path(self.args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        logger.info(f"Loading model checkpoint from {model_path} on device {self.device}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Instantiate model using saved config parameters
        cfg = checkpoint["model_config"]
        self.model = LSTMAutoencoder(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            latent_size=cfg["latent_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg.get("dropout", 0.0),
            bidirectional=cfg.get("bidirectional", False),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully. Trainable parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def connect_db(self) -> None:
        """Connects to TimescaleDB database with retries."""
        host = os.environ.get("DB_HOST", "127.0.0.1")
        port = int(os.environ.get("DB_PORT", 5433))
        name = os.environ.get("DB_NAME", "windguard")
        user = os.environ.get("DB_USER", "windguard_user")
        pwd = os.environ.get("DB_PASSWORD", "ceren123")

        logger.info(f"Connecting to TimescaleDB at {host}:{port}/{name} ...")
        while self.running:
            try:
                self.conn = psycopg2.connect(
                    host=host,
                    port=port,
                    dbname=name,
                    user=user,
                    password=pwd,
                )
                self.conn.autocommit = True
                logger.info("Connected to database successfully.")
                self.verify_schema()
                break
            except Exception as e:
                logger.error(f"Database connection failed: {e}. Retrying in 5s...")
                time.sleep(5)

    def verify_schema(self) -> None:
        """Verifies that target anomaly_results table exists."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_results (
                    time                 TIMESTAMPTZ      NOT NULL,
                    reconstruction_error DOUBLE PRECISION NOT NULL CHECK (reconstruction_error >= 0),
                    threshold            DOUBLE PRECISION NOT NULL CHECK (threshold > 0),
                    is_anomaly           BOOLEAN          NOT NULL,
                    model_version        TEXT             DEFAULT 'v1'
                );
            """)
            # Verify hypertable
            try:
                cur.execute("SELECT create_hypertable('anomaly_results', 'time', if_not_exists => TRUE);")
            except Exception:
                # TimescaleDB extension might not be enabled on simple PostgreSQL instances
                pass
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ar_time ON anomaly_results (time DESC);")
        self.conn.commit()
        logger.info("Database schema verified.")

    def get_last_processed_time(self) -> datetime | None:
        """Retrieves the timestamp of the latest prediction from anomaly_results."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT MAX(time) FROM anomaly_results")
            val = cur.fetchone()[0]
            return val

    def get_preceding_errors(self, ref_time: datetime, count: int = 4) -> list[float]:
        """Gets reconstruction errors of previous predictions to initialize the smoothing buffer."""
        if count <= 0:
            return []
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT reconstruction_error FROM anomaly_results 
                WHERE time < %s 
                ORDER BY time DESC 
                LIMIT %s
            """, (ref_time, count))
            rows = cur.fetchall()
            # Reverse to maintain chronological order: oldest to newest
            return [float(r[0]) for r in reversed(rows)]

    @torch.no_grad()
    def predict_sequences(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Runs batch model prediction and returns raw reconstruction errors in chunks."""
        self.model.eval()
        errors = []
        for i in range(0, len(X), batch_size):
            chunk = X[i : i + batch_size]
            x_tensor = torch.from_numpy(chunk).to(self.device)
            recon = self.model(x_tensor)
            err = ((recon - x_tensor) ** 2).mean(dim=(1, 2))
            errors.extend(err.cpu().numpy().tolist())
        return np.array(errors, dtype=np.float32)

    def process_data(self) -> datetime | None:
        """Fetches new feature vectors, runs inference, and writes results back."""
        last_time = self.get_last_processed_time()
        
        with self.conn.cursor() as cur:
            if last_time is None:
                logger.info("No existing predictions found in DB. Performing complete backfill...")
                cur.execute("SELECT time, features FROM feature_vectors ORDER BY time ASC")
                rows = cur.fetchall()
            else:
                # Find the starting window timestamp to ensure we have SEQUENCE_LENGTH overlapping samples
                # We look up the 20th row before or equal to last_time
                cur.execute("""
                    SELECT MIN(time) FROM (
                        SELECT time FROM feature_vectors 
                        WHERE time <= %s 
                        ORDER BY time DESC 
                        LIMIT %s
                    ) AS sub
                """, (last_time, SEQUENCE_LENGTH))
                start_time = cur.fetchone()[0]
                if start_time is None:
                    start_time = last_time

                cur.execute("""
                    SELECT time, features FROM feature_vectors 
                    WHERE time >= %s 
                    ORDER BY time ASC
                """, (start_time,))
                rows = cur.fetchall()

        if len(rows) < SEQUENCE_LENGTH:
            return last_time

        times = [r[0] for r in rows]
        features = np.array([r[1] for r in rows], dtype=np.float32)

        # Find starting index for stride loop
        if last_time is None:
            start_idx = 0
        else:
            try:
                idx = times.index(last_time)
                start_idx = idx + SEQUENCE_STRIDE - SEQUENCE_LENGTH + 1
            except ValueError:
                # Fallback in case of database timezone shifts or missing timestamps
                start_idx = 0
                while start_idx + SEQUENCE_LENGTH <= len(times):
                    if times[start_idx + SEQUENCE_LENGTH - 1] > last_time:
                        break
                    start_idx += 1

        seqs = []
        seq_times = []
        end_idx = start_idx + SEQUENCE_LENGTH - 1

        while end_idx < len(times):
            seqs.append(features[end_idx - SEQUENCE_LENGTH + 1 : end_idx + 1])
            seq_times.append(times[end_idx])
            end_idx += SEQUENCE_STRIDE

        if not seqs:
            return last_time

        X = np.stack(seqs)
        raw_errors = self.predict_sequences(X)

        # Apply score smoothing if enabled (default=True)
        if not self.args.no_smoothing:
            # For the first sequence in the batch, fetch historical errors from DB
            preceding_count = self.args.smoothing_window - 1
            if last_time is not None:
                preceding_errors = self.get_preceding_errors(seq_times[0], preceding_count)
            else:
                preceding_errors = []

            all_errors = preceding_errors + list(raw_errors)
            smoothed_errors = []
            for i in range(len(preceding_errors), len(all_errors)):
                window = all_errors[max(0, i - preceding_count) : i + 1]
                smoothed_errors.append(float(np.mean(window)))
            errors_to_save = smoothed_errors
        else:
            errors_to_save = [float(e) for e in raw_errors]

        # Write predictions back to anomaly_results table
        results = []
        for t, err in zip(seq_times, errors_to_save):
            is_anomaly = bool(err >= self.threshold)
            results.append((t, err, self.threshold, is_anomaly, self.args.model_version))

        logger.info(f"Writing {len(results)} new prediction results to database...")
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO anomaly_results (time, reconstruction_error, threshold, is_anomaly, model_version)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, results)
        self.conn.commit()

        new_anomalies = sum(1 for r in results if r[3])
        logger.info(f"Successfully wrote {len(results)} rows. Anomalies detected: {new_anomalies}.")
        
        return seq_times[-1]

    def run(self) -> None:
        """Main execution entrypoint."""
        self.connect_db()

        if self.args.mode == "batch":
            logger.info("Executing service in BATCH mode.")
            self.process_data()
            logger.info("Batch execution completed successfully. Exiting.")
            if self.conn:
                self.conn.close()
            return

        logger.info(f"Executing service in STREAM mode. Polling interval: {self.args.poll_interval}s.")
        try:
            while self.running:
                try:
                    self.process_data()
                except psycopg2.InterfaceError:
                    logger.warning("Database interface error. Reconnecting...")
                    self.connect_db()
                except Exception as e:
                    logger.error(f"Unexpected error in polling loop: {e}", exc_info=True)
                
                time.sleep(self.args.poll_interval)
        finally:
            if self.conn:
                self.conn.close()
                logger.info("Database connection closed.")

    def stop(self, *_) -> None:
        """Clean shutdown handler."""
        logger.info("Shutdown signal received. Stopping inference service...")
        self.running = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM-AE Cloud Inference Service")
    parser.add_argument("--mode", type=str, choices=["stream", "batch"], default="stream",
                        help="Execution mode: stream (polling) or batch (single run)")
    parser.add_argument("--model-path", type=str, default="bucket/best_model.pth",
                        help="Path to trained PyTorch model checkpoint (.pth)")
    parser.add_argument("--metrics-path", type=str, default="bucket/metrics.json",
                        help="Path to training metrics JSON containing optimal threshold")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Database polling interval in stream mode (seconds)")
    parser.add_argument("--model-version", type=str, default="v1",
                        help="Model version string for database logging")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="Disable rolling score smoothing (default enabled)")
    parser.add_argument("--smoothing-window", type=int, default=5,
                        help="Window size for rolling average smoothing")
    parser.add_argument("--cpu", action="store_true",
                        help="Force PyTorch execution on CPU")
    args = parser.parse_args()

    service = InferenceService(args)
    signal.signal(signal.SIGINT, service.stop)
    signal.signal(signal.SIGTERM, service.stop)
    service.run()
