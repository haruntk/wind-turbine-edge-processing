# Wind Turbine Predictive Maintenance

Real-time predictive maintenance system for wind turbines using edge computing and deep learning.

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────┐
│   NVIDIA Jetson     │     │   TimescaleDB       │     │  Dashboard  │
│   (Edge Device)     │────▶│   (Time-Series DB)  │◀────│  (Grafana)  │
│                     │     │                     │     │             │
│  Raw .mat → FFT →   │     │  feature_vectors    │     │  Real-time  │
│  Feature Extraction │     │  anomaly_results    │     │  Monitoring │
│  → 426-dim vector   │     │  anomaly_logs       │     │             │
└─────────────────────┘     └─────────────────────┘     └─────────────┘
```

**Data Flow:** `Sensor Data (.mat) → Edge Processing (Jetson) → Feature Vectors → TimescaleDB → LSTM Autoencoder → Dashboard`

## Features

- **Edge Processing Pipeline**: Multi-rate signal processing on NVIDIA Jetson
- **426-Dimensional Feature Vectors**: FFT spectral + time-domain feature extraction
- **Multi-Rate Sensor Fusion**: Bearing (74 kHz), Nacelle (37 kHz), Tower (2.96 kHz), Slow (1.48 kHz)
- **50% Overlapping Windows**: 1s windows for high-freq, 5s for low-freq sensors
- **Window-Step Feature Rows**: One original 5s/50% overlap vector per window, no hold duplicates
- **TimescaleDB Storage**: Hypertable-backed time-series storage with compression

## Dataset

[Fraunhofer LBF Wind Turbine Vibration Dataset](https://fordatis.fraunhofer.de/handle/fordatis/151.2) — 28 acceleration, temperature, and wind channels across 7 fault scenarios.

## Project Structure

```
wind-turbine-predictive-maintenance/
├── edge/                          # Edge processing package (runs on Jetson)
│   ├── config/
│   │   ├── config.yaml            # Pipeline configuration
│   │   ├── channel_groups.py      # Sensor group definitions
│   │   └── settings.py            # YAML loader with env var interpolation
│   ├── data_ingestion/
│   │   └── data_loader.py         # MATLAB v5/v7.3 (HDF5) file loader
│   ├── data_sender/
│   │   └── db_client.py           # TimescaleDB batch insertion client
│   ├── feature_extraction/
│   │   ├── spectral_features.py   # FFT-based feature extraction
│   │   ├── time_features.py       # Time-domain statistical features
│   │   └── vector_assembler.py    # Multi-rate alignment & 426-dim assembly
│   ├── signal_processing/
│   │   ├── preprocessing.py       # DC removal, bandstop filtering
│   │   ├── fft_engine.py          # Hann-windowed FFT computation
│   │   └── windowing.py           # 50% overlap sliding window
│   ├── utils/
│   │   └── logging.py             # Structured logging setup
│   ├── main.py                    # Pipeline orchestrator & CLI
│   └── requirements.txt           # Python dependencies
├── scripts/
│   ├── init_db.sql                # Full TimescaleDB schema
│   ├── deploy_jetson.sh           # Jetson deployment helper
│   └── generate_test_data.py      # Synthetic .mat generator for testing
├── docs/
│   ├── jetson_setup.md            # Jetson hardware setup guide
│   └── local_testing.md           # Local development guide
├── .env.example                   # Environment variable template
└── .gitignore
```

## Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/your-username/wind-turbine-predictive-maintenance.git
cd wind-turbine-predictive-maintenance

# Set up environment variables
cp .env.example .env
# Edit .env with your database credentials
```

### 2. Database Setup

```bash
# Start TimescaleDB (Docker)
docker run -d --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=your_password \
  timescale/timescaledb:latest-pg16

# Create database and schema
docker exec -u postgres timescaledb psql -c "CREATE DATABASE edge_db;"
docker exec -u postgres timescaledb psql -d edge_db -f /scripts/init_db.sql
```

### 3. Install Dependencies

```bash
pip install -r edge/requirements.txt
```

### 4. Run Pipeline

```bash
# Set environment variables (or use .env file)
export DB_HOST=192.168.1.100
export DB_NAME=edge_db
export DB_USER=postgres
export DB_PASSWORD=your_password

# Dry run (no database)
python -m edge.main --dry-run

# Full pipeline
python -m edge.main --mode batch
```

### 5. Generate Test Data (Optional)

```bash
python scripts/generate_test_data.py --output data/test.mat --duration 30
```

## Feature Vector Breakdown (426 dimensions)

| Sensor Group | Channels | Fs (Hz) | Window | Features/Ch | Total |
|---|---|---|---|---|---|
| Bearing | 6 | 74,000 | 1s | 16 (9 spectral + 7 bands) | 96 |
| Nacelle | 3 | 37,000 | 1s | 15 (9 spectral + 6 bands) | 45 |
| Tower/Tach | 13 | 2,960 | 5s | 15 (9 spectral + 6 bands) | 195 |
| Slow | 6 | 1,480 | 5s | 15 (time-domain only) | 90 |
| **Total** | **28** | | | | **426** |

## Environment Variables

| Variable | Description | Example |
|---|---|---|
| `DB_HOST` | TimescaleDB host IP | `192.168.1.100` |
| `DB_NAME` | Database name | `edge_db` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | *(set securely)* |

## License

This project is part of a Bachelor's thesis on Wind Turbine Predictive Maintenance.
