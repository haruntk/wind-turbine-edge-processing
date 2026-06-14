import pytest
import numpy as np

@pytest.fixture
def sample_config():
    return {
        "mqtt": {
            "host": "localhost",
            "port": 1883,
            "topic_prefix": "turbines",
            "qos": 1,
            "tls": False,
            "username": "testuser",
            "password": "testpassword"
        },
        "preprocessing": {
            "detrend": True,
            "bandstop": {
                "enabled": True,
                "low_hz": 45.0,
                "high_hz": 55.0
            }
        },
        "windowing": {
            "overlap_ratio": 0.5,
            "drop_last": True
        },
        "processing": {
            "output_step_seconds": 2.5
        },
        "model_input": {
            "scaler_path": "dummy_scaler.pkl",
            "expected_group_order": ["bearing", "nacelle", "tower", "slow"],
            "clip_min": -10.0,
            "clip_max": 10.0
        },
        "sensor_groups": {
            "bearing": {
                "sampling_rate_hz": 74000.0,
                "channels": ["bearing_x", "bearing_y"],
                "representation": "fft",
                "window_size_seconds": 5.0,
                "band_energy_edges_hz": [0, 1000, 5000, 10000, 20000]
            },
            "slow": {
                "sampling_rate_hz": 1480.0,
                "channels": ["temp", "wind_speed"],
                "representation": "time",
                "window_size_seconds": 5.0
            }
        }
    }

@pytest.fixture
def sine_wave():
    """Generates a simple 50Hz sine wave for 1 second at 1000Hz sampling rate."""
    fs = 1000.0
    t = np.arange(0, 1.0, 1/fs)
    # 50 Hz sine wave
    return np.sin(2 * np.pi * 50 * t)

@pytest.fixture
def noisy_signal(sine_wave):
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, len(sine_wave))
    return sine_wave + noise
