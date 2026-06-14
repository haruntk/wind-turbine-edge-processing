import numpy as np
import pytest

from edge.feature_extraction.feature_extractor import (
    extract_spectral_features,
    extract_time_features,
    _band_energies,
    _stable_spectral_moments
)

def test_extract_time_features():
    # Simple constant window
    window = np.ones(10) * 5.0
    features = extract_time_features(window)
    
    assert features["mean"] == 5.0
    assert features["std"] == 0.0
    assert features["peak"] == 5.0
    assert features["variance"] == 0.0
    assert features["kurtosis"] == 0.0 # Handled by stable moments check
    assert features["skewness"] == 0.0
    assert features["rms"] == 5.0

def test_extract_spectral_features():
    # Construct a dummy spectrum
    freqs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    amplitudes = np.array([0.0, 5.0, 10.0, 2.0, 1.0])
    power = amplitudes ** 2
    band_edges_hz = [0.0, 15.0, 50.0]
    
    features = extract_spectral_features(
        freqs=freqs,
        amplitudes=amplitudes,
        power=power,
        band_edges_hz=band_edges_hz,
        sampling_rate_hz=100.0,
        n_signal_samples=100
    )
    
    # Dominant frequency should be 20.0 (max amplitude 10.0)
    assert features["dominant_frequency"] == 20.0
    assert features["peak_frequency_amplitude"] == 10.0
    
    # Check band energies
    # Band 1: 0 to 15Hz. Freqs in band: 0, 10. Power sum: 0^2 + 5^2 = 25.
    # Band 2: 15 to 50Hz. Freqs in band: 20, 30, 40. Power sum: 100 + 4 + 1 = 105.
    # Total power = 130
    assert features["band_energy_0_15"] == pytest.approx(25.0 / 130.0)
    assert features["band_energy_15_50"] == pytest.approx(105.0 / 130.0)

def test_stable_spectral_moments():
    amps = np.array([1.0, 1.0, 1.0, 1.0])
    skew, kurt = _stable_spectral_moments(amps)
    assert skew == 0.0
    assert kurt == 0.0
