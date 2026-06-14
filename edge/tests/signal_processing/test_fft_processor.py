import numpy as np
import pytest

from edge.signal_processing.fft_processor import apply_hann_window, compute_fft

def test_apply_hann_window(sine_wave):
    # Apply Hann window
    windowed = apply_hann_window(sine_wave)
    
    # Check dimensions
    assert len(windowed) == len(sine_wave)
    
    # Hann window should go to zero at ends (or close to it)
    assert abs(windowed[0]) < 1e-5
    assert abs(windowed[-1]) < 1e-5

def test_compute_fft(sine_wave):
    # sine_wave is 50 Hz sampled at 1000 Hz, length 1000
    # Add Hann window to avoid leakage
    windowed = apply_hann_window(sine_wave)
    freqs, amplitudes, power = compute_fft(windowed, sampling_rate_hz=1000.0)
    
    # Size check (rfft size)
    assert len(freqs) == len(sine_wave) // 2 + 1
    assert len(amplitudes) == len(freqs)
    assert len(power) == len(freqs)
    
    # Power is amplitude squared
    np.testing.assert_array_almost_equal(power, amplitudes ** 2)
    
    # Dominant frequency should be 50 Hz
    dominant_idx = np.argmax(amplitudes)
    assert freqs[dominant_idx] == 50.0

def test_compute_fft_empty():
    with pytest.raises(ValueError, match="empty signal"):
        compute_fft(np.array([]), 1000.0)
