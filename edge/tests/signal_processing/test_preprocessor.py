import numpy as np
import pytest

from edge.signal_processing.preprocessor import (
    detrend_signal,
    bandstop_filter,
    preprocess_signal
)

def test_detrend_signal():
    signal = np.array([10.0, 12.0, 10.0, 12.0])
    detrended = detrend_signal(signal)
    
    # Mean should be 0
    assert abs(np.mean(detrended)) < 1e-10
    np.testing.assert_array_equal(detrended, np.array([-1.0, 1.0, -1.0, 1.0]))

def test_bandstop_filter(sine_wave, noisy_signal):
    # sine_wave is 50 Hz, let's create a signal with 50 Hz and 200 Hz
    fs = 1000.0
    t = np.arange(0, 1.0, 1/fs)
    signal_200hz = np.sin(2 * np.pi * 200 * t)
    combined_signal = sine_wave + signal_200hz
    
    # Filter out 50 Hz (45 to 55)
    filtered = bandstop_filter(combined_signal, low_hz=45.0, high_hz=55.0, sampling_rate_hz=fs)
    
    # Perform FFT on filtered signal
    fft_vals = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1/fs)
    
    idx_50 = np.argmin(np.abs(freqs - 50.0))
    idx_200 = np.argmin(np.abs(freqs - 200.0))
    
    # 50 Hz should be attenuated
    assert fft_vals[idx_50] < 10.0
    # 200 Hz should be preserved (it is dominant now)
    assert fft_vals[idx_200] > 100.0

def test_bandstop_filter_invalid_freqs():
    with pytest.raises(ValueError):
        bandstop_filter(np.array([1, 2, 3]), low_hz=60, high_hz=50, sampling_rate_hz=1000)

def test_preprocess_signal(sample_config):
    signal = np.ones(100) * 10.0
    processed = preprocess_signal(signal, sample_config, sampling_rate_hz=1000.0)
    
    # Detrend is enabled in sample_config
    assert abs(np.mean(processed)) < 1e-10
    np.testing.assert_array_almost_equal(processed, np.zeros(100))
