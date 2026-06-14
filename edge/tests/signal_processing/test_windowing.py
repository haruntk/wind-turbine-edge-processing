import numpy as np
import pytest

from edge.signal_processing.windowing import sliding_window

def test_sliding_window_50_percent_overlap():
    signal = np.arange(10)
    windows = sliding_window(signal, window_size_samples=4, overlap_ratio=0.5, drop_last=True)
    
    assert len(windows) == 4
    
    # window 0: [0, 1, 2, 3]
    np.testing.assert_array_equal(windows[0].values, [0, 1, 2, 3])
    assert windows[0].start == 0
    assert windows[0].end == 4
    
    # window 1: [2, 3, 4, 5]
    np.testing.assert_array_equal(windows[1].values, [2, 3, 4, 5])
    assert windows[1].start == 2
    assert windows[1].end == 6

    # window 2: [4, 5, 6, 7]
    np.testing.assert_array_equal(windows[2].values, [4, 5, 6, 7])
    
    # window 3: [6, 7, 8, 9]
    np.testing.assert_array_equal(windows[3].values, [6, 7, 8, 9])

def test_sliding_window_drop_last_false():
    # Signal size 9, window 4, step 2. 
    # Starts: 0, 2, 4, 6.
    # Window at 6: [6, 7, 8]. If drop_last=False, it should pad.
    signal = np.arange(9)
    windows = sliding_window(signal, window_size_samples=4, overlap_ratio=0.5, drop_last=False)
    
    # There should be 5 windows: 
    # [0, 1, 2, 3]
    # [2, 3, 4, 5]
    # [4, 5, 6, 7]
    # [6, 7, 8] -> Padded tail: [5, 6, 7, 8] since we don't zero pad but take tail of original signal length window_size_samples
    assert len(windows) == 4
    # Wait, the sliding window implementation for drop_last=False when len(signal) > window size:
    # it appends `signal[-window_size_samples:]` as a tail if the last end is < len(signal).
    # Ah, the loop goes:
    # start 0: end 4
    # start 2: end 6
    # start 4: end 8
    # start 6: end 10. len=9, so loop range is (0, 9 - 4 + 1 = 6, 2). The range yields 0, 2, 4.
    # At start=4, end=8.
    # Last end is 8. Since 8 < 9, it adds tail: signal[-4:] -> [5, 6, 7, 8]
    assert len(windows) == 4
    np.testing.assert_array_equal(windows[3].values, [5, 6, 7, 8])

def test_sliding_window_invalid():
    with pytest.raises(ValueError):
        sliding_window(np.zeros((2, 2)), 4)
    
    with pytest.raises(ValueError):
        sliding_window(np.zeros(10), -1)
