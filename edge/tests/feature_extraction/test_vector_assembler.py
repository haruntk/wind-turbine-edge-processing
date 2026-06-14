import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from edge.config.channel_groups import ChannelGroup
from edge.feature_extraction.vector_assembler import (
    _fill_missing_like_training,
    prepare_vectors_for_model_db,
    assemble_feature_vectors
)

def test_fill_missing_like_training():
    # Test ffill and bfill logic
    # NaN in middle takes previous (ffill)
    # NaN at start takes next (bfill)
    # NaN at end takes previous
    matrix = np.array([
        [np.nan, 2.0],
        [1.0, np.nan],
        [np.nan, 4.0]
    ])
    filled = _fill_missing_like_training(matrix)
    
    # Expected:
    # row 0: bfill -> takes from row 1 for col 0. col 1 is 2.0 -> [1.0, 2.0]
    # row 1: [1.0, ffill takes 2.0 from row 0] -> [1.0, 2.0]
    # row 2: [ffill takes 1.0 from row 1, 4.0] -> [1.0, 4.0]
    expected = np.array([
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 4.0]
    ])
    np.testing.assert_array_equal(filled, expected)

@patch("edge.feature_extraction.vector_assembler._load_training_scaler")
def test_prepare_vectors_for_model_db(mock_load_scaler, sample_config):
    # Mock scaler behavior
    mock_scaler = MagicMock()
    # Assume 426 features, let's create a dummy matrix of 2 rows, 426 columns
    dummy_matrix = np.ones((2, 426))
    
    # Scaler returns same array * 2
    mock_scaler.transform.return_value = dummy_matrix * 2.0
    mock_scaler.n_features_in_ = 426
    mock_load_scaler.return_value = mock_scaler
    
    prepared = prepare_vectors_for_model_db(list(dummy_matrix), sample_config)
    
    # length of prepared should be 2
    assert len(prepared) == 2
    # values should be 2.0
    assert np.all(prepared[0] == 2.0)

def test_prepare_vectors_for_model_db_invalid_shape(sample_config):
    with pytest.raises(ValueError, match="Expected a 2-D feature matrix"):
        prepare_vectors_for_model_db([np.ones(10)], sample_config)

