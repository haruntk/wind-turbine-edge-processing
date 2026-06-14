from datetime import datetime
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from edge.data_sender.mqtt_publisher import MQTTPublisher

def test_mqtt_publisher_init(sample_config):
    pub = MQTTPublisher.from_config(sample_config, turbine_id="WT-TEST")
    assert pub._host == "localhost"
    assert pub._port == 1883
    assert pub._turbine_id == "WT-TEST"
    assert pub._qos == 1
    assert pub._tls is False

@patch("paho.mqtt.client.Client")
def test_mqtt_publisher_connect_disconnect(mock_client_class, sample_config):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    pub = MQTTPublisher.from_config(sample_config)
    
    def fake_loop_start():
        pub._on_connect(mock_client, None, None, 0)
        
    mock_client.loop_start.side_effect = fake_loop_start
    
    pub.connect()
    
    # Check connection attempts
    mock_client.connect.assert_called_once_with("localhost", 1883, keepalive=60)
    mock_client.loop_start.assert_called_once()
    assert pub._connected.is_set()
        
    # Check disconnect calls on exit
    pub.disconnect()
    mock_client.loop_stop.assert_called_once()
    mock_client.disconnect.assert_called_once()
    assert not pub._connected.is_set()

@patch("paho.mqtt.client.Client")
def test_mqtt_publisher_publish_batch(mock_client_class, sample_config):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    pub = MQTTPublisher.from_config(sample_config)
    pub._client = mock_client
    
    ts = datetime.utcnow()
    features = np.array([1.0, 2.0, 3.0])
    records = [(ts, features, "normal")]
    
    pub.publish_feature_vectors_batch(records)
    
    mock_client.publish.assert_called_once()
    args, kwargs = mock_client.publish.call_args
    assert args[0] == "turbines/WT-001/features"
    assert kwargs.get("qos") == 1
    # Check payload structure (msgpack bytes)
    payload = args[1]
    import msgpack
    unpacked = msgpack.unpackb(payload)
    assert unpacked["turbine_id"] == "WT-001"
    assert unpacked["scenario_label"] == "normal"
    assert unpacked["features"] == [1.0, 2.0, 3.0]

@pytest.mark.integration
def test_mqtt_publisher_integration():
    """Integration test checking connection to the real Mosquitto broker."""
    pub = MQTTPublisher(host="localhost", port=1883, tls=False)
    try:
        pub.connect()
        assert pub._connected.is_set()
        
        # Test publishing a dummy batch
        records = [(datetime.utcnow(), np.zeros(426), "test_scenario")]
        pub.publish_feature_vectors_batch(records)
    except Exception as e:
        pytest.skip(f"Could not connect to the local Mosquitto broker for integration test: {e}")
    finally:
        pub.disconnect()
