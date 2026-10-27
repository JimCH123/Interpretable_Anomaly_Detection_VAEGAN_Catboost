"""Smoke tests: preprocessing artifacts, distance metrics, generators and the API."""
import numpy as np
import pandas as pd
import pytest

from anomaly_detection import config
from anomaly_detection.distance_metrics import compare
from anomaly_detection.preprocessing import selected_feature_names, transform


@pytest.fixture(scope="module")
def streaming_data():
    return pd.read_csv(config.STREAMING_DATA)


def raw_line(row):
    raw = row.drop([config.TARGET_COL, config.TIME_COL]).values
    return ",".join("" if pd.isna(v) else str(v) for v in raw)


def test_transform_selects_and_scales_features(streaming_data):
    raw = streaming_data.iloc[0].drop([config.TARGET_COL, config.TIME_COL]).tolist()
    x = transform(raw)
    assert x.shape == (1, len(selected_feature_names()))
    assert np.isfinite(x).all()


def test_transform_rejects_wrong_length():
    with pytest.raises(ValueError):
        transform([1.0, 2.0])


def test_distance_metrics_prefer_identical_samples():
    anomalies = pd.read_csv(config.ANOMALY_DATA)
    same = compare(anomalies, anomalies)
    shifted = compare(anomalies, anomalies.assign(**{c: anomalies[c] + 0.5 for c in anomalies.columns[:-1]}))
    for metric in same:
        assert same[metric][1] < shifted[metric][1]


@pytest.mark.parametrize("model_name", ["vae", "vaegan"])
def test_generator_trains_and_samples(model_name):
    from anomaly_detection.generation import feature_columns, generate_anomaly_data, prepare_vae_data

    data = pd.read_csv(config.CLEANED_DATA)
    fail_df, _ = prepare_vae_data(data)
    synthetic = generate_anomaly_data(model_name, fail_df, 16, feature_columns(data), epochs=1)
    assert synthetic.shape == (16, data.shape[1])
    assert (synthetic[config.TARGET_COL] == config.ANOMALY_LABEL).all()


def test_api_flags_streamed_records(streaming_data):
    from fastapi.testclient import TestClient
    from anomaly_detection.api import app

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}

    normal = streaming_data[streaming_data[config.TARGET_COL] == config.NORMAL_LABEL].iloc[0]
    response = client.post("/predict", json={"line": raw_line(normal)})
    assert response.status_code == 200
    assert response.json()["is_anomaly"] is False

    assert client.post("/predict", json={"line": "1,2,3"}).status_code == 400
