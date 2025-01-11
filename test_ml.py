import pytest
from ml.model import train_model, compute_model_metrics
from ml.data import process_data
import numpy as np
import pandas as pd


def test_process_data():
    """
    Test the process_data function to ensure it returns expected shapes.
    """
    data = pd.DataFrame({
        "feature1": ["A", "B", "A", "C"],
        "feature2": [1, 2, 3, 4],
        "label": [0, 1, 0, 1]
    })
    cat_features = ["feature1"]
    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="label",
        training=True
    )

    assert X.shape[0] == data.shape[0]
    assert len(y) == data.shape[0]
    assert encoder is not None
    assert lb is not None


def test_train_model():
    """
    Test the train_model function to ensure it trains a model.
    """
    X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    y = np.array([0, 1, 0, 1])
    model = train_model(X, y)

    assert model is not None
    assert hasattr(model, "predict")  # Ensure the model has a predict method


def test_compute_model_metrics():
    """
    Test if compute_model_metrics returns the expected metrics.
    """
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0, 0.1)
    assert recall == pytest.approx(0.5, 0.1)  
    assert fbeta == pytest.approx(0.6667, 0.1) 

