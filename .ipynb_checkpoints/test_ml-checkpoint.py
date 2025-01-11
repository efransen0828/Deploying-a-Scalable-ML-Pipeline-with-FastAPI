# TODO: add necessary import
from ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def test_train_model():
    """
    Test that the train_model function trains and returns a model.
    """
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_train = np.array([0, 1, 1, 0])
    model = train_model(X_train, y_train)
    assert isinstance(
        model, RandomForestClassifier
    ), "Model is not a RandomForestClassifier"
    assert hasattr(
        model, "predict"
    ), "Model does not have a predict method"


def test_compute_model_metrics():
    """
    Test that the compute_model_metrics function returns
    precision, recall, and F1-score.
    """
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1, "Precision is out of range"
    assert 0 <= recall <= 1, "Recall is out of range"
    assert 0 <= fbeta <= 1, "F1-score is out of range"


def test_inference():
    """
    Test that the inference function returns predictions from the model.
    """
    X_test = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_test = np.array([0, 1, 1, 0])
    model = train_model(X_test, y_test)
    predictions = inference(model, X_test)
    assert len(predictions) == len(
        X_test
    ), "Number of predictions does not match input size"
    assert set(predictions).issubset(
        {0, 1}
    ), "Predictions contain invalid labels"
