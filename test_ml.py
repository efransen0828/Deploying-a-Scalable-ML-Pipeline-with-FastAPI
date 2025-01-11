import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
)

# Mock data
MOCK_DATA = {
    "age": [39, 50, 38],
    "workclass": ["State-gov", "Self-emp-not-inc", "Private"],
    "fnlgt": [77516, 83311, 215646],
    "education": ["Bachelors", "Bachelors", "HS-grad"],
    "education-num": [13, 13, 9],
    "marital-status": ["Never-married", "Married-civ-spouse", "Divorced"],
    "occupation": ["Adm-clerical", "Exec-managerial", "Handlers-cleaners"],
    "relationship": ["Not-in-family", "Husband", "Not-in-family"],
    "race": ["White", "White", "Black"],
    "sex": ["Male", "Male", "Male"],
    "capital-gain": [2174, 0, 0],
    "capital-loss": [0, 0, 0],
    "hours-per-week": [40, 13, 40],
    "native-country": ["United-States", "United-States", "United-States"],
    "salary": ["<=50K", ">50K", "<=50K"],
}


def test_process_data():
    """
    Test if process_data returns arrays of expected shapes.
    """
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    data = pd.DataFrame(MOCK_DATA)
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
    assert X.shape[0] == len(data)
    assert y.shape[0] == len(data)


def test_train_model():
    """
    Test if train_model returns a RandomForestClassifier.
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    """
    Test if compute_model_metrics returns the expected metrics.
    """
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)

    # Update expectations to match the calculations
    assert precision == pytest.approx(1.0, 0.1)  # Precision is 1.0
    assert recall == pytest.approx(0.5, 0.1)     # Recall is 0.5
    assert fbeta == pytest.approx(2 / 3, 0.1)    # F1 score is 2/3

