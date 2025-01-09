import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    load_model,
    performance_on_categorical_slice,
)


# Mock data for testing
@pytest.fixture
def mock_data():
    data = pd.DataFrame({
        "age": [25, 32, 47],
        "workclass": ["Private", "Self-emp-not-inc", "Private"],
        "fnlgt": [226802, 89814, 336951],
        "education": ["HS-grad", "Bachelors", "HS-grad"],
        "education-num": [9, 13, 9],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced"],
        "occupation": ["Machine-op-inspct", "Exec-managerial", "Adm-clerical"],
        "relationship": ["Not-in-family", "Husband", "Unmarried"],
        "race": ["Black", "White", "White"],
        "sex": ["Male", "Male", "Female"],
        "capital-gain": [0, 0, 0],
        "capital-loss": [0, 0, 0],
        "hours-per-week": [40, 50, 40],
        "native-country": ["United-States", "United-States", "United-States"],
        "salary": ["<=50K", ">50K", "<=50K"]
    })
    return data


@pytest.fixture
def processed_data(mock_data):
    categorical_features = [
        "workclass", "education", "marital-status", "occupation", "relationship",
        "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(
        mock_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    return X, y, encoder, lb


def test_train_model(processed_data):
    X, y, _, _ = processed_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")


def test_compute_model_metrics():
    y = np.array([1, 0, 1])
    preds = np.array([1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0 
    assert recall == 0.5  
    assert fbeta == 0.6666666666666666 



def test_inference(processed_data):
    X, y, _, _ = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(X)
    assert isinstance(preds, np.ndarray)


def test_save_and_load_model(processed_data, tmp_path):
    X, y, _, _ = processed_data
    model = train_model(X, y)
    model_path = tmp_path / "model.pkl"
    save_model(model, model_path)
    loaded_model = load_model(model_path)
    assert isinstance(loaded_model, RandomForestClassifier)
    assert hasattr(loaded_model, "predict")


def test_performance_on_categorical_slice(mock_data, processed_data):
    X, y, encoder, lb = processed_data
    model = train_model(X, y)
    categorical_features = [
        "workclass", "education", "marital-status", "occupation", "relationship",
        "race", "sex", "native-country"
    ]
    precision, recall, fbeta = performance_on_categorical_slice(
        data=mock_data,
        column_name="education",
        slice_value="HS-grad",
        categorical_features=categorical_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        model=model,
    )
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
