import pytest
from fastapi.testclient import TestClient
from sklearn import datasets
from api.main import app
import numpy as np

client = TestClient(app)

# load digits dataset
digits = datasets.load_digits(as_frame=True)
X = digits["data"]
y = digits["target"]


def test_return_one_prediction():
    one_sample = [0.0] * 64
    response = client.post(
        "/predict/",
        json={
            "data": [one_sample]
        }
    )
    assert response.json() == {"y_pred": [5]}


def test_return_mult_prediction():
    mult_sample = [[0.0] * 64] * 2
    response = client.post(
        "/predict/",
        json={
            "data": mult_sample
        }
    )
    assert response.json() == {"y_pred": [5, 5]}


def test_real_data():
    mult_sample = X.iloc[:100].values
    y_true = y.iloc[:100].values
    response = client.post(
        "/predict/",
        json={
            "data": mult_sample.tolist()
        }
    )
    body = response.json()
    y_pred = np.array(body["y_pred"])
    accuracy = (y_true == y_pred).mean()
    assert accuracy > .9


