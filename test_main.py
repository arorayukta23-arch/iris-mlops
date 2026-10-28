from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    r = client.get("/")
    assert r.status_code == 200


def test_predict():
    r = client.post("/predict", json={
        "SepalLengthCm": 5.1,
        "SepalWidthCm": 3.5,
        "PetalLengthCm": 1.4,
        "PetalWidthCm": 0.2,
    })
    assert r.status_code == 200
    assert "prediction" in r.json()