import pytest
from src.app import app, start_app  # Import the Flask app instance
from unittest.mock import patch


@pytest.fixture
def client():
    # Set up Flask test client
    app.testing = True
    with app.test_client() as client:
        yield client


def test_predict_endpoint(client):
    # Simulate a POST request to the /predict endpoint
    test_data = {"features": [
        1.0, 0.5, 3.2, 2.1, 0.0, 1.5, 3.3, 2.8, 0.9, 1.7]}
    # Example feature vector
    response = client.post('/predict', json=test_data)

    assert response.status_code == 200
    response_json = response.get_json()
    assert 'prediction' in response_json
    assert isinstance(response_json['prediction'], list)


def test_start_app():
    with patch("src.app.app.run") as mock_run:
        start_app()
        mock_run.assert_called_once_with(host="0.0.0.0", port=5000)
