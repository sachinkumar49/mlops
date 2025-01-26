from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction.tolist()})


def start_app():  # Move app.run into a function
    app.run(debug=True, port=5001)


if __name__ == "__main__":
    start_app()
