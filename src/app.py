from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the incoming JSON request
    data = request.get_json()
    features = data['features']
    
    # Make prediction
    prediction = model.predict([features])
    
    # Return the result as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
