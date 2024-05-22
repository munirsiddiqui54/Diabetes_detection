# Import necessary libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model from the pkl file
model = pickle.load(open('DiabetesModel.pkl', 'rb'))

@app.route('/')  # Homepage
def home():
    return render_template('index.html')  # You can create an HTML template for your homepage

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the user (you'll need to adapt this part)
        input_features = [float(x) for x in request.form.values()]

        # Make predictions using the loaded model
        prediction = model.predict([input_features])

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
