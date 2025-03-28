from flask import Flask, request, jsonify, render_template
from model import ProfileDetector
import json

app = Flask(_name_)
detector = ProfileDetector()

# Load the trained model at startup
detector.load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Get data from the request
    data = request.get_json()
    features = [
        data['account_age'],  # X1
        data['followers'],     # X2
        data['following'],     # X3
        data['post_frequency'], # X4
   ]
    
    # Predict
    result = detector.predict(features)
    return jsonify({'is_fake': bool(result)})

if _name_ == '_main_':
    app.run(debug=True)
