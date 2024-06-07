from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
with open('digit_classification.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    # poop = pd.DataFrame(data['resizedArray'])
    # poop = poop.drop('Unnamed: 0', axis=1)
    # np_data = data.values
    # reshaped_data = np_data.reshape(-1,28, 28,1)
    img = np.array(data['resizedArray']).reshape(-1,28,28,1)
    # img.drop('Unnamed: 0', axis=1)
    

    # Make a prediction using the model
    prediction = model.predict(img)
    prediction = np.argmax(prediction,axis = 1)
    # Return the prediction
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(port=5000, debug=True)