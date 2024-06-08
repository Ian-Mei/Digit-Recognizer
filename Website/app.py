from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
with open('digit_classification.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        resized_array = data['resizedArray']
        
        # Convert the list to a numpy array
        image_array = np.array(resized_array, dtype=np.float32)
        
        # Reshape the array to match the input shape expected by the model
        image_array = image_array.reshape(1, 28, 28, 1)

        # Make a prediction
        predictions = model.predict(image_array)
        predicted_digit = np.argmax(predictions, axis=1)[0]
        prediction_confidences = predictions[0].tolist()

        return jsonify(predicted_digit=int(predicted_digit), prediction_confidences=prediction_confidences)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
