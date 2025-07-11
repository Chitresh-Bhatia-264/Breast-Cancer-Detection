from flask import Flask, render_template, request
import numpy as np
from models import model, scaler

app = Flask(__name__)


FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius',
    'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness',
    'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

@app.route('/')
def home():
    return render_template('index.html', feature_names=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    try:

        features = [float(request.form[f'feature{i}']) for i in range(1, 31)]


        features = np.array(features).reshape(1, -1)


        features_scaled = scaler.transform(features)


        prediction = model.predict(features_scaled)


        prediction_label = np.argmax(prediction, axis=1)[0]

        if prediction_label == 0:
            result = 'Malignant'
        else:
            result = 'Benign'

        return render_template('index.html', prediction_text=f'The tumor is: {result}', feature_names=FEATURE_NAMES)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", feature_names=FEATURE_NAMES)


if __name__ == "__main__":
    app.run(debug=True)
