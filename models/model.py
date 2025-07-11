import pickle
from keras.models import load_model
import os


def load_keras_model():
    model_path = os.path.join(os.path.dirname(__file__), 'breast_cancer_nn_model.h5')
    model = load_model(model_path)
    return model

# Load the scaler
def load_scaler():
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler


model = load_keras_model()
scaler = load_scaler()

