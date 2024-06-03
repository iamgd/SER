import joblib
from tensorflow.keras.models import load_model
import numpy as np
from feature_extraction import extract_audio_features
import os

def ensemble_predict(audio_file):
    # Check if file exists
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Load models and encoders
    svm_model = joblib.load("models/svm_model.pkl")
    cnn_model = load_model("models/cnn_model.h5", compile=False)  # Suppress compile warning
    lstm_model = load_model("models/lstm_model.h5", compile=False)  # Suppress compile warning
    label_encoder = joblib.load("models/label_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # Extract features
    features = extract_audio_features(audio_file)
    features_scaled = scaler.transform([features])
    features_reshaped = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)

    # Predict with each model
    svm_pred = svm_model.predict(features_scaled)
    cnn_pred = np.argmax(cnn_model.predict(features_reshaped), axis=1)
    lstm_pred = np.argmax(lstm_model.predict(features_reshaped), axis=1)

    # Decode labels
    svm_emotion = label_encoder.inverse_transform(svm_pred)[0]
    cnn_emotion = label_encoder.inverse_transform(cnn_pred)[0]
    lstm_emotion = label_encoder.inverse_transform(lstm_pred)[0]

    # Majority voting
    predictions = [svm_emotion, cnn_emotion, lstm_emotion]
    final_prediction = max(set(predictions), key=predictions.count)

    return final_prediction

# Example usage
audio_file = "D:/MCA/4th sem/SER3/dataset/Actor_01/01_01_01_01_dogs-sitting_disgust.wav"  # Replace with an actual file path
try:
    predicted_emotion = ensemble_predict(audio_file)
    print(f"Predicted Emotion: {predicted_emotion}")
except FileNotFoundError as e:
    print(e)
except PermissionError as e:
    print(f"Permission error: {e}")
