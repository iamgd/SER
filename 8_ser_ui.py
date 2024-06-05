import os
import gradio as gr
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import joblib
import soundfile as sf
import tensorflow as tf
import traceback

# Placeholder functions (replace with actual implementation)
def noise_reduction(audio):
    # Perform noise reduction
    print("Noise reduction step")
    return audio

def feature_extraction(audio, sr):
    # Extract features (e.g., MFCCs)
    print("Feature extraction step")
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def feature_scaling(features):
    # Scale features
    print("Feature scaling step")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.reshape(-1, 1))
    return features_scaled.flatten()

def split_audio(audio, sr):
    # Split audio into chunks (if necessary)
    print("Audio splitting step")
    return [audio]

def load_models():
    print("Loading models")
    svm_model = joblib.load("D:/MCA/4th sem/SER3/models/svm_model.pkl")
    cnn_model = tf.keras.models.load_model("D:/MCA/4th sem/SER3/models/cnn_model.h5", compile=False)
    lstm_model = tf.keras.models.load_model("D:/MCA/4th sem/SER3/models/lstm_model.h5", compile=False)
    label_encoder = joblib.load("D:/MCA/4th sem/SER3/models/label_encoder.pkl")
    return svm_model, cnn_model, lstm_model, label_encoder

def audio_classification(svm_model, cnn_model, lstm_model, label_encoder, features_scaled):
    print("Predicting emotion using SVM")
    svm_pred = svm_model.predict(features_scaled.reshape(1, -1))
    
    print("Predicting emotion using CNN")
    features_cnn = features_scaled.reshape((1, features_scaled.shape[0], 1))
    cnn_pred = cnn_model.predict(features_cnn)
    cnn_emotion = label_encoder.inverse_transform([np.argmax(cnn_pred)])[0]

    print("Predicting emotion using LSTM")
    lstm_pred = lstm_model.predict(features_cnn)
    lstm_emotion = label_encoder.inverse_transform([np.argmax(lstm_pred)])[0]
    
    svm_emotion = label_encoder.inverse_transform(svm_pred)[0]
    return svm_emotion, cnn_emotion, lstm_emotion

def predict_emotion(audio_tuple):
    print("Received audio data and sample rate")
    
    if audio_tuple is None:
        print("No audio file provided")
        return "No audio file provided"

    print("Audio tuple:", audio_tuple)
    if len(audio_tuple) != 2:
        print("Invalid audio tuple format")
        return "Invalid audio tuple format"

    try:
        # Save the audio data to a temporary file
        temp_folder = "temp"
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        temp_audio_path = os.path.join(temp_folder, "temp_audio.wav")
        sample_rate, audio_data = audio_tuple
        
        # Check if audio data is mono, reshape if necessary
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=1)
        
        sf.write(temp_audio_path, audio_data, sample_rate, subtype='PCM_24')
        print("Audio saved successfully")
        
        # Load the saved audio file for processing
        audio, sr = librosa.load(temp_audio_path)
        print("Audio loaded successfully")
    except Exception as e:
        print("Error processing audio data:")
        traceback.print_exc()  # Print full traceback
        return f"Error processing audio data: {e}"

    audio = noise_reduction(audio)
    chunks = split_audio(audio, sr)
    
    svm_model, cnn_model, lstm_model, label_encoder = load_models()
    
    predictions = []
    for chunk in chunks:
        features = feature_extraction(chunk, sr)
        features_scaled = feature_scaling(features)
        svm_emotion, cnn_emotion, lstm_emotion = audio_classification(svm_model, cnn_model, lstm_model, label_encoder, features_scaled)
        predictions.extend([svm_emotion, cnn_emotion, lstm_emotion])
    
    # Assuming label mapping for the given emotions
    label_mapping = {0: "Angry", 1: "Calm", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Neutral", 6: "Pleasant Surprise", 7: "Sad"}
    final_prediction = max(set(predictions), key=predictions.count)
    
    return final_prediction


# Create Gradio interface
input_audio = gr.Audio(label="Upload Audio File", type="numpy")
output_text = gr.Textbox(label="Predicted Emotion")

gr.Interface(fn=predict_emotion, inputs=input_audio, outputs=output_text, 
             title="Speech Emotion Recognition", 
             description="Upload an audio file to predict the emotion present in the audio.").launch()
