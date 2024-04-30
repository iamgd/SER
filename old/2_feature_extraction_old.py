import librosa
import numpy as np
import os
import pandas as pd

def extract_audio_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Extract pitch
    pitch = librosa.yin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'))
    
    # Extract intensity
    intensity = np.abs(y)
    
    # Extract spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    # Extract harmonic-to-noise ratio (HNR)
    harmonic, percussive = librosa.effects.hpss(y)
    hnr = librosa.effects.harmonic(y)
    
    return mfccs, pitch, intensity, spectral_centroid, hnr, sr

def save_features_to_excel(features_dict, output_folder, sheet_name):
    # Initialize a dictionary to store transposed data
    transposed_data = {}

    for audio_file, features in features_dict.items():
        # Transpose features
        transposed_features = {key: [value] for key, value in features.items()}
        transposed_features['Audio File'] = [audio_file]

        # Append transposed features to dictionary
        for key, value in transposed_features.items():
            transposed_data.setdefault(key, []).extend(value)
    
    # Create DataFrame from transposed data
    df = pd.DataFrame(transposed_data)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Construct the output file path
    output_file = os.path.join(output_folder, f"{sheet_name}.xlsx")
    
    # Write DataFrame to Excel file with specified sheet name
    df.to_excel(output_file, sheet_name=sheet_name, index=False)

    
def process_audio_folder(folder):
    train_data = {}
    test_data = {}
    
    # List all audio files in the folder
    audio_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.ogg'))]
    
    # Separate the first 100 files for training and the next 38 files for testing
    for i, audio_file in enumerate(audio_files):
        mfccs, pitch, intensity, spectral_centroid, hnr, sr = extract_audio_features(audio_file)
        if i < 100:
            train_data[audio_file] = {"mfccs": mfccs, "pitch": pitch, "intensity": intensity, 
                                      "spectral_centroid": spectral_centroid, "hnr": hnr, "sr": sr}
        else:
            test_data[audio_file] = {"mfccs": mfccs, "pitch": pitch, "intensity": intensity, 
                                     "spectral_centroid": spectral_centroid, "hnr": hnr, "sr": sr}
    
    return train_data, test_data

# Example usage
output_folder = "output"
output_excel_file = "output_data.xlsx"

# Extract features from all audio files in the output1 folder
train_data, test_data = process_audio_folder(output_folder)

# Save features to a single Excel file with two sheets
save_features_to_excel(train_data, output_folder, "train_data")
save_features_to_excel(test_data, output_folder, "test_data")
