import os
import librosa
import numpy as np
import pandas as pd

def extract_audio_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Compute mean MFCCs across all frames
    mean_mfccs = np.mean(mfccs, axis=1)
    
    return mean_mfccs

def save_features_to_excel(features_dict, output_file):
    # Initialize DataFrame
    df = pd.DataFrame()

    for audio_file, features in features_dict.items():
        # Extract features
        mfccs = features
        
        # Create DataFrame for current audio file
        file_df = pd.DataFrame({
            'Audio File': [audio_file],
            'MFCC_1': [mfccs[0]],
            'MFCC_2': [mfccs[1]],
            'MFCC_3': [mfccs[2]],
            'MFCC_4': [mfccs[3]],
            'MFCC_5': [mfccs[4]],
            'MFCC_6': [mfccs[5]],
            'MFCC_7': [mfccs[6]],
            'MFCC_8': [mfccs[7]],
            'MFCC_9': [mfccs[8]],
            'MFCC_10': [mfccs[9]],
            'MFCC_11': [mfccs[10]],
            'MFCC_12': [mfccs[11]],
            'MFCC_13': [mfccs[12]],
        })

        # Append DataFrame to main DataFrame
        df = pd.concat([df, file_df], ignore_index=True)

    # Write DataFrame to Excel file
    df.to_excel(output_file, index=False)

def process_audio_folder(folder):
    data = {}
    
    # List all audio files in the folder
    audio_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.ogg'))]
    
    # Extract features for each audio file
    for audio_file in audio_files:
        data[audio_file] = extract_audio_features(audio_file)
    
    return data

# Example usage
output_folder = "output"
output_excel_file = "output_data.xlsx"

# Extract features from all audio files in the output folder
data = process_audio_folder(output_folder)

# Save features to Excel file
save_features_to_excel(data, output_excel_file)
