import os
import librosa
import pandas as pd

def extract_mfcc(audio_file, sr=22050, n_mfcc=13):
    # Load audio file
    y, _ = librosa.load(audio_file, sr=sr)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return mfccs.T  # Transpose to have MFCC coefficients as columns and frames as rows

def extract_features_from_folder(audio_folder):
    features_dict = {}

    # List all audio files in the folder
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')]

    # Extract features for each audio file
    for audio_file in audio_files:
        features = extract_mfcc(audio_file)
        features_dict[audio_file] = features

    return features_dict

def save_features_to_excel(features_dict, output_file):
    # Initialize DataFrame
    df = pd.DataFrame()

    for audio_file, features in features_dict.items():
        # Create DataFrame for current audio file
        file_df = pd.DataFrame(features, columns=[f'MFCC_{i+1}' for i in range(features.shape[1])])

        # Add the 'Audio File' column
        file_df['Audio File'] = audio_file

        # Append DataFrame to main DataFrame
        df = pd.concat([df, file_df], ignore_index=True)

    # Write DataFrame to Excel file
    df.to_excel(output_file, index=False)
    print(f"Features saved to {output_file}")

# Example usage
if __name__ == "__main__":
    audio_folder = "output"  # Assuming cleaned audio files are in the 'output' folder
    output_excel_file = "features.xlsx"

    # Extract features from the cleaned audio files
    features_dict = extract_features_from_folder(audio_folder)

    # Save features to Excel file
    save_features_to_excel(features_dict, output_excel_file)
