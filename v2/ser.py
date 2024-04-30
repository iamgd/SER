import os
import pandas as pd
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pydub import AudioSegment
from scipy.signal import butter, lfilter

# Function to apply low-pass filter for noise reduction
def butter_lowpass(cutoff, fs, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff_freq, fs, order=3):
    b, a = butter_lowpass(cutoff_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y.astype(np.int16)

# Function to extract audio features
def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfccs = np.mean(mfccs, axis=1)
    return mean_mfccs

# Function to save features to Excel file
def save_features_to_excel(features_dict, output_file):
    df = pd.DataFrame()
    for audio_file, features in features_dict.items():
        mfccs = features
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
        emotion = os.path.basename(audio_file).split('_')[-1].split('.')[0]
        file_df['Emotion'] = emotion
        df = pd.concat([df, file_df], ignore_index=True)
    df.to_excel(output_file, index=False)

# Function to process audio folder
def process_audio_folder(folder):
    data = {}
    audio_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.ogg'))]
    for audio_file in audio_files:
        data[audio_file] = extract_audio_features(audio_file)
    return data

# Function to split data into training and testing sets
def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df

# Function to train SVM classifier
def train_svm_classifier(X_train, y_train):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

# Function to evaluate SVM classifier
def evaluate_svm_classifier(svm_classifier, X_test, y_test):
    y_pred = svm_classifier.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

# Function to apply noise reduction with low-pass filter
def noise_reduction_with_lowpass(audio_path, output_path, cutoff_freq=2000, order=3):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    audio_files = [f for f in os.listdir(audio_path) if f.endswith(('.wav', '.mp3', '.ogg'))]
    for file in audio_files:
        audio = AudioSegment.from_file(os.path.join(audio_path, file))
        audio_data = np.array(audio.get_array_of_samples())
        filtered_audio_data = apply_lowpass_filter(audio_data, cutoff_freq, audio.frame_rate, order)
        cleaned_audio = AudioSegment(
            filtered_audio_data.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
        output_file = os.path.join(output_path, f"cleaned_{file}")
        cleaned_audio.export(output_file, format="wav")
        print(f"Cleaned audio saved to {output_file}")

# Main function
if __name__ == "__main__":
    # Define input and output folders
    audio_folder = "D:/MCA/4th sem/SER3/v2/dataset"
    output_folder = "D:/MCA/4th sem/SER3/v2/output"
    output_excel_file = "output_data.xlsx"
    train_test_excel_file = "train_test_data.xlsx"
    svm_report_excel_file = "classify_report_svm.xlsx"

    # Noise reduction
    noise_reduction_with_lowpass(audio_folder, output_folder)

    # Feature extraction
    features_dict = process_audio_folder(output_folder)
    output_excel_path = os.path.join(output_folder, output_excel_file)
    save_features_to_excel(features_dict, output_excel_path)

    # Feature scaling
    df = pd.read_excel(output_excel_path)
    X = df.drop(columns=['Audio File', 'Emotion'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_df['Audio File'] = df['Audio File']
    scaled_df['Emotion'] = df['Emotion']
    column_order = ['Audio File'] + list(X.columns) + ['Emotion']
    scaled_df = scaled_df[column_order]
    scaled_df.to_excel(output_excel_path, index=False)

    # Split data into training and testing sets
    train_df, test_df = split_data(scaled_df)
    with pd.ExcelWriter(train_test_excel_file) as writer:
        train_df.to_excel(writer, sheet_name='Training', index=False)
        test_df.to_excel(writer, sheet_name='Testing', index=False)

    # Train SVM classifier
    X_train = train_df.drop(columns=['Audio File', 'Emotion'])
    y_train = train_df['Emotion']
    svm_classifier = train_svm_classifier(X_train, y_train)

    # Evaluate SVM classifier
    X_test = test_df.drop(columns=['Audio File', 'Emotion'])
    y_test = test_df['Emotion']
    svm_report = evaluate_svm_classifier(svm_classifier, X_test, y_test)

    # Save classification report to Excel file
    svm_report_excel_path = os.path.join(output_folder, svm_report_excel_file)
    report_df = pd.DataFrame(svm_report).transpose()
    report_df.to_excel(svm_report_excel_path)
    print(f"SVM classification report saved to {svm_report_excel_path}")
