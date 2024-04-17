import numpy as np
import pandas as pd

# Sample data (replace this with your actual data)
mfccs = np.array([[-812.23486, -812.23486, -812.23486, ..., -812.23486, -812.23486, -812.23486],
                  [0., 0., 0., ..., 0., 0., 0.],
                  [0., 0., 0., ..., 0., 0., 0.],
                  ...,
                  [0., 0., 0., ..., 0., 0., 0.],
                  [0., 0., 0., ..., 0., 0., 0.],
                  [0., 0., 0., ..., 0., 0., 0.]])
pitch = np.array([2205., 2205., 2205., ..., 0., 0., 0.])
intensity = np.array([0., 0., 0., ..., 0., 0., 0.])
spectral_centroid = np.array([[0., 0., 0., ..., 0., 0., 0.]])
hnr = np.array([0., 0., 0., ..., 0., 0., 0.])
sr = np.array([48000])
audio_file = np.array(['output\\cleaned_03-02-02-01-02-01-01_calm.wav'])

# Reshape MFCCs to be a 1D array
mfccs_flat = mfccs.flatten()

# Combine features into a single array
data = np.concatenate((mfccs_flat, pitch, intensity, spectral_centroid.flatten(), hnr, sr))

# Convert to DataFrame
df = pd.DataFrame(data.reshape(1, -1))

# Add audio file name as the last column
df['Audio File'] = audio_file

# Rename columns
column_names = [f'MFCC_{i}' for i in range(mfccs_flat.shape[0])] + ['Pitch', 'Intensity', 'Spectral Centroid', 'HNR', 'SR', 'Audio File']
df.columns = column_names

# Save to CSV for SVM classifier
df.to_csv('svm_input.csv', index=False)
