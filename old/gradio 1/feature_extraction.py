import librosa
import numpy as np

def extract_audio_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
    except Exception as e:
        raise RuntimeError(f"Error loading audio file {audio_file}: {e}")
    
    # Example feature extraction: MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    return mfccs_mean
