import os
from pydub import AudioSegment
import numpy as np
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff_freq, fs, order=3):
    b, a = butter_lowpass(cutoff_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y.astype(np.int16)

def noise_reduction_with_lowpass(audio_path, output_path, cutoff_freq=2000, order=3):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # List all audio files in the input folder
    audio_files = [f for f in os.listdir(audio_path) if f.endswith(('.wav', '.mp3', '.ogg'))]
    
    for file in audio_files:
        # Load audio file
        audio = AudioSegment.from_file(os.path.join(audio_path, file))

        # Convert audio to numpy array
        audio_data = np.array(audio.get_array_of_samples())

        # Apply low-pass filter for noise reduction
        filtered_audio_data = apply_lowpass_filter(audio_data, cutoff_freq, audio.frame_rate, order)

        # Convert back to audio
        cleaned_audio = AudioSegment(
            filtered_audio_data.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )

        # Save cleaned audio in the output folder
        cleaned_audio.export(os.path.join(output_path, f"cleaned_{file}"), format="wav")
        print(f"Cleaned audio saved to {os.path.join(output_path, f'cleaned_{file}')}")

# Example usage
if __name__ == "__main__":
    audio_folder = "D:/MCA/4th sem/SER3/dataset/Actor_01"
    output_folder = "D:/MCA/4th sem/SER3/output/Actor_01"
    
    noise_reduction_with_lowpass(audio_folder, output_folder)
