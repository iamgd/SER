# SER3
- actor 1's 138 audio files are loaded in the dataset folder
- then 1_noise_reduction is performed to remove the noise from the audio files and cleaned audio files are stored in the output folder
- 2_feature_extraction is performed and the features from the audio files are saved as output_data.xlsx
- 3_feature_scaling is done and the output is saved to scaled_output_data.xlsx
- 4_split_data is done and the output is saved as 2sheets one for training and other for testing as split_data.xlsx

---- working till 4th ----

- 5_audio_classification_svm and 5_audio_classification_lstm are not working, 
error:  File "C:\Users\dheep\AppData\Roaming\Python\Python312\site-packages\pandas\core\indexes\base.py", line 3812, in get_loc
    raise KeyError(key) from err
KeyError: 'Emotion' ---- 
The error indicates that there is no column named 'Emotion' in your DataFrame. This error occurs when the code tries to access the 'Emotion' column, but it doesn't exist in the DataFrame loaded from the Excel file.