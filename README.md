***To run the speech emotion recognition project, you'll need the following requirements:***

**Python:**

Ensure you have Python installed on your system. You can download and install Python from the official Python website https://www.python.org/downloads/

**Libraries:**

- pandas: For data manipulation and handling Excel files.
- scikit-learn: For machine learning algorithms and evaluation metrics.
- NumPy: For numerical operations.
- librosa: For audio feature extraction.
- pydub: For audio processing and manipulation.
- SciPy: For signal processing and filtering.
- Keras with TensorFlow backend: For building and training deep learning models.
- seaborn: For statistical data visualization based on matplotlib.
- matplotlib: For creating static, animated, and interactive visualizations in Python.
- standardscaler: For scaling features to a standard distribution.
- soundfile: For reading and writing audio files, for saving temp audio files.
- tensorflow: For loading and running pre-trained neural network models.
- gradio: For building web-based UIs for machine learing models

**You can install these libraries using pip:**

pip install pandas scikit-learn soundfile joblib numpy librosa pydub scipy keras tensorflow seaborn matplotlib gradio 


***Steps for running this project:***

- actor 1's 138 audio files are loaded in the dataset folder

- then 1_noise_reduction is performed to remove the noise from the audio files and cleaned audio files are stored in the output folder

- 2_feature_extraction is performed and the features from the audio files are saved as output_data.xlsx

- 3_feature_scaling is done and the output is saved to scaled_output_data.xlsx

- 4_split_data is done and the output is saved as 2sheets one for training and other for testing as train_test_data.xlsx

- 5_audio_classification_svm - here the audio classification is done using SVM classifier and the output is saved as classify_report_svm.xlsx

- 5_audio_classification_lstm - here the audio classification is done using LSTM classifier and the output is saved as classify_report_lstm.xlsx

- 5_audio_classification_cnn - here the audio classification is done using CNN classifier and the output is saved as classify_report_cnn.xlsx

- 6_confusion_matrix_svm - here confusion matrix and confusion metrics are created using SVM and saved as confusion_matrix_svm.png

- 6_confusion_matrix_lstm - here confusion matrix and confusion metrics are created using LSTM and saved as confusion_matrix_lstm.png

- 6_confusion_matrix_cnn - here confusion matrix and confusion metrics are created using CNN and saved as confusion_matrix_cnn.png

- 7_train_cnn - here a CNN model is created using the training phase in the train_test_data file and saved as cnn_model.h5 under the models folder

- 7_train_lstm - here a LSTM model is created using the training phase in the train_test_data file and saved as lstm_model.h5 under the models folder

- 7_train_svm - here a SVM model is created using the training phase in the train_test_data file and saved as svm_model.pk1 under the models folder

- 8_ser_ui_1 - here a UI is created using gradio for loading or capturing the audio files and using machine learning the emotion is the audio is predicted using SVM model

- 8_ser_ui_ - here a UI is created using gradio for loading or capturing the audio files and using machine learning the emotion is the audio is predicted using all 3 models


If you have any doubts or queries feel free to post your quries to this mail id: gdisgd4004@gmail.com
