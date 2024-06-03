import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Input

def train_cnn_model(X_train, y_train):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)
    
    return model

def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)
    
    return model

# Load and prepare data
data = pd.read_excel("output/Actor_01/train_test_data.xlsx", sheet_name='Training')
X = data.drop(columns=['Audio File', 'Emotion'])
y = data['Emotion']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data before reshaping for CNN and LSTM
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Reshape data for CNN and LSTM
X_train_cnn_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Check shapes
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_train_cnn_lstm shape: {X_train_cnn_lstm.shape}')

# Train SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
joblib.dump(svm_classifier, "models/svm_model.pkl")

# Train CNN
cnn_model = train_cnn_model(X_train_cnn_lstm, y_train)
cnn_model.save("models/cnn_model.h5")

# Train LSTM
lstm_model = train_lstm_model(X_train_cnn_lstm, y_train)
lstm_model.save("models/lstm_model.h5")

# Save label encoder and scaler
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")
