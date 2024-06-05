import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

# Load the training data
train_data = pd.read_excel("output/Actor_01/train_test_data.xlsx", sheet_name='Training')

# Separate features (MFCCs) and target variable (Emotion)
X_train = train_data.drop(columns=['Audio File', 'Emotion'])
y_train = train_data['Emotion']

# Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_categorical = to_categorical(y_train_encoded)

# Reshape X_train to 3D tensor for LSTM input
X_train_lstm = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)

# Define the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data
model.fit(X_train_lstm, y_train_categorical, epochs=500, batch_size=32)

# Save the trained model
model.save("models/lstm_model.h5")
