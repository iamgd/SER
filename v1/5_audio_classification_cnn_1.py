import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.utils import to_categorical

# Load the training data from the Excel file
train_data = pd.read_excel("train_test_data.xlsx", sheet_name='Training')

# Separate features (MFCCs) and target variable (Emotion)
X_train = train_data.drop(columns=['Audio File', 'Emotion'])
y_train = train_data['Emotion']

# Initialize the label encoder
label_encoder = LabelEncoder()

# Encode the target variable
y_train_encoded = label_encoder.fit_transform(y_train)

# Convert target variable to categorical
y_train_categorical = to_categorical(y_train_encoded)

# Reshape X_train to 3D tensor for CNN input
X_train_cnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)

# Define the CNN model
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data
model.fit(X_train_cnn, y_train_categorical, epochs=10, batch_size=32)

# Load the testing data from the Excel file
test_data = pd.read_excel("train_test_data.xlsx", sheet_name='Testing')

# Separate features (MFCCs) and target variable (Emotion)
X_test = test_data.drop(columns=['Audio File', 'Emotion'])
y_test = test_data['Emotion']

# Encode the target variable
y_test_encoded = label_encoder.transform(y_test)

# Convert target variable to categorical
y_test_categorical = to_categorical(y_test_encoded)

# Reshape X_test to 3D tensor for CNN input
X_test_cnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test_cnn, y_test_categorical)

# Predict probabilities for each class for the testing data
y_pred_probabilities = model.predict(X_test_cnn)

# Get the predicted class labels
y_pred = y_pred_probabilities.argmax(axis=1)

# Decode the predicted labels
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Generate classification report
report = classification_report(y_test, y_pred_decoded, output_dict=True)

# Convert report to DataFrame
report_df = pd.DataFrame(report).transpose()

# Save classification report to Excel file
report_df.to_excel("classify_report_cnn1.xlsx")
