import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.utils import to_categorical
import os

# Load the training data from the Excel file
train_data = pd.read_excel("D:/MCA/4th sem/SER3/output/Actor_01/train_test_data.xlsx", sheet_name='Training')

# Encode the target variable
label_encoder = LabelEncoder()
train_data['Emotion'] = label_encoder.fit_transform(train_data['Emotion'])
# Store the mapping of encoded labels to original labels
label_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

# Separate features (MFCCs) and target variable (Emotion)
X_train = train_data.drop(columns=['Audio File', 'Emotion'])
y_train = train_data['Emotion']

# Reshape X_train to 3D tensor for CNN input (assuming you have 13 MFCCs)
X_train_cnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)

# Convert target variable to categorical
y_train_categorical = to_categorical(y_train)

# Define the CNN model with increased complexity
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model with adjusted hyperparameters
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data with adjusted batch size and epochs
model.fit(X_train_cnn, y_train_categorical, epochs=20, batch_size=64)

# Load the testing data from the Excel file
test_data = pd.read_excel("D:/MCA/4th sem/SER3/output/Actor_01/train_test_data.xlsx", sheet_name='Testing')

# Encode the target variable
test_data['Emotion'] = label_encoder.transform(test_data['Emotion'])

# Separate features (MFCCs) and target variable (Emotion)
X_test = test_data.drop(columns=['Audio File', 'Emotion'])
y_test = test_data['Emotion']

# Reshape X_test to 3D tensor for CNN input (assuming you have 13 MFCCs)
X_test_cnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert target variable to categorical
y_test_categorical = to_categorical(y_test)

# Predict probabilities for each class for the testing data
y_pred_probabilities = model.predict(X_test_cnn)

# Get the predicted class labels
y_pred = y_pred_probabilities.argmax(axis=1)

# Convert the predicted and true labels back to their original emotion names
y_pred_emotions = [label_mapping[label] for label in y_pred]
y_true_emotions = [label_mapping[label] for label in y_test]

# Evaluate the classifier
report = classification_report(y_true_emotions, y_pred_emotions, output_dict=True)

# Convert report to DataFrame
report_df = pd.DataFrame(report).transpose()

# Specify the full path for the output Excel file
output_folder = "D:/MCA/4th sem/SER3/output/Actor_01"
output_excel_file = os.path.join(output_folder, "classify_report_cnn.xlsx")

# Save classification report to Excel file
report_df.to_excel(output_excel_file)

print("Classification report saved successfully.")
