import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

# Load the training data from the Excel file
train_data = pd.read_excel("train_test_data.xlsx", sheet_name='Training')

# Encode the target variable
label_encoder = LabelEncoder()
train_data['Emotion'] = label_encoder.fit_transform(train_data['Emotion'])
# Store the mapping of encoded labels to original labels
label_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

# Separate features (MFCCs) and target variable (Emotion)
X_train = train_data.drop(columns=['Audio File', 'Emotion'])
y_train = train_data['Emotion']

# Reshape X_train to 3D tensor for LSTM input
X_train_lstm = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)

# Convert target variable to categorical
y_train_categorical = to_categorical(y_train)

# Define the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data
model.fit(X_train_lstm, y_train_categorical, epochs=10, batch_size=32)

# Load the testing data from the Excel file
test_data = pd.read_excel("train_test_data.xlsx", sheet_name='Testing')

# Encode the target variable
test_data['Emotion'] = label_encoder.transform(test_data['Emotion'])

# Separate features (MFCCs) and target variable (Emotion)
X_test = test_data.drop(columns=['Audio File', 'Emotion'])
y_test = test_data['Emotion']

# Reshape X_test to 3D tensor for LSTM input
X_test_lstm = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert target variable to categorical
y_test_categorical = to_categorical(y_test)

# Predict probabilities for each class for the testing data
y_pred_probabilities = model.predict(X_test_lstm)

# Get the predicted class labels
y_pred = y_pred_probabilities.argmax(axis=1)

# Convert the predicted and true labels back to their original emotion names
y_pred_emotions = [label_mapping[label] for label in y_pred]
y_true_emotions = [label_mapping[label] for label in y_test]

# Generate classification report
report = classification_report(y_true_emotions, y_pred_emotions, output_dict=True)

# Convert report to DataFrame
report_df = pd.DataFrame(report).transpose()

# Save classification report to Excel file
report_df.to_excel("classify_report_lstm.xlsx")

# Create confusion matrix
conf_matrix = confusion_matrix(y_true_emotions, y_pred_emotions)

# Calculate accuracy
accuracy = conf_matrix.diagonal().sum() / conf_matrix.sum()

# Calculate misclassification rate
misclassification_rate = 1 - accuracy

# Calculate precision for each class
precision = conf_matrix.diagonal() / conf_matrix.sum(axis=0)

# Handle division by zero
precision = np.nan_to_num(precision, nan=0)

# Calculate sensitivity (recall) for each class
sensitivity = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Calculate specificity for each class
specificity = [(conf_matrix.sum() - conf_matrix[:, i].sum() - conf_matrix[i, :].sum() + conf_matrix[i, i]) /
               (conf_matrix.sum() - conf_matrix[:, i].sum()) for i in range(conf_matrix.shape[0])]

# Calculate mean precision, sensitivity, and specificity
mean_precision = np.mean(precision)
mean_sensitivity = sensitivity.mean()
mean_specificity = sum(specificity) / len(specificity)


# Create a DataFrame for better visualization
labels = sorted(set(y_true_emotions) | set(y_pred_emotions))
conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (LSTM)')

# Save confusion matrix plot as an image file
plt.savefig("confusion_matrix_lstm.png")

# Show the confusion matrix plot
plt.show()

# Describe the confusion matrix
print("\nConfusion Matrix (LSTM):\n", conf_matrix_df)


# Print the metrics
print("\nAccuracy:", accuracy)
print("Misclassification Rate:", misclassification_rate)
print("\nMean Precision:", mean_precision)
print("Mean Sensitivity (Recall):", mean_sensitivity)
print("Mean Specificity:", mean_specificity)
