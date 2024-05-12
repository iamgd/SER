import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the training data from the Excel file
train_data = pd.read_excel("train_test_data.xlsx", sheet_name='Training')

# Separate features (MFCCs) and target variable (Emotion)
X_train = train_data.drop(columns=['Audio File', 'Emotion'])
y_train = train_data['Emotion']

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear')

# Fit the classifier to the training data
svm_classifier.fit(X_train, y_train)

# Load the testing data from the Excel file
test_data = pd.read_excel("train_test_data.xlsx", sheet_name='Testing')

# Separate features (MFCCs) and target variable (Emotion)
X_test = test_data.drop(columns=['Audio File', 'Emotion'])
y_test = test_data['Emotion']

# Predict emotions for the testing data
y_pred = svm_classifier.predict(X_test)

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a DataFrame for better visualization
labels = sorted(set(y_test) | set(y_pred))
conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (SVM)')

# Save confusion matrix plot as an image file
plt.savefig("confusion_matrix_svm.png")

# Show the confusion matrix plot
plt.show()

# Describe the confusion matrix
print("Confusion Matrix (SVM):\n", conf_matrix_df)

# Calculate accuracy
accuracy = conf_matrix.diagonal().sum() / conf_matrix.sum()

# Calculate misclassification rate
misclassification_rate = 1 - accuracy

# Calculate precision for each class
precision = conf_matrix.diagonal() / conf_matrix.sum(axis=0)

# Calculate sensitivity (recall) for each class
sensitivity = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Calculate specificity for each class
specificity = [(conf_matrix.sum() - conf_matrix[:, i].sum() - conf_matrix[i, :].sum() + conf_matrix[i, i]) /
               (conf_matrix.sum() - conf_matrix[:, i].sum()) for i in range(conf_matrix.shape[0])]

# Calculate mean precision, sensitivity, and specificity
mean_precision = precision.mean()
mean_sensitivity = sensitivity.mean()
mean_specificity = sum(specificity) / len(specificity)

# Print the metrics
print("\nAccuracy:", accuracy)
print("Misclassification Rate:", misclassification_rate)
print("\nMean Precision:", mean_precision)
print("Mean Sensitivity (Recall):", mean_sensitivity)
print("Mean Specificity:", mean_specificity)
