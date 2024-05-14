import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Load the training data from the Excel file
train_data = pd.read_excel("D:/MCA/4th sem/SER3/output/Actor_01/train_test_data.xlsx", sheet_name='Training')

# Separate features (MFCCs) and target variable (Emotion)
X_train = train_data.drop(columns=['Audio File', 'Emotion'])
y_train = train_data['Emotion']

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear')

# Fit the classifier to the training data
svm_classifier.fit(X_train, y_train)

# Load the testing data from the Excel file
test_data = pd.read_excel("D:/MCA/4th sem/SER3/output/Actor_01/train_test_data.xlsx", sheet_name='Testing')

# Separate features (MFCCs) and target variable (Emotion)
X_test = test_data.drop(columns=['Audio File', 'Emotion'])
y_test = test_data['Emotion']

# Predict emotions for the testing data
y_pred = svm_classifier.predict(X_test)

# Evaluate the classifier
report = classification_report(y_test, y_pred, output_dict=True)

# Convert report to DataFrame
report_df = pd.DataFrame(report).transpose()

# Specify the full path for the output Excel file
output_folder = "D:/MCA/4th sem/SER3/output/Actor_01"
output_excel_file = os.path.join(output_folder, "classify_report_svm.xlsx")

# Save classification report to Excel file
report_df.to_excel(output_excel_file)

print("Classification report saved successfully.")
