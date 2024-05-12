import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def load_classifier():
    svm_classifier = joblib.load("svm_classifier_model.pkl")
    return svm_classifier

def main():
    # Load the testing data from the Excel file
    test_data = pd.read_excel("train_test_data.xlsx", sheet_name='Testing')

    # Separate features (MFCCs) and target variable (Emotion)
    X_test = test_data.drop(columns=['Audio File', 'Emotion'])
    y_test = test_data['Emotion']

    # Load the trained SVM classifier
    svm_classifier = load_classifier()

    # Predict emotions for the testing data
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the classifier
    report = classification_report(y_test, y_pred, output_dict=True)

    # Convert report to DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Save classification report to Excel file
    report_df.to_excel("classify_report_svm.xlsx")

if __name__ == "__main__":
    main()
