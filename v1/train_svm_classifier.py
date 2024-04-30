import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def train_classifier():
    # Load the training data from the Excel file
    train_data = pd.read_excel("train_test_data.xlsx", sheet_name='Training')

    # Separate features (MFCCs) and target variable (Emotion)
    X_train = train_data.drop(columns=['Audio File', 'Emotion'])
    y_train = train_data['Emotion']

    # Initialize the SVM classifier
    svm_classifier = SVC(kernel='linear')

    # Fit the classifier to the training data
    svm_classifier.fit(X_train, y_train)

    # Save the trained classifier
    joblib.dump(svm_classifier, "svm_classifier_model.pkl")

def main():
    train_classifier()

if __name__ == "__main__":
    main()
