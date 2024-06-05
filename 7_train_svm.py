import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the training data
train_data = pd.read_excel("output/Actor_01/train_test_data.xlsx", sheet_name='Training')

# Separate features (MFCCs) and target variable (Emotion)
X_train = train_data.drop(columns=['Audio File', 'Emotion'])
y_train = train_data['Emotion']

# Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear')

# Fit the classifier to the training data
svm_classifier.fit(X_train, y_train_encoded)

# Save the trained model
joblib.dump(svm_classifier, "models/svm_model.pkl")
