from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the extracted features from the Excel file
df = pd.read_excel("output_data.xlsx")

# Separate the features from the target variable (if any)
X = df.drop(columns=['Audio File'])

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the features and transform them
X_scaled = scaler.fit_transform(X)

# Convert the scaled features back to a DataFrame
scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Add the 'Audio File' column back to the DataFrame
scaled_df['Audio File'] = df['Audio File']

# Display the scaled DataFrame
print(scaled_df.head())

# Save the scaled features to a new Excel file
scaled_excel_file = "scaled_output_data.xlsx"
scaled_df.to_excel(scaled_excel_file, index=False)
