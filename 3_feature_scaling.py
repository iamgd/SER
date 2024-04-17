from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the extracted features from the Excel file
df = pd.read_excel("output_data.xlsx")

# Separate the MFCC features
mfcc_columns = [col for col in df.columns if col.startswith('MFCC')]
X_mfcc = df[mfcc_columns]

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the MFCC features and transform them
X_mfcc_scaled = scaler.fit_transform(X_mfcc)

# Convert the scaled MFCC features back to a DataFrame
scaled_df_mfcc = pd.DataFrame(X_mfcc_scaled, columns=mfcc_columns)

# Add the 'Audio File' and 'Emotion' columns back to the DataFrame
scaled_df_mfcc['Audio File'] = df['Audio File']
scaled_df_mfcc['Emotion'] = df['Emotion']

# Reorder the columns to have 'Audio File' first, followed by MFCC and Emotion
column_order = ['Audio File'] + mfcc_columns + ['Emotion']
scaled_df_mfcc = scaled_df_mfcc[column_order]

# Display the scaled MFCC DataFrame
print(scaled_df_mfcc.head())

# Save the scaled MFCC features to a new Excel file
scaled_mfcc_excel_file = "scaled_output_data.xlsx"
scaled_df_mfcc.to_excel(scaled_mfcc_excel_file, index=False)
