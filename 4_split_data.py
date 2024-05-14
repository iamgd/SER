import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the scaled MFCC features from the Excel file
df = pd.read_excel("D:/MCA/4th sem/SER3/output/Actor_01/scaled_output_data.xlsx")

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create the full path for the output Excel file
output_folder = "D:/MCA/4th sem/SER3/output/Actor_01"
output_excel_file = os.path.join(output_folder, "train_test_data.xlsx")

# Create a new Excel file
with pd.ExcelWriter(output_excel_file) as writer:
    # Write the training data to the first sheet
    train_df.to_excel(writer, sheet_name='Training', index=False)

    # Write the testing data to the second sheet
    test_df.to_excel(writer, sheet_name='Testing', index=False)

print("Data split and saved successfully.")
