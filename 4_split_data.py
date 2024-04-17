import pandas as pd

# Load the scaled data from the Excel file
df_scaled = pd.read_excel("scaled_output_data.xlsx")

# Split the data into training and testing sets
df_training = df_scaled[:100]  # Select the first 100 rows for training
df_testing = df_scaled[100:138]  # Select the next 38 rows for testing

# Create a new Excel writer object
with pd.ExcelWriter("split_data.xlsx", engine="xlsxwriter") as writer:
    # Write training data to a new sheet named "Training"
    df_training.to_excel(writer, sheet_name="Training", index=False)

    # Write testing data to a new sheet named "Testing"
    df_testing.to_excel(writer, sheet_name="Testing", index=False)

print("Split data saved successfully to split_data.xlsx")
