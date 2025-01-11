import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
import argparse

# Set up argument parser for flexible data path input
parser = argparse.ArgumentParser(description="Train ML model")
parser.add_argument(
    "--data_path",
    type=str,
    default=os.path.join(os.getcwd(), "data", "census.csv"),
    help="Path to the census CSV file"
)
args = parser.parse_args()

# Load the census.csv data with error handling
data_path = args.data_path
try:
    data = pd.read_csv(data_path)
    print(f"Data loaded successfully from: {data_path}")
except FileNotFoundError:
    print(f"Error: File not found at {data_path}")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: The file at {data_path} is empty.")
    exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    exit(1)

# Split the provided data into a train dataset and a test dataset
train, test = train_test_split(data, test_size=0.2, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the data using the process_data function
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model using the train_model function
model = train_model(X_train, y_train)

# Save the model and the encoder
project_path = os.getcwd()
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
print(f"Model saved to {model_path}")

encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)
print(f"Encoder saved to {encoder_path}")

# Load the model
model = load_model(model_path)

# Run the model inferences on the test dataset
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute the performance on model slices
slice_output_file = "slice_output.txt"
with open(slice_output_file, "w") as f:  # Clear the file before appending
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                test,
                column_name=col,
                slice_value=slicevalue,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model,
            )
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(
                f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}",
                file=f
            )
print(f"Slice performance saved to {slice_output_file}")
