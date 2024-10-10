import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
import numpy as np
from data_preprocessing import load_data_from_directory, preprocess_data, get_dataloaders
from model import TransformerModel
from torch.utils.data import DataLoader

# Load the new data
directory_path = '../csv_output/Collection'  # Set your new dataset path
new_data = load_data_from_directory(directory_path)

# Preprocess the new data (reuse scaler and label_encoder from training if needed)
_, test_dataset, scaler, label_encoder = preprocess_data(new_data)

# Create a DataLoader for the test data
test_loader = DataLoader(test_dataset, batch_size=64)

# Define model parameters (make sure they match the saved model's parameters)
numerical_dims = test_dataset[0][0].shape[0]  # Get number of features from the test data

# Initialize the Transformer model (same architecture as in training)
model = TransformerModel(
    input_dim=numerical_dims,  # Input feature dimensions
    hidden_dim=64,  # Hidden layer dimensions
    output_dim=len(label_encoder.classes_),  # Number of output classes
    num_heads=4,  # Number of attention heads
    num_layers=2  # Number of Transformer layers
)

# Load the saved model state
model.load_state_dict(torch.load('model_epoch_16.pth'))  # Adjust the filename as needed

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set model to evaluation mode
model.eval()

# Inference on the test data
correct = 0
total = 0
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        X_batch, y_batch = [b.to(device) for b in batch]
        outputs = model(X_batch)
        
        # Get the predicted class with the highest score
        _, predicted = torch.max(outputs, 1)
        
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(y_batch.cpu().numpy())

        # Calculate accuracy
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

# Calculate overall accuracy
test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

# Convert lists to numpy arrays for metrics calculation
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Precision, Recall, F1-Score for each class
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Count of each label in the dataset
label_counts = Counter(true_labels)
print("Label counts in the test data:")
for label, count in label_counts.items():
    print(f"Label {label_encoder.inverse_transform([label])[0]}: {count} samples")

# Optionally, you can also save predictions and true labels for further analysis
