import torch
import torch.nn as nn
from data_preprocessing import load_data_from_directory, preprocess_data, get_dataloaders
from model import TransformerModel
import logging

# 配置logging
logging.basicConfig(
    filename='training.log',  # 日志文件名
    filemode='a',  # 'a' 代表追加模式，'w' 代表覆盖
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    level=logging.INFO  # 设置日志记录级别为INFO
)

# Load and preprocess data
directory_path = '../csv_output/CIRA-CIC-DoHBrw-2020'
combined_data = load_data_from_directory(directory_path)

# Preprocess data
train_dataset, val_dataset, scaler, label_encoder = preprocess_data(combined_data)

# Create data loaders
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, batch_size=64)

# Define model parameters
numerical_dims = train_dataset[0][0].shape[0]  # Get the number of features

# Initialize the Transformer model
model = TransformerModel(
    input_dim=numerical_dims,  # Input feature dimensions
    hidden_dim=64,  # Hidden layer dimensions
    output_dim=len(label_encoder.classes_),  # Number of output classes
    num_heads=4,  # Number of attention heads
    num_layers=2  # Number of Transformer layers
)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        X_batch, y_batch = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(X_batch)  # Forward pass
        
        # Print sample output for debugging in the first epoch
        if epoch == 0 and total_loss == 0:
            print(f"Sample output: {outputs[:5].cpu().data.numpy()}")
        
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

    # Validation step
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = [b.to(device) for b in batch]
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    val_accuracy = correct / total
    logging.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the model every 5 epochs
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
