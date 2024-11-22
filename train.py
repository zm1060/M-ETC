# train.py
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils.prune as prune
import logging
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score
from xgboost import XGBClassifier


def apply_pruning(model, amount=0.3):
    """
    Apply L1 unstructured pruning to Linear and LSTM layers in the model.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            logging.info(f"Pruned {name}.weight with amount={amount}")
        
        elif isinstance(module, nn.LSTM):
            for layer in range(module.num_layers):
                prune.l1_unstructured(module, name=f'weight_ih_l{layer}', amount=amount)
                prune.l1_unstructured(module, name=f'weight_hh_l{layer}', amount=amount)
                logging.info(f"Pruned LSTM layer {layer} weights.")

def save_model(model, optimizer, epoch, file_path='model_checkpoint.pth'):
    """
    Save the model checkpoint.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, file_path)
    logging.info(f"Model saved after epoch {epoch} at {file_path}")

def load_model(model, optimizer=None, file_path='model_checkpoint.pth'):
    """
    Load the model checkpoint, handling cases where pruned layers are used.
    """
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path, weights_only=True)
        state_dict = checkpoint['model_state_dict']

        # Remove '_orig' and '_mask' suffixes from pruned weights
        new_state_dict = {}
        for key in state_dict.keys():
            if key.endswith("_orig"):
                new_key = key.replace("_orig", "")
            elif key.endswith("_mask"):
                continue  # skip mask keys
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]

        # Load adjusted state_dict
        model.load_state_dict(new_state_dict, strict=False)
        logging.info(f"Loaded model from {file_path} at epoch {checkpoint['epoch']}")
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return model, optimizer, checkpoint['epoch']
        return model, None, checkpoint['epoch']
    else:
        logging.error(f"No checkpoint found at {file_path}")
        return model, optimizer, 0


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()  # Convert to list for JSON serialization
    }

    if y_pred_proba is not None:
        y_pred_proba = np.array(y_pred_proba)
        y_pred_proba /= y_pred_proba.sum(axis=1, keepdims=True)
        metrics['loss'] = log_loss(y_true, y_pred_proba)

    return metrics

def evaluate_model(model, val_loader, device, criterion):
    """
    Evaluate the model on the validation set and compute various metrics.
    """
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    all_probs = []

    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            val_outputs = model(X_val_batch)
            loss = criterion(val_outputs, y_val_batch)
            val_loss += loss.item()

            probs = torch.nn.functional.softmax(val_outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            val_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            val_labels.extend(y_val_batch.cpu().numpy())

    metrics = calculate_metrics(val_labels, val_preds, all_probs)
    metrics['loss'] = val_loss / len(val_loader)

    return metrics

def train_model(model, model_type, train_loader, val_loader, device, num_epochs=100, lr=1e-4, optimizer_type='adam', momentum=0.9, weight_decay=0.01, checkpoint_path='model_checkpoint.pth', resume=False, save_best=True, best_val_loss=float('inf')):
    """
    Train the model and validate on the validation set.
    """
    try:
        # Select optimizer based on user input
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Criterion and scheduler
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
        start_epoch = 0
        final_metrics = {}  # To store final metrics after training

        # Load model if resuming
        if resume:
            model, optimizer, start_epoch = load_model(model, optimizer, checkpoint_path)

        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = 0
            train_preds, train_labels = [], []

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_labels.extend(y_batch.cpu().numpy())

            avg_train_loss = total_loss / len(train_loader)
            train_metrics = calculate_metrics(train_labels, train_preds)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Metrics: {train_metrics}")
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Metrics: {train_metrics}")
            
            # Validation
            val_metrics = evaluate_model(model, val_loader, device, criterion)
            logging.info(f"Epoch {epoch+1} Validation Metrics: {val_metrics}")
            print(f"Epoch {epoch+1} Validation Metrics: {val_metrics}")

            scheduler.step(val_metrics['loss'])

            # Save current model checkpoint
            save_model(model, optimizer, epoch + 1, checkpoint_path)
            # Save best model checkpoint
            if save_best and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_model(model, optimizer, epoch + 1, f'{model_type}_best_model_checkpoint.pth')
                logging.info(f"Best model updated at epoch {epoch+1} with val_loss {best_val_loss:.4f}")
                print(f"Best model updated at epoch {epoch+1} with val_loss {best_val_loss:.4f}")

            # Update final metrics after each epoch
            final_metrics = {
                'train': train_metrics,
                'val': val_metrics
            }

            logging.info(final_metrics)

        return final_metrics
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        return None

def train_baseline_model(model, model_type, X_train, y_train, X_val, y_val, checkpoint_path, save_best=True, best_val_loss=float('inf')):
    """
    Train and evaluate scikit-learn or XGBoost models with the ability to save the best model.
    """
    model.fit(X_train, y_train)
    # Calculate train metrics
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_proba)

    # Calculate validation metrics
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_pred_proba)

    # Save current model checkpoint
    joblib.dump(model, checkpoint_path)
    logging.info(f"Model saved at {checkpoint_path}")
    
    # Save the best model if required
    if save_best and val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        best_checkpoint_path = f'{model_type}_best_model_checkpoint.pkl'
        joblib.dump(model, best_checkpoint_path)
        logging.info(f"Best model updated with val_loss {best_val_loss:.4f}")
        print(f"Best model updated with val_loss {best_val_loss:.4f}")

    # Log and return metrics
    metrics = {'train': train_metrics, 'val': val_metrics}
    logging.info(metrics)
    return metrics


def test_model(model, test_loader, device, filtered_indices=None):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    test_preds, test_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            outputs = model(X_test_batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            test_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            test_labels.extend(y_test_batch.cpu().numpy())

    metrics = calculate_metrics(test_labels, test_preds, all_probs)
    return metrics

def fine_tune_model(model, train_loader, val_loader, device, optimizer, num_epochs=10, pretrained_path='pretrained_model.pth'):
    """
    Fine-tunes a given model with the provided training and validation loaders.
    """
    # Step 1: Load the pretrained model checkpoint
    if pretrained_path and os.path.isfile(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        logging.info(f"Loaded pretrained checkpoint from {pretrained_path}")
        
        try:
            # Attempt to load the checkpoint with strict=True
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            logging.error(f"RuntimeError during state_dict loading: {e}")
            logging.info("Attempting to load with strict=False...")
            # If strict=True fails, load with strict=False
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Freeze all layers except the final fully connected layers
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            logging.error("Model does not have an attribute 'fc' for fine-tuning.")
            raise AttributeError("Ensure the model has a final layer defined as 'fc'.")
    else:
        logging.error(f"Pretrained model checkpoint not found at {pretrained_path}.")
        return

    # Step 2: Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Step 3: Fine-tuning process
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Fine-tuning Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Step 4: Validate the model
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Step 5: Compute evaluation metrics
    preds = all_preds
    labels = all_labels

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds, average='weighted', zero_division=0),
        'f1': f1_score(labels, preds, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(labels, preds)
    }

    logging.info(f"Validation Metrics: {metrics}")
    return metrics


# Helper functions
def inspect_checkpoint(pretrained_path):
    """
    Utility function to inspect the keys of a checkpoint.
    """
    if os.path.isfile(pretrained_path):
        checkpoint = torch.load(pretrained_path)
        model_state_dict_keys = checkpoint['model_state_dict'].keys()
        logging.info(f"Checkpoint keys: {model_state_dict_keys}")
    else:
        logging.error(f"Checkpoint file {pretrained_path} not found.")