# train.py
import os
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils.prune as prune
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score
from xgboost import XGBClassifier

def apply_pruning(model, amount=0.3):
    """
    Apply L1 unstructured pruning to Linear and LSTM layers in the model.
    
    Parameters:
        model (nn.Module): The neural network model.
        amount (float): Fraction of connections to prune.
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
    
    Parameters:
        model (nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch number.
        file_path (str): Path to save the checkpoint.
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
    Load the model checkpoint.
    
    Parameters:
        model (nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer, optional): The optimizer.
        file_path (str): Path to the checkpoint file.
    
    Returns:
        tuple: (model, optimizer, epoch)
    """
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded model from {file_path} at epoch {checkpoint['epoch']}")
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return model, optimizer, checkpoint['epoch']
        return model, None, checkpoint['epoch']
    else:
        logging.error(f"No checkpoint found at {file_path}")
        return model, optimizer, 0

def evaluate_model(model, val_loader, device, criterion):
    """
    Evaluate the model on the validation set and compute various metrics.
    
    Parameters:
        model (nn.Module): The neural network model.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to run the evaluation on.
        criterion (nn.Module): Loss function.
    
    Returns:
        dict: Dictionary containing evaluation metrics.
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

    # Calculate metrics
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
    
    # Calculate AUC for multi-class
    try:
        val_auc = roc_auc_score(val_labels, all_probs, multi_class='ovo')
    except ValueError:
        val_auc = float('nan')  # Handle cases where AUC cannot be computed

    val_conf_matrix = confusion_matrix(val_labels, val_preds)
    val_kappa = cohen_kappa_score(val_labels, val_preds)

    # Log evaluation metrics
    logging.info(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    logging.info(f"Validation Accuracy: {val_accuracy:.4f}")
    logging.info(f"Validation Precision: {val_precision:.4f}")
    logging.info(f"Validation Recall: {val_recall:.4f}")
    logging.info(f"Validation F1 Score: {val_f1:.4f}")
    logging.info(f"Validation AUC: {val_auc:.4f}")
    logging.info(f"Validation Kappa: {val_kappa:.4f}")
    logging.info(f"Confusion Matrix:\n{val_conf_matrix}")

    return {
        'loss': val_loss / len(val_loader),
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'auc': val_auc,
        'kappa': val_kappa,
        'confusion_matrix': val_conf_matrix
    }

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4, resume=False, checkpoint_path='model_checkpoint.pth', save_best=True):
    """
    Train the model and validate on the validation set.
    
    Parameters:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to run the training on.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        resume (bool): Whether to resume training from a checkpoint.
        checkpoint_path (str): Path to save/load the model checkpoint.
        save_best (bool): Whether to save the best model based on validation loss.
    """
    try:
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
        
        best_val_loss = float('inf')
        start_epoch = 0
        final_metrics = {}  # To store final metrics after training

        if resume:
            model, optimizer, start_epoch = load_model(model, optimizer, checkpoint_path)
        
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = 0
            train_preds = []
            train_labels = []
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                optimizer.step()
                total_loss += loss.item()

                # Collect predictions and labels for training metrics
                train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_labels.extend(y_batch.cpu().numpy())

            avg_train_loss = total_loss / len(train_loader)
            # Calculate training metrics
            train_accuracy = accuracy_score(train_labels, train_preds)
            train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=0)
            train_recall = recall_score(train_labels, train_preds, average='weighted', zero_division=0)
            train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
            
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            
            # Evaluate on validation set
            metrics = evaluate_model(model, val_loader, device, criterion)
            
            # Log the fold metrics
            logging.info(f"Epoch {epoch+1} Metrics: {metrics}")

            # Scheduler step
            scheduler.step(metrics['loss'])
            
            # Save checkpoint
            save_model(model, optimizer, epoch + 1, checkpoint_path)
            
            # Save best model
            if save_best and metrics['loss'] < best_val_loss:
                best_val_loss = metrics['loss']
                save_model(model, optimizer, epoch + 1, 'best_model_checkpoint.pth')
                logging.info(f"Best model updated at epoch {epoch+1} with val_loss {best_val_loss:.4f}")
                print(f"Best model updated at epoch {epoch+1} with val_loss {best_val_loss:.4f}")

            # Save final metrics after each epoch
            final_metrics = {
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'val_loss': metrics['loss'],
                'val_accuracy': metrics['accuracy'],
                'val_precision': metrics['precision'],
                'val_recall': metrics['recall'],
                'val_f1': metrics['f1'],
                'val_auc': metrics['auc'],
                'val_kappa': metrics['kappa'],
                'val_confusion_matrix': metrics['confusion_matrix']
            }

        return final_metrics
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        return None


def test_model(model, test_loader, device, filtered_indices=None):
    """
    Evaluate the model on the test set.
    
    Parameters:
        model (nn.Module): The neural network model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run the evaluation on.
        filtered_indices (list): Indices of flows that were filtered out.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            outputs = model(X_test_batch)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y_test_batch.cpu().numpy())
    
    # Calculate standard metrics
    accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(test_labels, test_preds)

    # Log metrics
    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info(f"Test Precision: {precision:.4f}")
    logging.info(f"Test Recall: {recall:.4f}")
    logging.info(f"Test F1 Score: {f1:.4f}")
    logging.info(f"Test Confusion Matrix:\n{conf_matrix}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }


def fine_tune_model(model, train_loader, val_loader, device, optimizer, num_epochs=10, pretrained_path='pretrained_model.pth'):
    if pretrained_path and os.path.isfile(pretrained_path):
        checkpoint = torch.load(pretrained_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded pretrained model from {pretrained_path}")
    
    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():  # Unfreeze final layer
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Fine-tuning Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

def train_baseline_model(model, X_train, y_train, X_val, y_val):
    """
    Train and evaluate scikit-learn or XGBoost models.
    
    Parameters:
        model: The model to train (RandomForest or XGBoost).
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        X_val (np.array): Validation features.
        y_val (np.array): Validation labels.
    
    Returns:
        dict: A dictionary containing the accuracy, precision, recall, f1 score, and AUC.
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Predict validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else 'N/A'
    
    # Return a dictionary of metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

def train_pytorch_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4):
    """
    Train a PyTorch model, like BiLSTM or CNN-BiLSTM-Attention.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # 评估模型
    metrics = evaluate_model(model, val_loader, device, criterion)
    logging.info(f"{model.__class__.__name__} Validation Metrics: {metrics}")
    return metrics        