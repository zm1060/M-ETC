# train.py
import os
import time
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
from tqdm import tqdm
from xgboost import XGBClassifier


# def apply_pruning(model, amount=0.3):
#     """
#     Apply L1 unstructured pruning to Linear and LSTM layers in the model.
#     """
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             prune.l1_unstructured(module, name='weight', amount=amount)
#             logging.info(f"Pruned {name}.weight with amount={amount}")
        
#         elif isinstance(module, nn.LSTM):
#             for layer in range(module.num_layers):
#                 prune.l1_unstructured(module, name=f'weight_ih_l{layer}', amount=amount)
#                 prune.l1_unstructured(module, name=f'weight_hh_l{layer}', amount=amount)
#                 logging.info(f"Pruned LSTM layer {layer} weights.")

def apply_pruning(model, amount=0.3, structured=False, global_prune=False):
    """
    Apply pruning to Linear and LSTM layers in the model.
    
    Parameters:
    - model: nn.Module, the PyTorch model to prune.
    - amount: float, the proportion of parameters to prune.
    - structured: bool, whether to apply structured pruning (default: False).
    - global_prune: bool, whether to apply global pruning across layers (default: False).
    """
    parameters_to_prune = []

    for name, module in model.named_modules():
        # Prune Linear layers
        if isinstance(module, nn.Linear):
            if structured:
                # Structured pruning (entire neurons)
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            else:
                # Unstructured pruning
                prune.l1_unstructured(module, name='weight', amount=amount)
            logging.info(f"Pruned {name}.weight with amount={amount}")
            parameters_to_prune.append((module, 'weight'))

        # Prune LSTM layers
        elif isinstance(module, nn.LSTM):
            for layer in range(module.num_layers):
                if structured:
                    # Structured pruning for LSTM
                    prune.ln_structured(module, name=f'weight_ih_l{layer}', amount=amount, n=2, dim=0)
                    prune.ln_structured(module, name=f'weight_hh_l{layer}', amount=amount, n=2, dim=0)
                else:
                    # Unstructured pruning for LSTM
                    prune.l1_unstructured(module, name=f'weight_ih_l{layer}', amount=amount)
                    prune.l1_unstructured(module, name=f'weight_hh_l{layer}', amount=amount)

                logging.info(f"Pruned LSTM layer {layer} weights.")
                parameters_to_prune.extend([
                    (module, f'weight_ih_l{layer}'),
                    (module, f'weight_hh_l{layer}')
                ])
    
    if global_prune:
        # Global pruning: prune the least important weights across all layers
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        logging.info("Applied global pruning across all layers.")

    # Remove pruning reparameterization (optional)
    for module, param in parameters_to_prune:
        prune.remove(module, param)
        logging.info(f"Removed pruning reparameterization for {param}.")

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
    start_time = time.time()
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
    end_time = time.time()
    inference_time = end_time - start_time
    logging.info(f"Inference time: {inference_time:.2f} seconds")
    metrics = calculate_metrics(val_labels, val_preds, all_probs)
    metrics['loss'] = val_loss / len(val_loader)

    return metrics

def train_model(model, model_type, train_loader, val_loader, device, num_epochs=100, lr=1e-4, optimizer_type='adam', momentum=0.9, weight_decay=0.01, checkpoint_path='model_checkpoint.pth', resume=False, save_best=True, save_current=True, best_val_loss=float('inf')):
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
            start_time = time.time()
            total_loss = 0
            train_preds, train_labels = [], []
            train_pred_proba = []
            # Create a tqdm progress bar for the training loop
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Training]", unit="batch") as pbar:
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
                    train_pred_proba.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

                    # Update progress bar with the current loss
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)
            end_time = time.time()
            train_time = end_time - start_time
            logging.info(f"Train time: {train_time:.2f} seconds")
            train_metrics = calculate_metrics(train_labels, train_preds, train_pred_proba)            
            # Validation
            val_metrics = evaluate_model(model, val_loader, device, criterion)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Metrics: {train_metrics}")
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Metrics: {val_metrics}")
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Metrics: {train_metrics}")
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Metrics: {val_metrics}")

            scheduler.step(val_metrics['loss'])
            # Save current model checkpoint
            if save_current:
                save_model(model, optimizer, epoch + 1, checkpoint_path)
            # Save best model checkpoint
            if save_best and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_model(model, optimizer, epoch + 1, f'{model_type}_best_model_checkpoint.pth')
                # logging.info(f"Best model updated at epoch {epoch+1} with val_loss {best_val_loss:.4f}")
                print(f"Best model updated at epoch {epoch+1} with val_loss {best_val_loss:.4f}")

            # Update final metrics after each epoch
            final_metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            # logging.info(final_metrics)
            # Apply pruning dynamically every 5 epochs
            # if epoch % 5 == 0:
            #     logging.info(f"Applying pruning at epoch {epoch}")
            #     apply_pruning(model, amount=0.1, structured=True)
        return final_metrics
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        return None

def train_baseline_model(model, model_type, X_train, y_train, X_val, y_val, checkpoint_path, save_best=True, best_val_loss=float('inf')):
    """
    Train and evaluate scikit-learn or XGBoost models with the ability to save the best model.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    logging.info(f"Train time: {train_time:.2f} seconds")
    # Calculate train metrics
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_proba)
    # Calculate validation metrics
    start_time = time.time()
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_pred_proba)
    end_time = time.time()
    inference_time = end_time - start_time
    logging.info(f"Inference time: {inference_time:.2f} seconds")
    # Save current model checkpoint
    joblib.dump(model, checkpoint_path)
    # logging.info(f"Model saved at {checkpoint_path}")
    
    # Save the best model if required
    if save_best and val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        best_checkpoint_path = f'{model_type}_best_model_checkpoint.pkl'
        joblib.dump(model, best_checkpoint_path)
        # logging.info(f"Best model updated with val_loss {best_val_loss:.4f}")
        print(f"Best model updated with val_loss {best_val_loss:.4f}")

    # Log and return metrics
    metrics = {'train': train_metrics, 'val': val_metrics}
    logging.info(f"{metrics}")
    return metrics

def test_model(model, test_loader, device, filtered_indices=None):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    test_preds, test_labels, all_probs = [], [], []
    
    # Use tqdm for progress bar
    with tqdm(total=len(test_loader), desc="Testing", unit="batch") as pbar:
        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:
                X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
                outputs = model(X_test_batch)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                test_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                test_labels.extend(y_test_batch.cpu().numpy())
                pbar.update(1)  # Update progress bar by one batch

    metrics = calculate_metrics(test_labels, test_preds, all_probs)
    return metrics

def fine_tune_model(model, model_type, train_loader, val_loader, device, num_epochs=10, lr=1e-4, optimizer_type='adam', momentum=0.9, weight_decay=0.01, checkpoint_path='fine_tuned_checkpoint.pth', resume=False, save_best=True, save_current=True, best_val_loss=float('inf')):
    """
    Fine-tunes the model on the training dataset and evaluates on the validation dataset.
    """
    try:
        # Select optimizer
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Define criterion and scheduler
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
        start_epoch = 0
        final_metrics = {}

        # Load checkpoint if resuming
        if resume:
            model, optimizer, start_epoch = load_model(model, optimizer, checkpoint_path)

        # Fine-tuning loop
        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()

            # Training phase
            model.train()
            total_loss = 0
            train_preds, train_labels = [], []
            train_pred_proba = []
            # Add a tqdm progress bar for the training phase
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Fine Tuning]", unit="batch") as pbar:
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
                    train_pred_proba.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

                    # Update progress bar
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

            train_metrics = calculate_metrics(train_labels, train_preds, train_pred_proba)
            train_time = time.time() - start_time
            logging.info(f"Train time: {train_time:.2f} seconds")

            # Validation phase
            val_metrics = evaluate_model(model, val_loader, device, criterion)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Metrics: {train_metrics}")
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Metrics: {val_metrics}")

            print(f"Epoch {epoch+1}/{num_epochs}, Train Metrics: {train_metrics}")
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Metrics: {val_metrics}")

            # Adjust learning rate based on validation loss
            scheduler.step(val_metrics['loss'])

            # Save current checkpoint
            if save_current:
                save_model(model, optimizer, epoch + 1, checkpoint_path)

            # Save best checkpoint
            if save_best and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_model(model, optimizer, epoch + 1, f'{model_type}_fine_tuned_best_model.pth')
                logging.info(f"Best model updated at epoch {epoch+1} with val_loss {best_val_loss:.4f}")
                print(f"Best model updated at epoch {epoch+1} with val_loss {best_val_loss:.4f}")

            # Update final metrics
            final_metrics = {
                'train': train_metrics,
                'val': val_metrics
            }

        return final_metrics
    except Exception as e:
        logging.error(f"Error during fine-tuning: {str(e)}")
        return None


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