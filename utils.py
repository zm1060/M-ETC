import logging
import os
import joblib
import torch
import numpy as np
from torch.optim import Adam
import csv
import matplotlib.pyplot as plt

from model import BiGRU_Attention_Model, BiGRU_Model, BiLSTM_Attention_Model, BiLSTM_Model, CNN_Attention_Model, CNN_BiGRU_Attention_Model, CNN_BiGRU_Model, CNN_BiLSTM_Model, CNN_BiLSTM_Attention_Model, CNN_Model
from train import calculate_metrics, train_model
from explain import explain_with_shap
from data_preprocessing import get_dataloaders
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def log_results(results, log_file='model_performance_log.csv'):
    file_exists = os.path.isfile(log_file)
    headers = results.keys()
    with open(log_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_data_splits(use_exist, X, y, k_folds=5, random_state=42, split_file='data_splits.pkl'):
    if use_exist and os.path.exists(split_file):
        splits = joblib.load(split_file)
        return splits

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    fold_indices = list(skf.split(X, y))

    splits = {
        'train_val_indices': fold_indices,
        'train_data': (X, y)
    }

    joblib.dump(splits, split_file)
    return splits

def create_fine_tune_splits(use_exist, X_fine_tune, y_fine_tune, test_size=0.1, random_state=42, split_file='fine_tune_splits.pkl'):
    if use_exist and os.path.exists(split_file):
        splits = joblib.load(split_file)
        return splits

    fine_tune_train_indices, fine_tune_val_indices, _, _ = train_test_split(
        np.arange(len(y_fine_tune)), y_fine_tune, test_size=test_size, random_state=random_state, stratify=y_fine_tune
    )

    splits = {
        'fine_tune_train_indices': fine_tune_train_indices,
        'fine_tune_val_indices': fine_tune_val_indices
    }

    joblib.dump(splits, split_file)
    return splits

def initialize_model(model_type, input_dim, output_dim, device, cnn_out_channels=64, hidden_dim=64, num_layers=2, random_state=42):
    """
    Initialize the model based on the specified model type and dynamic hyperparameters.

    Args:
        model_type (str): The type of model to initialize.
        input_dim (int): The input dimension of the model.
        output_dim (int): The output dimension of the model.
        device (torch.device): The device to which the model should be moved.
        cnn_out_channels (int): The number of output channels for CNN layers.
        hidden_dim (int): The hidden size for LSTM/GRU layers.
        num_layers (int): The number of layers for LSTM/GRU.
        random_state (int): The random state for reproducibility (for ML models).

    Returns:
        model: The initialized model.
    """
    if model_type == 'CNN_BiLSTM_Attention':
        return CNN_BiLSTM_Attention_Model(
            input_dim=input_dim,
            cnn_out_channels=cnn_out_channels,
            lstm_hidden_dim=hidden_dim,
            lstm_layers=num_layers,
            output_dim=output_dim
        ).to(device)
    
    elif model_type == 'CNN_BiGRU_Attention':
        return CNN_BiGRU_Attention_Model(
            input_dim=input_dim,
            cnn_out_channels=cnn_out_channels,
            gru_hidden_dim=hidden_dim,
            gru_layers=num_layers,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'CNN_BiLSTM':
        return CNN_BiLSTM_Model(
            input_dim=input_dim,
            cnn_out_channels=cnn_out_channels,
            lstm_hidden_dim=hidden_dim,
            lstm_layers=num_layers,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'CNN_BiGRU':
        return CNN_BiGRU_Model(
            input_dim=input_dim,
            cnn_out_channels=cnn_out_channels,
            gru_hidden_dim=hidden_dim,
            gru_layers=num_layers,
            output_dim=output_dim
        ).to(device)
    
    elif model_type == 'BiLSTM_Attention':
        return BiLSTM_Attention_Model(
            input_dim=input_dim,
            lstm_hidden_dim=hidden_dim,
            lstm_layers=num_layers,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'BiGRU_Attention':
        return BiGRU_Attention_Model(
            input_dim=input_dim,
            gru_hidden_dim=hidden_dim,
            gru_layers=num_layers,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'CNN_Attention':
        return CNN_Attention_Model(
            input_dim=input_dim,
            cnn_out_channels=cnn_out_channels,
            output_dim=output_dim
        ).to(device)


    elif model_type == 'BiLSTM':
        return BiLSTM_Model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(device)
    
    elif model_type == 'BiGRU':
        return BiGRU_Model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'CNN':
        return CNN_Model(
            input_dim=input_dim,
            cnn_out_channels=cnn_out_channels,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'RandomForest':
        return RandomForestClassifier(n_estimators=100, random_state=random_state)

    elif model_type == 'XGBoost':
        return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_optimizer(model, args):
    """
    Get the optimizer based on the user-specified type and hyperparameters.

    Args:
        model: The PyTorch model whose parameters will be optimized.
        args: Parsed command-line arguments containing optimizer configurations.

    Returns:
        torch.optim.Optimizer: Configured optimizer for the model.

    Raises:
        ValueError: If an unsupported optimizer type is specified in args.
    """
    if args.optimizer == 'adam':
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        from torch.optim import SGD
        return SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        from torch.optim import RMSprop
        return RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

def hyperparameter_search(param_grid_tree, param_grid_dl, model_type, input_dim, output_dim, X_train, y_train, device, args):
    best_metrics = None
    best_params = None

    if model_type in ['RandomForest', 'XGBoost']:
        # Tree-based model hyperparameter search
        for params in ParameterGrid(param_grid_tree):
            logging.info(f"Testing tree-based hyperparameters: {params}")

            # Initialize model
            model = initialize_model(
                model_type=model_type,
                input_dim=input_dim,
                output_dim=output_dim,
                device=device,
                random_state=42
            )
            
            # Apply parameter grid settings
            model.set_params(**{k: v for k, v in params.items() if k in model.get_params()})
            
            # Split data for validation
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # Train and evaluate
            model.fit(X_train_split, y_train_split)
            y_val_pred = model.predict(X_val_split)
            y_val_pred_proba = model.predict_proba(X_val_split) if hasattr(model, "predict_proba") else None
            
            # Calculate metrics
            metrics = calculate_metrics(y_val_split, y_val_pred, y_val_pred_proba)
            logging.info(f"Tree-based model metrics: {metrics}")

            # Compare based on accuracy or another metric
            if not best_metrics or metrics['accuracy'] > best_metrics['accuracy']:
                best_metrics = metrics
                best_params = params

    else:
        # Deep learning model hyperparameter search
        for params in ParameterGrid(param_grid_dl):
            logging.info(f"Testing deep learning hyperparameters: {params}")

            # Initialize model
            model = initialize_model(
                model_type=model_type,
                input_dim=input_dim,
                output_dim=output_dim,
                device=device,
                cnn_out_channels=params.get('cnn_out_channels', args.cnn_out_channels),
                hidden_dim=params.get('hidden_dim', args.hidden_dim),
                num_layers=params.get('num_layers', args.num_layers)
            )

            # Dynamically configure optimizer and other training hyperparameters
            optimizer = Adam(
                model.parameters(),
                lr=params.get('lr', args.lr),
                weight_decay=params.get('weight_decay', args.weight_decay)
            )
            
            train_loader, val_loader = get_dataloaders(
                X_train,
                y_train,
                batch_size=params.get('batch_size', args.batch_size),
                sample_size=args.sample_size
            )

            # Train and evaluate
            metrics = train_model(
                model=model,
                model_type=model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=params.get('epochs', args.epochs),
                lr=params.get('lr', args.lr),  # Pass learning rate
                optimizer_type=args.optimizer,
                momentum=args.momentum,
                weight_decay=params.get('weight_decay', args.weight_decay),
                checkpoint_path=None,  # No need to save checkpoints during search
                resume=False,
                save_best=False,
                save_current=False,
                best_val_loss=None
            )
            logging.info(f"Deep learning model metrics: {metrics}")

            # Compare based on validation loss
            if not best_metrics or metrics['val']['loss'] < best_metrics['val']['loss']:
                best_metrics = metrics
                best_params = params

    logging.info(f"Best Hyperparameters: {best_params}")
    logging.info(f"Best Metrics: {best_metrics}")
    return best_params, best_metrics