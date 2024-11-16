import joblib
import torch
import logging
import argparse
import os
import numpy as np
from torch.optim import Adam
import csv

from model import CNN, BiGRU, CNN_BiGRU, CNN_BiGRU_Attention_Model, CNN_BiLSTM, CNN_BiLSTM_Attention_Model, BiLSTM
from train import calculate_metrics, save_model, train_model, apply_pruning, load_model, test_model, fine_tune_model, train_baseline_model
from explain import explain_with_shap
from data_preprocessing import get_dataloaders, get_test_loader, load_data_from_directory, preprocess_data
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

def create_data_splits(X, y, k_folds=5, random_state=42, split_file='data_splits.pkl'):
    if os.path.exists(split_file):
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

def create_fine_tune_splits(X_fine_tune, y_fine_tune, test_size=0.1, random_state=42, split_file='fine_tune_splits.pkl'):
    if os.path.exists(split_file):
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

def initialize_model(model_type, input_dim, output_dim, device, random_state=42):
    """
    Initialize the model based on the specified model type.

    Args:
        model_type (str): The type of model to initialize.
        input_dim (int): The input dimension of the model.
        output_dim (int): The output dimension of the model.
        device (torch.device): The device to which the model should be moved.
        random_state (int): The random state for reproducibility (for ML models).

    Returns:
        model: The initialized model.
    """
    if model_type == 'CNN_BiLSTM_Attention':
        return CNN_BiLSTM_Attention_Model(
            input_dim=input_dim,
            cnn_out_channels=64,
            lstm_hidden_dim=64,
            lstm_layers=2,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'CNN_BiLSTM':
        return CNN_BiLSTM(
            input_dim=input_dim,
            cnn_out_channels=64,
            lstm_hidden_dim=64,
            lstm_layers=2,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'BiLSTM':
        return BiLSTM(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'CNN_BiGRU_Attention':
        return CNN_BiGRU_Attention_Model(
            input_dim=input_dim,
            cnn_out_channels=64,
            gru_hidden_dim=64,
            gru_layers=2,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'CNN_BiGRU':
        return CNN_BiGRU(
            input_dim=input_dim,
            cnn_out_channels=64,
            gru_hidden_dim=64,
            gru_layers=2,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'BiGRU':
        return BiGRU(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'CNN':
        return CNN(
            input_dim=input_dim,
            cnn_out_channels=64,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'RandomForest':
        return RandomForestClassifier(n_estimators=100, random_state=random_state)

    elif model_type == 'XGBoost':
        return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Train, Test, Fine-tune, and Explain Models with Various Architectures.')
    
    parser.add_argument('--model_type', type=str, default='CNN_BiLSTM_Attention', 
                        choices=['CNN_BiLSTM_Attention', 'CNN_BiGRU_Attention_Model', 'BiLSTM', 'BiGRU', 'CNN', 'CNN_BiGRU', 'CNN_BiLSTM'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--explain', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prune', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--fine_tune_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sample_size', type=float)
    parser.add_argument('--data_dir', type=str, default='./csv_output/CIRA-CIC-DoHBrw-2020')
    parser.add_argument('--fine_tune_data_dir', type=str)
    parser.add_argument('--test_data_dir', type=str, default='./csv_output/doh_dataset')
    parser.add_argument('--checkpoint_path', type=str, default='model_checkpoint.pth')
    parser.add_argument('--best_checkpoint_path', type=str, default='best_model_checkpoint.pth')
    parser.add_argument('--test_checkpoint_path', type=str, default='test_model_checkpoint.pth')
    parser.add_argument('--fine_tuned_model_checkpoint_path', type=str, default='fine_tuned_model_checkpoint_path.pth')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename='logs/training.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Started the training, testing, fine-tuning, and explanation process.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    combined_data = load_data_from_directory(args.data_dir)
    X, y, scaler, label_encoder = preprocess_data(combined_data)
    input_dim = X.shape[1]
    output_dim = len(label_encoder.classes_)

    # Create or load consistent train-validation splits
    data_splits = create_data_splits(X, y, k_folds=args.k_folds, random_state=42)
    train_val_indices = data_splits['train_val_indices']
    X_train, y_train = data_splits['train_data']

    # Initialize model
    model = initialize_model(
        model_type=args.model_type,
        input_dim=input_dim,
        output_dim=output_dim,
        device=device
    )

    # Cross-validation training phase
    if args.train:
        best_val_loss = float('inf')
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(train_val_indices, 1):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            train_loader, val_loader = get_dataloaders(X_train_fold, y_train_fold, batch_size=args.batch_size, sample_size=args.sample_size)

            if args.prune > 0.0 and args.model_type in ['CNN_BiLSTM_Attention', 'CNN_BiGRU_Attention_Model', 'BiLSTM', 'BiGRU', 'CNN', 'CNN_BiGRU', 'CNN_BiLSTM']:
                apply_pruning(model, amount=args.prune)

            dl_checkpoint_path = f"{args.model_type}_fold_{fold}_checkpoint.pth"
            ml_checkpoint_path = f"{args.model_type}_fold_{fold}_checkpoint.pkl"
            if args.model_type in ['RandomForest', 'XGBoost']:
                metrics = train_baseline_model(model, args.model_type, X_train_fold, y_train_fold, X_val_fold, y_val_fold, ml_checkpoint_path, save_best=True, best_val_loss=best_val_loss)
            else:
                metrics = train_model(model, args.model_type, train_loader, val_loader, device, args.epochs, lr=1e-4, checkpoint_path=dl_checkpoint_path, save_best=True, best_val_loss=best_val_loss)

            if metrics:
                fold_metrics.append(metrics)
            else:
                logging.error(f"No metrics for Fold {fold}")
            # Reinitialize model
            model = initialize_model(
                model_type=args.model_type,
                input_dim=input_dim,
                output_dim=output_dim,
                device=device
            )
        valid_metrics = [m for m in fold_metrics if m]
        if valid_metrics:
            avg_metrics = {
                'train': {metric: np.mean([fold['train'][metric] for fold in valid_metrics]) for metric in valid_metrics[0]['train']},
                'val': {metric: np.mean([fold['val'][metric] for fold in valid_metrics]) for metric in valid_metrics[0]['val']}
            }
            results = {
                'model_type': args.model_type,
                'phase': 'train',
                'data_dir': args.data_dir,
                **{f'train_{k}': v for k, v in avg_metrics['train'].items()},
                **{f'val_{k}': v for k, v in avg_metrics['val'].items()}
            }
            log_results(results)
        else:
            logging.error("No valid fold metrics collected")

    # Fine-tuning phase
    if args.fine_tune:
        # 1. Load and preprocess fine-tuning data
        fine_tune_data = load_data_from_directory(args.fine_tune_data_dir)
        X_fine_tune, y_fine_tune, _, _ = preprocess_data(fine_tune_data)
        
        # 2. Load or create fixed fine-tuning splits
        fine_tune_splits = create_fine_tune_splits(X_fine_tune, y_fine_tune)
        fine_tune_train_indices = fine_tune_splits['fine_tune_train_indices']
        fine_tune_val_indices = fine_tune_splits['fine_tune_val_indices']
        
        # Use consistent fine-tuning split across models
        X_fine_train, y_fine_train = X_fine_tune[fine_tune_train_indices], y_fine_tune[fine_tune_train_indices]
        X_fine_val, y_fine_val = X_fine_tune[fine_tune_val_indices], y_fine_tune[fine_tune_val_indices]
        
        fine_tune_train_loader, fine_tune_val_loader = get_dataloaders(X_fine_train, y_fine_train, batch_size=args.batch_size, sample_size=args.sample_size)

        if args.model_type in ['CNN_BiLSTM_Attention', 'CNN_BiGRU_Attention_Model', 'BiLSTM', 'BiGRU', 'CNN', 'CNN_BiGRU', 'CNN_BiLSTM']:
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
            fine_tune_metrics = fine_tune_model(model, fine_tune_train_loader, fine_tune_val_loader, device, optimizer=optimizer, num_epochs=args.fine_tune_epochs, pretrained_path=args.best_checkpoint_path)
            # Save model
            fine_tuned_model_path = f"{args.model_type}_fine_tuned_model.pth"
            torch.save(model.state_dict(), fine_tuned_model_path)
   
            log_results({'model_type': args.model_type, 'phase': 'fine_tune', 'data_dir': args.fine_tune_data_dir, **fine_tune_metrics})

        elif args.model_type in ['RandomForest', 'XGBoost']:
            # model_file = f'{args.model_type}_model.pkl'
            model_file = args.best_checkpoint_path
            fine_tuned_model_file = f'{args.model_type}_fine_tuned_model.pkl'

            if os.path.exists(model_file):
                fine_tuned_model = joblib.load(model_file)
            else:
                fine_tuned_model = RandomForestClassifier(n_estimators=100, random_state=42) if args.model_type == 'RandomForest' else XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

            fine_tuned_model.fit(X_fine_train, y_fine_train)
            y_fine_pred = fine_tuned_model.predict(X_fine_val)
            y_fine_pred_proba = fine_tuned_model.predict_proba(X_fine_val) if hasattr(fine_tuned_model, "predict_proba") else None
            fine_tune_metrics = calculate_metrics(y_fine_val, y_fine_pred, y_fine_pred_proba)
            joblib.dump(fine_tuned_model, fine_tuned_model_file)
            log_results({'model_type': args.model_type, 'phase': 'fine_tune', 'data_dir': args.fine_tune_data_dir, **fine_tune_metrics})


    # Testing phase on new test data
    if args.test:
        test_data = load_data_from_directory(args.test_data_dir)
        X_test, y_test, _, _ = preprocess_data(test_data)

        if args.model_type in ['RandomForest', 'XGBoost']:
            model = joblib.load(args.test_checkpoint_path)
            y_test_pred = model.predict(X_test)
            y_test_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
            logging.info(f"{args.model_type} Test Metrics: {metrics}")

        else:
            test_loader = get_test_loader(X_test, y_test, batch_size=args.batch_size)
            model, _, _ = load_model(model, file_path=args.test_checkpoint_path)
            metrics = test_model(model, test_loader, device)
            logging.info(f"{args.model_type} Test Metrics: {metrics}")

        log_results({'model_type': args.model_type, 'phase': 'test', 'data_dir': args.test_data_dir, **metrics})

    logging.info("Completed all phases")

if __name__ == '__main__':
    main()
