import coloredlogs
import torch
import logging
import argparse
import os
import numpy as np
from phase import explain_phase, fine_tune_phase, test_phase, train_phase
from data_preprocessing import load_data_from_directory, preprocess_data
from utils import create_data_splits, hyperparameter_search, initialize_model, set_seed
from datetime import datetime


def setup_logger(model_type):
    os.makedirs('logs', exist_ok=True)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    
    coloredlogs.install(
        level='INFO',
        fmt=log_format,
        datefmt=datefmt,
        logger=logging.getLogger()
    )
    
    file_handler = logging.FileHandler(f'logs/{model_type}.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, datefmt=datefmt)
    file_handler.setFormatter(file_formatter)
    
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Logger initialized for model: {model_type}")


def log_phase(phase_name, status="start"):
    """Logs the start or end of a phase with clear and aligned formatting."""
    phase_name = phase_name.upper()  # Optional: Make the phase name all caps for emphasis
    border = '=' * 100  # Border length
    
    # Prepare the log message
    if status == "start":
        log_message = f"{'STARTING PHASE:':<25} {phase_name:>70}"
    elif status == "end":
        log_message = f"{'COMPLETED PHASE:':<25} {phase_name:>70}"
    
    # Log the message with borders
    logging.info(f"\n{border}\n{log_message}\n{border}\n")

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Train, Test, Fine-tune, and Explain Models with Various Architectures.')
    
    # Model configuration parameters
    parser.add_argument('--model_type', type=str, default='CNN_BiLSTM_Attention', 
                        choices=['CNN_BiLSTM_Attention', 'CNN_BiGRU_Attention', 'CNN_LSTM_Attention', 'CNN_GRU_Attention', 'CNN_GRU', 'CNN_BiLSTM', 'CNN_Attention', 'BiGRU_Attention', 'BiLSTM_Attention','CNN_BiGRU', 'CNN_BiLSTM', 'BiLSTM', 'BiGRU', 'LSTM', 'GRU', 'CNN', 'RNN', 'DNN', 'MLP', 'Transformer', 'RandomForest', 'XGBoost'],
                        help='Type of model architecture to use (e.g., CNN_BiLSTM_Attention, BiGRU, etc.).')
    parser.add_argument('--cnn_out_channels', type=int, default=64, help='Number of output channels for CNN layers.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden size for LSTM/GRU layers.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers for LSTM/GRU.')

    # Training-related flags
    parser.add_argument('--train', action='store_true', help='Flag to train the model.')
    parser.add_argument('--fine_tune', action='store_true', help='Flag to fine-tune the model.')
    parser.add_argument('--test', action='store_true', help='Flag to test the model.')
    parser.add_argument('--resume', action='store_true', help='Resume training from a saved checkpoint.')
    parser.add_argument('--save_current', action='store_true', default=True, help='Flags to save the current model.')

    # Training parameters
    parser.add_argument('--prune', type=float, default=0.3, help='Pruning ratio for model compression.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--k_folds', type=int, default=2, help='Number of folds for cross-validation.')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of epochs for fine-tuning.')

    # Optimizer configuration
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'sgd', 'rmsprop'], help='Optimizer to use for training.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for regularization.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training.')

    # Data and file paths
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation.')
    parser.add_argument('--sample_size', type=float, help='Fraction of training data to use (0 < sample_size <= 1).')
    parser.add_argument('--data_dir', type=str, default='./csv_output/CIRA-CIC-DoHBrw-2020', help='Directory containing training data.')
    parser.add_argument('--fine_tune_data_dir', type=str, help='Directory containing fine-tuning data.')
    parser.add_argument('--test_data_dir', type=str, default='./csv_output/doh_dataset', help='Directory containing test data.')
    parser.add_argument('--explain_data_dir', type=str, default='./csv_output/doh_dataset', help='Directory containing explain data.')
    parser.add_argument('--checkpoint_path', type=str, default='model_checkpoint.pth', help='Path to save training checkpoint.')
    parser.add_argument('--best_checkpoint_path', type=str, default='best_model_checkpoint.pth', help='Path to save the best model checkpoint.')
    parser.add_argument('--test_checkpoint_path', type=str, default='test_model_checkpoint.pth', help='Path to load checkpoint for testing.')
    parser.add_argument('--fine_tuned_model_checkpoint_path', type=str, default='fine_tuned_model_checkpoint_path.pth', help='Path to save fine-tuned model checkpoint.')

    # Hyperparameter_search
    parser.add_argument('--hyperparameter_search', action='store_true', help='Enable hyperparameter search.')

    # Explain
    parser.add_argument('--explain', action='store_true', help='Enable explainability for model predictions.')
    parser.add_argument('--explain_checkpoint_path', type=str, default='explain_checkpoint.pth', help='Path to load checkpoint for explaining.')
    parser.add_argument('--use_exist', default=True, action='store_true', help='')

    args = parser.parse_args()
    if args.model_type:
        # setup_logger('all')
        setup_logger(args.model_type)

    log_phase("Overall Process", "start")
    logging.info(f"Using model type: {args.model_type}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.data_dir:
        # Load and preprocess data
        log_phase("Data Loading", "start")
        logging.info(f"Training data directory: {args.data_dir}")
        combined_data = load_data_from_directory(args.data_dir)
        X, y, scaler, label_encoder = preprocess_data(combined_data)
        input_dim = X.shape[1]
        output_dim = len(label_encoder.classes_)
        # Create or load consistent train-validation splits
        data_splits = create_data_splits(args.use_exist, X, y, k_folds=args.k_folds, random_state=42)
        train_val_indices = data_splits['train_val_indices']
        X_train, y_train = data_splits['train_data']
        log_phase("Data Loading", "end")
    else:
        logging.error("No data_dir is setted!")

    model = initialize_model(
        model_type=args.model_type,
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
        cnn_out_channels=args.cnn_out_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )

    # Hyperparameter search
    if args.hyperparameter_search:
        log_phase("Hyperparameter Search", "start")
        param_grid_tree = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'learning_rate': [0.001, 0.01, 0.1]  # Only applicable to tree-base model
        }
        param_grid_dl = {
            'cnn_out_channels': [32, 64, 128],
            'hidden_dim': [64, 128],
            'num_layers': [1, 2],
            'batch_size': [32, 64],
            'lr': [1e-4, 1e-3, 1e-2],
            'weight_decay': [0.01, 0.001],
            'epochs': [30, 50]
        }

        best_params, best_metrics = hyperparameter_search(
            param_grid_tree=param_grid_tree,
            param_grid_dl=param_grid_dl,
            model_type=args.model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            X_train=X,
            y_train=y,
            device=device,
            args=args
        )
        logging.info(f"Hyperparameter Search Completed. Best Params: {best_params}, Best Metrics: {best_metrics}")
        log_phase("Hyperparameter Search", "end")

    if args.train:
        log_phase("Training", "start")
        train_phase(args, model, input_dim, output_dim, X_train, y_train, train_val_indices, device)
        log_phase("Training", "end")

    if args.fine_tune:
        log_phase("Fine-Tuning", "start")
        fine_tune_phase(args, model, device)
        log_phase("Fine-Tuning", "end")

    if args.test:
        log_phase("Testing", "start")
        test_phase(args, model, device)
        log_phase("Testing", "end")

    if args.explain:
        log_phase("Explainability", "start")
        explain_phase(args, model, device)
        log_phase("Explainability", "end")

    log_phase("Overall Process", "end")

if __name__ == '__main__':
    main()
