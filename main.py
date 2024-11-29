import joblib
import torch
import logging
import argparse
import os
import numpy as np
from torch.optim import Adam
from phase import explain_phase, fine_tune_phase, test_phase, train_phase
from train import calculate_metrics, train_model, apply_pruning, load_model, test_model, fine_tune_model, train_baseline_model
from explain import explain_with_shap
from data_preprocessing import get_dataloaders, get_test_loader, load_data_from_directory, preprocess_data
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils import create_data_splits, create_fine_tune_splits, hyperparameter_search, initialize_model, log_results, set_seed

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Train, Test, Fine-tune, and Explain Models with Various Architectures.')
    
    # Model configuration parameters
    parser.add_argument('--model_type', type=str, default='CNN_BiLSTM_Attention', 
                        choices=['CNN_BiLSTM_Attention', 'CNN_BiGRU_Attention', 'CNN_Attention', 'BiGRU_Attention', 'BiLSTM_Attention','CNN_BiGRU', 'CNN_BiLSTM', 'BiLSTM', 'BiGRU', 'CNN', 'RandomForest', 'XGBoost'],
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
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation.')
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
    parser.add_argument('--use_exist', action='store_true', help='')

    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename='logs/training.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Started the training, testing, fine-tuning, and explanation process.")
    logging.info(f"Using model type: {args.model_type}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.data_dir:
        # Load and preprocess data
        logging.info(f"Training data directory: {args.data_dir}")
        combined_data = load_data_from_directory(args.data_dir)
        X, y, scaler, label_encoder = preprocess_data(combined_data)
        input_dim = X.shape[1]
        output_dim = len(label_encoder.classes_)
        # Create or load consistent train-validation splits
        data_splits = create_data_splits(args.use_exist, X, y, k_folds=args.k_folds, random_state=42)
        train_val_indices = data_splits['train_val_indices']
        X_train, y_train = data_splits['train_data']
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

    if args.train:
        train_phase(args, model, input_dim, output_dim, X_train, y_train, train_val_indices, device)

    if args.fine_tune:
        fine_tune_phase(args, model, device)

    if args.test:
        test_phase(args, model, device)

    if args.explain:
        explain_phase(args, model, device)
           
    logging.info("Completed all phases")

if __name__ == '__main__':
    main()
