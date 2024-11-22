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
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
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

    elif model_type == 'CNN_BiLSTM':
        return CNN_BiLSTM(
            input_dim=input_dim,
            cnn_out_channels=cnn_out_channels,
            lstm_hidden_dim=hidden_dim,
            lstm_layers=num_layers,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'BiLSTM':
        return BiLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
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

    elif model_type == 'CNN_BiGRU':
        return CNN_BiGRU(
            input_dim=input_dim,
            cnn_out_channels=cnn_out_channels,
            gru_hidden_dim=hidden_dim,
            gru_layers=num_layers,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'BiGRU':
        return BiGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(device)

    elif model_type == 'CNN':
        return CNN(
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


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Train, Test, Fine-tune, and Explain Models with Various Architectures.')
    
    # Model configuration parameters
    parser.add_argument('--model_type', type=str, default='CNN_BiLSTM_Attention', 
                        choices=['CNN_BiLSTM_Attention', 'CNN_BiGRU_Attention_Model', 'BiLSTM', 'BiGRU', 'CNN', 'CNN_BiGRU', 'CNN_BiLSTM', 'RandomForest', 'XGBoost'],
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
        data_splits = create_data_splits(X, y, k_folds=args.k_folds, random_state=42)
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
            'learning_rate': [0.01, 0.1]  # Only applicable to XGBoost
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

        
    # Cross-validation training phase
    if args.train:
        best_val_loss = float('inf')
        fold_metrics = []
        aggregated_confusion_matrix = None  # 用于汇总所有折的混淆矩阵

        for fold, (train_idx, val_idx) in enumerate(train_val_indices, 1):
            # 获取当前折的训练和验证数据
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            train_loader, val_loader = get_dataloaders(X_train_fold, y_train_fold, batch_size=args.batch_size, sample_size=args.sample_size)

            # 模型剪枝（可选）
            if args.prune > 0.0 and args.model_type in ['CNN_BiLSTM_Attention', 'CNN_BiGRU_Attention_Model', 'BiLSTM', 'BiGRU', 'CNN', 'CNN_BiGRU', 'CNN_BiLSTM']:
                apply_pruning(model, amount=args.prune)

            dl_checkpoint_path = f"{args.model_type}_fold_{fold}_checkpoint.pth"
            ml_checkpoint_path = f"{args.model_type}_fold_{fold}_checkpoint.pkl"

            # 训练和验证模型
            if args.model_type in ['RandomForest', 'XGBoost']:
                metrics = train_baseline_model(model, args.model_type, X_train_fold, y_train_fold, X_val_fold, y_val_fold, ml_checkpoint_path, save_best=True, best_val_loss=best_val_loss)
            else:
                metrics = train_model(
                    model=model,
                    model_type=args.model_type,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    num_epochs=args.epochs,
                    lr=args.lr, 
                    optimizer_type=args.optimizer, 
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    checkpoint_path=dl_checkpoint_path,
                    resume=args.resume,
                    save_best=True, 
                    save_current=args.save_current,
                    best_val_loss=best_val_loss
                )

            # 检查是否成功返回指标
            if metrics:
                fold_metrics.append(metrics)

                # 获取每折的混淆矩阵并累加
                fold_cm = np.array(metrics['val']['confusion_matrix'])
                if aggregated_confusion_matrix is None:
                    aggregated_confusion_matrix = fold_cm
                else:
                    aggregated_confusion_matrix += fold_cm

                print(f"Fold {fold} Confusion Matrix:\n{fold_cm}")
            else:
                logging.error(f"No metrics for Fold {fold}")

            # 每折重新初始化模型
            model = initialize_model(
                model_type=args.model_type,
                input_dim=input_dim,
                output_dim=output_dim,
                device=device,
                cnn_out_channels=args.cnn_out_channels,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers
            )

        # 汇总每折的指标
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
                **{f'val_{k}': v for k, v in avg_metrics['val'].items()},
                'aggregated_confusion_matrix': aggregated_confusion_matrix.tolist(),  # 转换为列表以便JSON序列化
            }

            # 记录汇总结果
            log_results(results)
            print(f"Aggregated Confusion Matrix:\n{aggregated_confusion_matrix}")
        else:
            logging.error("No valid fold metrics collected")

    # Fine-tuning phase
    if args.fine_tune:
        logging.info(f"Fine-tuning data directory: {args.fine_tune_data_dir}")
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
        logging.info(f"Test data directory: {args.test_data_dir}")
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
    
    features_to_remove = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp']

    if args.explain:
        if not os.path.exists(args.explain_checkpoint_path):
            logging.error(f"Checkpoint not found at {args.explain_checkpoint_path} for explanation phase.")
        else:
            # Load the model and explain data
            if args.model_type in ['RandomForest', 'XGBoost']:
                model = joblib.load(args.explain_checkpoint_path)
                model_type = 'tree'
            else:
                model, _, _ = load_model(model, file_path=args.explain_checkpoint_path)
                model_type = 'torch'

            # Load explain data
            explain_data = load_data_from_directory(args.explain_data_dir)

            # Drop the specified features
            explain_data = explain_data.drop(columns=features_to_remove, errors='ignore')

            # Preprocess the data
            X_explain, y_explain, _, label_encoder = preprocess_data(explain_data)

            # Retrieve feature names dynamically or use predefined
            feature_names = explain_data.columns[:-1].tolist()
            if not feature_names:
                feature_names = [f'Feature_{i}' for i in range(X_explain.shape[1])]

            # Retrieve class names from the label encoder
            if label_encoder:
                class_names = label_encoder.classes_.tolist()
            else:
                class_names = [f'Class_{i}' for i in range(output_dim)]

            # Get predictions and identify correctly/misclassified samples
            if model_type == 'tree':
                y_pred = model.predict(X_explain)
            else:
                model.eval()
                with torch.no_grad():
                    batch_size = 1024  # Adjust batch size as needed to fit into GPU memory
                    y_pred_probs = []

                    for i in range(0, len(X_explain), batch_size):
                        batch = torch.tensor(X_explain[i:i+batch_size], dtype=torch.float32).to(device)
                        with torch.no_grad():
                            batch_probs = model(batch)
                        y_pred_probs.append(batch_probs.cpu())

                    y_pred_probs = torch.cat(y_pred_probs)  # Combine all batches
                    y_pred = torch.argmax(y_pred_probs, dim=1).cpu().numpy()

            # Identify indices for misclassified and correctly classified samples
            misclassified_indices = np.where(y_pred != y_explain)[0]
            correctly_classified_indices = np.where(y_pred == y_explain)[0]

            # Limit to the first 100 samples for explanation (or customize as needed)
            misclassified_indices = misclassified_indices[:100]
            correctly_classified_indices = correctly_classified_indices[:100]

            # Prepare datasets for SHAP explanations
            X_misclassified = X_explain[misclassified_indices]
            X_correct = X_explain[correctly_classified_indices]

            logging.info(f"Number of misclassified samples: {len(misclassified_indices)}")
            logging.info(f"Number of correctly classified samples: {len(correctly_classified_indices)}")

            explain_data_dir = args.explain_data_dir
            explain_data_dir = explain_data_dir.lstrip("./csv_output/")
            explain_data_dir = explain_data_dir.replace("/", "_")

            # Generate SHAP explanations for misclassified samples
            if len(X_misclassified) > 0:
                shap_values_misclassified = explain_with_shap(
                    model=model, 
                    X_sample=X_misclassified, 
                    device=device, 
                    feature_names=feature_names, 
                    class_names=class_names,
                    origin_model_type=args.model_type,
                    model_type=model_type,
                    db_name=explain_data_dir,
                    description='misclassified_samples'
                )
                logging.info("SHAP explanations generated for misclassified samples.")

            # Generate SHAP explanations for correctly classified samples
            if len(X_correct) > 0:
                shap_values_correct = explain_with_shap(
                    model=model, 
                    X_sample=X_correct, 
                    device=device, 
                    feature_names=feature_names, 
                    class_names=class_names, 
                    origin_model_type=args.model_type,
                    model_type=model_type,
                    db_name=explain_data_dir,
                    description='correct_samples'
                )
                logging.info("SHAP explanations generated for correctly classified samples.")

    logging.info("Completed all phases")

if __name__ == '__main__':
    main()
