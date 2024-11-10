# import joblib
# import torch
# import logging
# import argparse
# import os
# import numpy as np
# from torch.optim import Adam
# import csv

# from model import CNN_BiLSTM_Attention_Model, BiLSTM, get_baseline_models
# from train import save_model, train_model, apply_pruning, load_model, test_model, fine_tune_model, train_baseline_model
# from explain import explain_with_shap
# from data_preprocessing import get_dataloaders, load_data_from_directory, preprocess_data
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# def log_results(results, log_file='model_performance_log.csv'):
#     """
#     Logs the model performance results to a CSV file.
    
#     Parameters:
#         results (dict): Dictionary containing model name, parameters, dataset, and metrics.
#         log_file (str): Path to the CSV file where results will be logged.
#     """
#     file_exists = os.path.isfile(log_file)
    
#     headers = results.keys()
#     with open(log_file, mode='a', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=headers)
#         if not file_exists:
#             writer.writeheader()  # Write header if the file is new
#         writer.writerow(results)

# def main():
#     parser = argparse.ArgumentParser(description='Train, Test, Fine-tune, and Explain Models with Various Architectures.')
    
#     # Model selection parameter
#     parser.add_argument('--model_type', type=str, default='CNN_BiLSTM_Attention', 
#                         choices=['CNN_BiLSTM_Attention', 'BiLSTM', 'RandomForest', 'XGBoost'], 
#                         help='Specify the model type to use: CNN_BiLSTM_Attention, BiLSTM, RandomForest, or XGBoost.')
    
#     # Other arguments
#     parser.add_argument('--train', action='store_true', help='Flag to indicate whether to train the model.')
#     parser.add_argument('--fine_tune', action='store_true', help='Flag to indicate whether to fine-tune the model on new data.')
#     parser.add_argument('--test', action='store_true', help='Flag to indicate whether to test the model on a new dataset.')
#     parser.add_argument('--explain', action='store_true', help='Flag to indicate whether to explain the model.')
#     parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint.')
#     parser.add_argument('--prune', type=float, default=0.3, help='Fraction of weights to prune (0.3 means 30% pruning).')
#     parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model in each fold.')
#     parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation during training.')
#     parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of epochs for fine-tuning.')
#     parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing.')
#     parser.add_argument('--sample_size', type=float, help='Proportion or number of samples to use for fine-tuning (e.g., 0.1 for 10% or 100 for 100 samples).')
#     parser.add_argument('--data_dir', type=str, default='./csv_output/CIRA-CIC-DoHBrw-2020', help='Path to training data directory.')
#     parser.add_argument('--fine_tune_data_dir', type=str, help='Path to new data directory for fine-tuning.')
#     parser.add_argument('--test_data_dir', type=str, help='Path to testing data directory.')
#     parser.add_argument('--checkpoint_path', type=str, default='model_checkpoint.pth', help='Path to save/load the model checkpoint.')
#     parser.add_argument('--best_checkpoint_path', type=str, default='best_model_checkpoint.pth', help='Path to save/load the best model checkpoint.')

#     args = parser.parse_args()

#     os.makedirs('logs', exist_ok=True)
#     logging.basicConfig(filename='logs/training.log', level=logging.INFO, 
#                         format='%(asctime)s - %(levelname)s - %(message)s')
#     logging.info("Started the training, testing, fine-tuning, and explanation process.")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Load and preprocess training data
#     combined_data = load_data_from_directory(args.data_dir)
#     X, y, scaler, label_encoder = preprocess_data(combined_data)

#     input_dim = X.shape[1]
#     output_dim = len(label_encoder.classes_)

#     # Initialize cross-validation setup
#     skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
#     fold_metrics = []

#     # Cross-validation training phase
#     if args.train:
#         for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
#             print(f"Fold {fold}/{args.k_folds}")

#             # Split data for this fold
#             X_train, X_val = X[train_idx], X[val_idx]
#             y_train, y_val = y[train_idx], y[val_idx]
            
#             # Create data loaders for the fold
#             train_loader, val_loader = get_dataloaders(X_train, y_train, batch_size=args.batch_size, sample_size=args.sample_size)

#             # Initialize model based on user selection
#             if args.model_type == 'CNN_BiLSTM_Attention':
#                 model = CNN_BiLSTM_Attention_Model(input_dim=input_dim, cnn_out_channels=64, 
#                                                    lstm_hidden_dim=64, lstm_layers=2, output_dim=output_dim).to(device)
#             elif args.model_type == 'BiLSTM':
#                 model = BiLSTM(input_dim=input_dim, hidden_dim=64, output_dim=output_dim).to(device)
#             elif args.model_type in ['RandomForest', 'XGBoost']:
#                 models = get_baseline_models(input_dim, output_dim)
#                 model = models[args.model_type]
#             else:
#                 raise ValueError(f"Unsupported model type: {args.model_type}")

#             # Apply weight pruning (only for PyTorch models)
#             if args.prune > 0.0 and args.model_type in ['CNN_BiLSTM_Attention', 'BiLSTM']:
#                 apply_pruning(model, amount=args.prune)

#             # Train and evaluate the model for the fold
#             if args.model_type in ['RandomForest', 'XGBoost']:
#                 # Train and evaluate scikit-learn models
#                 metrics = train_baseline_model(model, args.model_type, X_train, y_train, X_val, y_val)
#                 fold_metrics.append(metrics)
#             else:
#                 # Save model for each fold with descriptive filename
#                 checkpoint_path = f"{args.model_type}_fold_{fold}_checkpoint.pth"
#                 best_checkpoint_path = f"{args.model_type}_best_fold_{fold}.pth"
                
#                 fold_train_metrics = train_model(
#                     model, train_loader, val_loader, device, num_epochs=args.epochs, lr=1e-4,
#                     checkpoint_path=checkpoint_path, save_best=True
#                 )

#                 # Save fold results and log
#                 if fold_train_metrics:
#                     fold_metrics.append(fold_train_metrics)
#                 else:
#                     logging.error(f"Metrics for Fold {fold} are None. Skipping this fold.")

#             print(f"Fold {fold} Metrics: {fold_metrics[-1]}")

#         # Filter out None values from fold_metrics
#         valid_fold_metrics = [fold for fold in fold_metrics if fold is not None]

#         # Ensure that there are valid metrics before averaging
#         if valid_fold_metrics:
#             # Average metrics across all folds
#             averaged_metrics = {metric: np.mean([fold[metric] for fold in valid_fold_metrics]) for metric in valid_fold_metrics[0]}
#             logging.info(f"Averaged Cross-Validation Metrics: {averaged_metrics}")

#             # Log averaged cross-validation results
#             results = {
#                 'model_type': args.model_type,
#                 'phase': 'cross_validation_train',
#                 'data_dir': args.data_dir,
#                 **averaged_metrics
#             }
#             log_results(results)
#         else:
#             raise ValueError("No valid metrics were collected across folds.")

#     # Fine-tuning phase
#     if args.fine_tune and args.model_type in ['CNN_BiLSTM_Attention', 'BiLSTM']:
#         if not args.fine_tune_data_dir:
#             raise ValueError("Please specify the path for the fine-tuning data directory using --fine_tune_data_dir.")
        
#         fine_tune_data = load_data_from_directory(args.fine_tune_data_dir)
#         X_fine_tune, y_fine_tune, _, _ = preprocess_data(fine_tune_data)
        
#         fine_tune_train_loader, fine_tune_val_loader = get_dataloaders(
#             X_fine_tune, y_fine_tune, batch_size=args.batch_size, sample_size=args.sample_size
#         )
        
#         optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        
#         fine_tune_metrics = fine_tune_model(
#             model, 
#             fine_tune_train_loader, 
#             fine_tune_val_loader, 
#             device, 
#             optimizer=optimizer, 
#             num_epochs=args.fine_tune_epochs, 
#             pretrained_path=args.best_checkpoint_path
#         )

#         results = {
#             'model_type': args.model_type,
#             'phase': 'fine_tune',
#             'data_dir': args.fine_tune_data_dir,
#             **fine_tune_metrics
#         }
#         log_results(results)

#     # Testing phase
#     if args.test:
#         if not args.test_data_dir:
#             raise ValueError("Please specify the path for the test data directory using --test_data_dir.")
        
#         # Load test data
#         test_combined_data = load_data_from_directory(args.test_data_dir)
#         X_test, y_test, _, _ = preprocess_data(test_combined_data)
        
#         if args.model_type in ['RandomForest', 'XGBoost']:
#             # Load the trained model
#             model = joblib.load(f'{args.model_type}_model.pkl')
            
#             # Make predictions on the test set
#             y_test_pred = model.predict(X_test)
#             y_test_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            
#             # Calculate performance metrics on the test set
#             test_accuracy = accuracy_score(y_test, y_test_pred)
#             test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
#             test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
#             test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            
#             # Calculate AUC score based on whether the task is binary or multi-class
#             if y_test_pred_proba is not None:
#                 if y_test_pred_proba.shape[1] == 2:  # Binary classification
#                     test_auc = roc_auc_score(y_test, y_test_pred_proba[:, 1])  # Use probabilities of the positive class
#                 else:  # Multi-class classification
#                     test_auc = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr')
#             else:
#                 test_auc = 'N/A'
            
#             # Log and output the test results
#             logging.info(f"{args.model_type} Test Metrics - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}, AUC: {test_auc}")
            
#             results = {
#                 'model_type': args.model_type,
#                 'phase': 'test',
#                 'data_dir': args.test_data_dir,
#                 'accuracy': test_accuracy,
#                 'precision': test_precision,
#                 'recall': test_recall,
#                 'f1_score': test_f1,
#                 'auc': test_auc,
#             }
#             log_results(results)
#         else:
#             # Testing process for PyTorch models
#             test_loader = get_dataloaders(X_test, y_test, batch_size=args.batch_size)[0]
#             model, _, _ = load_model(model, file_path=args.best_checkpoint_path)
#             test_metrics = test_model(model, test_loader, device)
#             logging.info(f"{args.model_type} Test Metrics: {test_metrics}")
            
#             results = {
#                 'model_type': args.model_type,
#                 'phase': 'test',
#                 'data_dir': args.test_data_dir,
#                 **test_metrics
#             }
#             log_results(results)


#     logging.info("Completed the training, testing, fine-tuning, and explanation process.")

# if __name__ == '__main__':
#     main()

import joblib
import torch
import logging
import argparse
import os
import numpy as np
from torch.optim import Adam
import csv

from model import CNN_BiLSTM_Attention_Model, BiLSTM, get_baseline_models
from train import save_model, train_model, apply_pruning, load_model, fine_tune_model, train_baseline_model,evaluate_model
from explain import explain_with_shap
from data_preprocessing import get_dataloaders, load_data_from_directory, preprocess_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def log_results(results, log_file='model_performance_log.csv'):
    file_exists = os.path.isfile(log_file)
    headers = results.keys()
    with open(log_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()  # Write header if the file is new
        writer.writerow(results)

def initialize_model(model_type, input_dim, output_dim, device):
    if model_type == 'CNN_BiLSTM_Attention':
        return CNN_BiLSTM_Attention_Model(input_dim=input_dim, cnn_out_channels=64, 
                                          lstm_hidden_dim=64, lstm_layers=2, output_dim=output_dim).to(device)
    elif model_type == 'BiLSTM':
        return BiLSTM(input_dim=input_dim, hidden_dim=64, output_dim=output_dim).to(device)
    else:
        models = get_baseline_models(input_dim, output_dim)
        return models[model_type]

def main():
    parser = argparse.ArgumentParser(description='Train, Test, Fine-tune, and Explain Models with Various Architectures.')
    parser.add_argument('--model_type', type=str, default='CNN_BiLSTM_Attention', 
                        choices=['CNN_BiLSTM_Attention', 'BiLSTM', 'RandomForest', 'XGBoost'], 
                        help='Specify the model type to use: CNN_BiLSTM_Attention, BiLSTM, RandomForest, or XGBoost.')
    parser.add_argument('--train', action='store_true', help='Flag to indicate whether to train the model.')
    parser.add_argument('--fine_tune', action='store_true', help='Flag to indicate whether to fine-tune the model on new data.')
    parser.add_argument('--test', action='store_true', help='Flag to indicate whether to test the model on a new dataset.')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint.')
    parser.add_argument('--prune', type=float, default=0.3, help='Fraction of weights to prune (0.3 means 30% pruning).')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model in each fold.')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation during training.')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of epochs for fine-tuning.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing.')
    parser.add_argument('--sample_size', type=float, help='Proportion or number of samples to use for fine-tuning (e.g., 0.1 for 10% or 100 for 100 samples).')
    parser.add_argument('--data_dir', type=str, default='./csv_output/CIRA-CIC-DoHBrw-2020', help='Path to training data directory.')
    parser.add_argument('--fine_tune_data_dir', type=str, help='Path to new data directory for fine-tuning.')
    parser.add_argument('--test_data_dir', type=str, help='Path to testing data directory.')
    parser.add_argument('--checkpoint_path', type=str, default='model_checkpoint.pth', help='Path to save/load the model checkpoint.')
    parser.add_argument('--best_checkpoint_path', type=str, default='best_model_checkpoint.pth', help='Path to save/load the best model checkpoint.')

    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename='logs/training.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Started the training, testing, fine-tuning, and explanation process.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess training data
    combined_data = load_data_from_directory(args.data_dir)
    X, y, scaler, label_encoder = preprocess_data(combined_data)
    input_dim = X.shape[1]
    output_dim = len(label_encoder.classes_)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_metrics = []

    # Cross-validation training phase
    if args.train:
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"Fold {fold}/{args.k_folds}")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            train_loader, val_loader = get_dataloaders(X_train, y_train, batch_size=args.batch_size, sample_size=args.sample_size)
            
            model = initialize_model(args.model_type, input_dim, output_dim, device)
            
            if args.model_type in ['CNN_BiLSTM_Attention', 'BiLSTM']:
                checkpoint_path = f"{args.model_type}_fold_{fold}_checkpoint"
                best_checkpoint_path = f"{args.model_type}_best_fold_{fold}.pth"
                
                if args.prune > 0.0:
                    apply_pruning(model, amount=args.prune)

                fold_train_metrics = train_model(
                    model, train_loader, val_loader, device, num_epochs=args.epochs, lr=1e-4,
                    checkpoint_path=checkpoint_path, save_best=True
                )

                if fold_train_metrics:
                    fold_metrics.append(fold_train_metrics)
            else:
                model_file = f"{args.model_type}_fold_{fold}_model.pkl"
                metrics = train_baseline_model(model, args.model_type, X_train, y_train, X_val, y_val)
                fold_metrics.append(metrics)
                joblib.dump(model, model_file)

            print(f"Fold {fold} Metrics: {fold_metrics[-1]}")

        # Average metrics across all folds and log
        valid_fold_metrics = [fold for fold in fold_metrics if fold is not None]
        if valid_fold_metrics:
            averaged_metrics = {metric: np.mean([fold[metric] for fold in valid_fold_metrics]) for metric in valid_fold_metrics[0]}
            logging.info(f"Averaged Cross-Validation Metrics: {averaged_metrics}")
            results = {'model_type': args.model_type, 'phase': 'cross_validation_train', 'data_dir': args.data_dir, **averaged_metrics}
            log_results(results)
        else:
            raise ValueError("No valid metrics were collected across folds.")

    # Fine-tuning phase
    if args.fine_tune:
        fine_tune_data = load_data_from_directory(args.fine_tune_data_dir)
        X_fine_tune, y_fine_tune, _, _ = preprocess_data(fine_tune_data)
        
        if args.model_type in ['CNN_BiLSTM_Attention', 'BiLSTM']:
            model = initialize_model(args.model_type, input_dim, output_dim, device)
            fine_tune_train_loader, fine_tune_val_loader = get_dataloaders(X_fine_tune, y_fine_tune, batch_size=args.batch_size, sample_size=args.sample_size)
            
            fine_tune_checkpoint = f"{args.model_type}_fine_tuned_checkpoint.pth"
            best_checkpoint_path = f"{args.model_type}_best.pth"
            
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
            fine_tune_metrics = fine_tune_model(
                model, fine_tune_train_loader, fine_tune_val_loader, device, optimizer=optimizer,
                num_epochs=args.fine_tune_epochs, pretrained_path=args.best_checkpoint_path
            )

            save_model(model, optimizer=None, epoch=args.fine_tune_epochs, file_path=fine_tune_checkpoint)
            results = {'model_type': args.model_type, 'phase': 'fine_tune', 'data_dir': args.fine_tune_data_dir, **fine_tune_metrics}
            log_results(results)

        else:
            model = joblib.load(f"{args.model_type}_model.pkl")
            model.fit(X_fine_tune, y_fine_tune)  # Fine-tune with new data
            fine_tune_metrics = train_baseline_model(model, args.model_type, X_fine_tune, y_fine_tune, X_fine_tune, y_fine_tune, False)
            joblib.dump(model, f"{args.model_type}_fine_tuned_model.pkl")
            results = {'model_type': args.model_type, 'phase': 'fine_tune', 'data_dir': args.fine_tune_data_dir, **fine_tune_metrics}
            log_results(results)

    # Testing phase
    if args.test:
        test_combined_data = load_data_from_directory(args.test_data_dir)
        X_test, y_test, _, _ = preprocess_data(test_combined_data)
        
        if args.model_type in ['RandomForest', 'XGBoost']:
            model = joblib.load(f'{args.model_type}_fine_tuned_model.pkl' if args.fine_tune else f'{args.model_type}_model.pkl')
            y_test_pred = model.predict(X_test)
            y_test_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            if y_test_pred_proba is not None:
                if y_test_pred_proba.shape[1] == 2:  
                    test_auc = roc_auc_score(y_test, y_test_pred_proba[:, 1])  
                else:  
                    test_auc = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr')
            else:
                test_auc = 'N/A'
            
            results = {
                'model_type': args.model_type,
                'phase': 'test',
                'data_dir': args.test_data_dir,
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1,
                'auc': test_auc,
            }
            log_results(results)
            logging.info(f"{args.model_type} Test Metrics - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}, AUC: {test_auc}")
        
        else:
            test_loader = get_dataloaders(X_test, y_test, batch_size=args.batch_size)[0]
            model = initialize_model(args.model_type, input_dim, output_dim, device)
            model, _, _ = load_model(model, file_path=args.best_checkpoint_path)
            test_metrics = evaluate_model(model, test_loader, device)
            results = {'model_type': args.model_type, 'phase': 'test', 'data_dir': args.test_data_dir, **test_metrics}
            log_results(results)
            logging.info(f"{args.model_type} Test Metrics: {test_metrics}")

    logging.info("Completed the training, testing, fine-tuning, and explanation process.")

if __name__ == '__main__':
    main()
