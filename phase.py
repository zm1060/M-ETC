import joblib
import torch
import logging
import argparse
import os
import numpy as np
from torch.optim import Adam
from train import calculate_metrics, train_model, apply_pruning, load_model, test_model, fine_tune_model, train_baseline_model
from explain import explain_with_shap
from data_preprocessing import get_dataloaders, get_test_loader, load_data_from_directory, preprocess_data
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils import create_data_splits, create_fine_tune_splits, hyperparameter_search, initialize_model, log_results, set_seed

def train_phase(args, model, input_dim, output_dim, X_train, y_train, train_val_indices, device):
    best_val_loss = float('inf')
    fold_metrics = []
    aggregated_confusion_matrix = None
    for fold, (train_idx, val_idx) in enumerate(train_val_indices, 1):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        train_loader, val_loader = get_dataloaders(X_train_fold, y_train_fold, batch_size=args.batch_size, sample_size=args.sample_size)
        
        if args.prune > 0.0 and args.model_type not in ['RandomForest', 'XGBoost']:
            apply_pruning(model, amount=args.prune)
        
        dl_checkpoint_path = f"{args.model_type}_fold_{fold}_checkpoint.pth"
        ml_checkpoint_path = f"{args.model_type}_fold_{fold}_checkpoint.pkl"
        
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
        
        if metrics:
            fold_metrics.append(metrics)
            fold_cm = np.array(metrics['val']['confusion_matrix'])
            aggregated_confusion_matrix = fold_cm if aggregated_confusion_matrix is None else aggregated_confusion_matrix + fold_cm
        model = initialize_model(
            model_type=args.model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            cnn_out_channels=args.cnn_out_channels,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers
        )
    
    if fold_metrics:
        avg_metrics = {
            'train': {metric: np.mean([fold['train'][metric] for fold in fold_metrics]) for metric in fold_metrics[0]['train']},
            'val': {metric: np.mean([fold['val'][metric] for fold in fold_metrics]) for metric in fold_metrics[0]['val']}
        }
        results = {
            'model_type': args.model_type,
            'phase': 'train',
            'data_dir': args.data_dir,
            **{f'train_{k}': v for k, v in avg_metrics['train'].items()},
            **{f'val_{k}': v for k, v in avg_metrics['val'].items()},
            'aggregated_confusion_matrix': aggregated_confusion_matrix.tolist(),
        }
        log_results(results)

def fine_tune_phase(args, model, device):
    logging.info(f"Fine-tuning data directory: {args.fine_tune_data_dir}")
    fine_tune_data = load_data_from_directory(args.fine_tune_data_dir)
    X_fine_tune, y_fine_tune, _, _ = preprocess_data(fine_tune_data)
    fine_tune_splits = create_fine_tune_splits(args.use_exist, X_fine_tune, y_fine_tune)
    fine_tune_train_indices = fine_tune_splits['fine_tune_train_indices']
    fine_tune_val_indices = fine_tune_splits['fine_tune_val_indices']
    X_fine_train, y_fine_train = X_fine_tune[fine_tune_train_indices], y_fine_tune[fine_tune_train_indices]
    X_fine_val, y_fine_val = X_fine_tune[fine_tune_val_indices], y_fine_tune[fine_tune_val_indices]
    fine_tune_train_loader, fine_tune_val_loader = get_dataloaders(X_fine_train, y_fine_train, batch_size=args.batch_size, sample_size=args.sample_size)
    
    if args.model_type in ['RandomForest', 'XGBoost']:
        model_file = args.best_checkpoint_path
        fine_tuned_model = joblib.load(model_file) if os.path.exists(model_file) else RandomForestClassifier(n_estimators=100, random_state=42)
        fine_tuned_model.fit(X_fine_train, y_fine_train)
        y_fine_pred = fine_tuned_model.predict(X_fine_val)
        y_fine_pred_proba = fine_tuned_model.predict_proba(X_fine_val) if hasattr(fine_tuned_model, "predict_proba") else None
        fine_tune_metrics = calculate_metrics(y_fine_val, y_fine_pred, y_fine_pred_proba)
        joblib.dump(fine_tuned_model, f"{args.model_type}_fine_tuned_model.pkl")
    else:
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        fine_tune_metrics = fine_tune_model(model, fine_tune_train_loader, fine_tune_val_loader, device, optimizer=optimizer, num_epochs=args.fine_tune_epochs, pretrained_path=args.best_checkpoint_path)
        torch.save(model.state_dict(), f"{args.model_type}_fine_tuned_model.pth")
    
    log_results({'model_type': args.model_type, 'phase': 'fine_tune', 'data_dir': args.fine_tune_data_dir, **fine_tune_metrics})

def test_phase(args, model, device):
    logging.info(f"Test data directory: {args.test_data_dir}")
    test_data = load_data_from_directory(args.test_data_dir)
    X_test, y_test, _, _ = preprocess_data(test_data)
    if args.model_type in ['RandomForest', 'XGBoost']:
        model = joblib.load(args.test_checkpoint_path)
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
    else:
        test_loader = get_test_loader(X_test, y_test, batch_size=args.batch_size)
        model, _, _ = load_model(model, file_path=args.test_checkpoint_path)
        metrics = test_model(model, test_loader, device)
    
    logging.info(f"{args.model_type} Test Metrics: {metrics}")
    log_results({'model_type': args.model_type, 'phase': 'test', 'data_dir': args.test_data_dir, **metrics})

def explain_phase(args, model, device):
    features_to_remove = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp']
    explain_data = load_data_from_directory(args.explain_data_dir)
    explain_data = explain_data.drop(columns=features_to_remove, errors='ignore')
    X_explain, y_explain, _, label_encoder = preprocess_data(explain_data)
    feature_names = explain_data.columns[:-1].tolist()
    class_names = label_encoder.classes_.tolist() if label_encoder else [f'Class_{i}' for i in range(len(np.unique(y_explain)))]
    
    if args.model_type in ['RandomForest', 'XGBoost']:
        model = joblib.load(args.explain_checkpoint_path)
        y_pred = model.predict(X_explain)
    else:
        model.eval()
        y_pred_probs = []
        batch_size = 1024
        for i in range(0, len(X_explain), batch_size):
            batch = torch.tensor(X_explain[i:i+batch_size], dtype=torch.float32).to(device)
            with torch.no_grad():
                y_pred_probs.append(model(batch).cpu())
        y_pred_probs = torch.cat(y_pred_probs)
        y_pred = torch.argmax(y_pred_probs, dim=1).cpu().numpy()
    
    misclassified_indices = np.where(y_pred != y_explain)[0][:100]
    correctly_classified_indices = np.where(y_pred == y_explain)[0][:100]
    logging.info(f"Number of misclassified samples: {len(misclassified_indices)}")
    logging.info(f"Number of correctly classified samples: {len(correctly_classified_indices)}")

    X_misclassified = X_explain[misclassified_indices]
    X_correct = X_explain[correctly_classified_indices]
    
    if len(X_misclassified) > 0:
        explain_with_shap(model, X_sample=X_misclassified, device=device, feature_names=feature_names, class_names=class_names, origin_model_type=args.model_type)
        logging.info("SHAP explanations generated for misclassified samples.")

    if len(X_correct) > 0:
        explain_with_shap(model, X_sample=X_correct, device=device, feature_names=feature_names, class_names=class_names, origin_model_type=args.model_type)
        logging.info("SHAP explanations generated for correctly classified samples.")