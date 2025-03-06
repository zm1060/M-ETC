import joblib
import torch
import logging
import argparse
import os
import numpy as np
from torch.optim import Adam
from train import calculate_metrics, train_model, apply_pruning, load_model, test_model, fine_tune_model, train_baseline_model
from explain import explain_with_shap
from data_preprocessing import get_dataloaders, get_fine_tune_loader, get_test_loader, load_data_from_directory, preprocess_data, sample_data

from utils import create_fine_tune_splits, initialize_model, log_results

def train_phase(args, model, input_dim, output_dim, X_train, y_train, train_val_indices, device):
    best_val_loss = float('inf')
    fold_metrics = []
    aggregated_confusion_matrix = None
    total_val_samples = 0
    
    for fold, (train_idx, val_idx) in enumerate(train_val_indices, 1):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        total_val_samples += len(val_idx)
        logging.info(f"Fold {fold} - Validation set size: {len(val_idx)}")
        
        # Create train loader without splitting
        train_loader = get_dataloaders(X_train_fold, y_train_fold, batch_size=args.batch_size, sample_size=args.sample_size)
        # Create validation loader without splitting
        val_loader = get_dataloaders(X_val_fold, y_val_fold, batch_size=args.batch_size)

        if args.prune > 0.0 and args.model_type not in ['RandomForest', 'XGBoost', 'LogisticRegression', 'AdaBoost', 'DecisionTree', 'NaiveBayes', 'LDA', 'ExtraTrees', 'CatBoost', 'LightGBM', 'Transformer']:
            apply_pruning(model, amount=args.prune, structured=True, global_prune=False)

        dl_checkpoint_path = f"{args.model_type}_fold_{fold}_checkpoint.pth"
        ml_checkpoint_path = f"{args.model_type}_fold_{fold}_checkpoint.pkl"
        
        if args.model_type in ['RandomForest', 'XGBoost', 'LogisticRegression', 'AdaBoost', 'DecisionTree', 'NaiveBayes', 'LDA', 'ExtraTrees', 'CatBoost', 'LightGBM', 'Transformer']:
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
            
            # Verify confusion matrix samples match validation set size
            fold_cm_samples = np.sum(fold_cm)
            if fold_cm_samples != len(val_idx):
                logging.warning(f"Fold {fold} - Confusion matrix samples ({fold_cm_samples}) "
                              f"doesn't match validation set size ({len(val_idx)})")
            
            # Aggregate confusion matrices
            if aggregated_confusion_matrix is None:
                aggregated_confusion_matrix = fold_cm
            else:
                aggregated_confusion_matrix += fold_cm
                
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
        # Calculate average metrics across folds
        avg_metrics = {
            'train': {metric: np.mean([fold['train'][metric] for fold in fold_metrics]) for metric in fold_metrics[0]['train']},
            'val': {metric: np.mean([fold['val'][metric] for fold in fold_metrics]) for metric in fold_metrics[0]['val']}
        }
        
        # Verify final confusion matrix total matches total validation samples
        final_cm_samples = np.sum(aggregated_confusion_matrix)
        if final_cm_samples != total_val_samples:
            logging.warning(f"Final confusion matrix samples ({final_cm_samples}) doesn't match "
                          f"total validation samples ({total_val_samples})")
        logging.info(f"Final confusion matrix total samples: {total_val_samples}")
        logging.info(f"Expected total validation samples: {total_val_samples}")
        
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
    fine_tune_splits = create_fine_tune_splits(args.use_exist, X_fine_tune, y_fine_tune, test_size=0.8)
    fine_tune_train_indices = fine_tune_splits['fine_tune_train_indices']  # 0.8
    fine_tune_val_indices = fine_tune_splits['fine_tune_val_indices']      # 0.2
    X_fine_train, y_fine_train = X_fine_tune[fine_tune_train_indices], y_fine_tune[fine_tune_train_indices]
    X_fine_val, y_fine_val = X_fine_tune[fine_tune_val_indices], y_fine_tune[fine_tune_val_indices]

    if args.model_type in ['RandomForest', 'XGBoost', 'LogisticRegression',    'AdaBoost', 'DecisionTree', 'NaiveBayes', 'LDA', 'ExtraTrees', 'CatBoost', 'LightGBM', 'Transformer']:
        X_fine_train, y_fine_train = sample_data(X_fine_train, y_fine_train, args.sample_size)
        logging.info(f"Sampled data for RandomForest/XGBoost: {len(X_fine_train)} samples.")
        model_file = args.best_checkpoint_path
        
        if args.model_type == 'XGBoost':
            import xgboost as xgb
            num_classes = len(np.unique(y_fine_train))
            logging.info(f"Number of classes for fine-tuning: {num_classes}")
            
            # Load the model parameters
            old_model = joblib.load(model_file)
            params = old_model.get_params()
            
            # Update parameters for multi-class classification
            if num_classes > 2:
                params.update({
                    'objective': 'multi:softprob',
                    'num_class': num_classes
                })
            else:
                params.update({
                    'objective': 'binary:logistic'
                })
            
            # Create a new model with the updated parameters
            fine_tuned_model = xgb.XGBClassifier(**params)
        elif args.model_type == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            num_classes = len(np.unique(y_fine_train))
            
            # Load the old model to get its parameters
            old_model = joblib.load(model_file)
            params = old_model.get_params()
            
            # Update multi_class parameter based on number of classes
            if num_classes > 2:
                params['multi_class'] = 'multinomial'
            else:
                params['multi_class'] = 'ovr'
                
            # Remove any deprecated parameters
            params.pop('multi_class_old', None)
            params.pop('multi_class_deprecated', None)
            
            # Create a new model with the updated parameters
            fine_tuned_model = LogisticRegression(**params)
        else:
            fine_tuned_model = joblib.load(model_file)
        
        fine_tuned_model.fit(X_fine_train, y_fine_train)   # fine-tune
        y_fine_pred = fine_tuned_model.predict(X_fine_val) # test
        y_fine_pred_proba = fine_tuned_model.predict_proba(X_fine_val) if hasattr(fine_tuned_model, "predict_proba") else None
        fine_tune_metrics = calculate_metrics(y_fine_val, y_fine_pred, y_fine_pred_proba)
        joblib.dump(fine_tuned_model, f"{args.model_type}_fine_tuned_model.pkl")
    else:
        fine_tune_train_loader, fine_tune_val_loader = get_fine_tune_loader(
            X_fine_train, y_fine_train,
            X_fine_val, y_fine_val,
            sample_size=args.sample_size,
            batch_size=args.batch_size
        )
        logging.info(f"Sampled data for fine-tuning: {len(fine_tune_train_loader.dataset)} samples.")
        # optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        # fine_tune_metrics = fine_tune_model(model, fine_tune_train_loader, fine_tune_val_loader, device, optimizer=optimizer, num_epochs=args.fine_tune_epochs, pretrained_path=args.best_checkpoint_path)
        fine_tune_metrics = fine_tune_model(
            model=model,
            model_type=args.model_type,
            train_loader=fine_tune_train_loader,
            val_loader=fine_tune_val_loader,
            device=device,
            num_epochs=args.fine_tune_epochs,
            lr=1e-4,
            optimizer_type="adam",
            momentum=0.9,
            weight_decay=0.01,
            checkpoint_path=args.best_checkpoint_path,
            resume=False,
            save_best=True,
            save_current=True
        )
        torch.save(model.state_dict(), f"{args.model_type}_fine_tuned_model.pth")
    logging.info(f"{args.model_type} Fine-Tune Metrics: {fine_tune_metrics}")

    log_results({'model_type': args.model_type, 'phase': 'fine_tune', 'data_dir': args.fine_tune_data_dir, **fine_tune_metrics})

def test_phase(args, model, device):
    logging.info(f"Test data directory: {args.test_data_dir}")
    test_data = load_data_from_directory(args.test_data_dir)
    X_test, y_test, _, _ = preprocess_data(test_data)
    if args.model_type in ['RandomForest', 'XGBoost', 'LogisticRegression',    'AdaBoost', 'DecisionTree', 'NaiveBayes', 'LDA', 'ExtraTrees', 'CatBoost', 'LightGBM']:
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
    
    if args.model_type in ['RandomForest', 'XGBoost', 'LogisticRegression',    'AdaBoost', 'DecisionTree', 'NaiveBayes', 'LDA', 'ExtraTrees', 'CatBoost', 'LightGBM']:
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
    
    # Create output directory if it doesn't exist
    os.makedirs("shap_outputs", exist_ok=True)
    
    # Remove csv_output/ from data dir path
    db_name = args.explain_data_dir.replace('../csv_output/', '')
    
    if len(X_misclassified) > 0:
        explain_with_shap(model, X_sample=X_misclassified, device=device, feature_names=feature_names, class_names=class_names, origin_model_type=args.model_type, db_name=db_name, description='misclassified')
        logging.info("SHAP explanations generated for misclassified samples.")

    if len(X_correct) > 0:
        explain_with_shap(model, X_sample=X_correct, device=device, feature_names=feature_names, class_names=class_names, origin_model_type=args.model_type, db_name=db_name, description='correct')
        logging.info("SHAP explanations generated for correctly classified samples.")

