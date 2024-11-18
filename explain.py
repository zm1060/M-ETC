import os
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
def explain_with_shap(model, X_sample, device, feature_names, class_names, model_type='torch', db_name='dataset', description='explanation'):
    """
    Explain model predictions using SHAP and save plots with dynamic filenames.

    Args:
        model: The model to explain.
        X_sample (np.ndarray): A sample of the input data to explain.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') for PyTorch models.
        feature_names (list): List of feature names for visualization.
        class_names (list): List of class names for visualization.
        model_type (str): Type of the model (e.g., 'CNN_BiLSTM', 'RandomForest').
        db_name (str): Name of the database (e.g., 'CIRA-CIC-DoHBrw-2020').
        description (str): Description for the saved plots (e.g., 'misclassified_samples').
    """
    # Ensure the model is in evaluation mode
    if model_type == 'torch':
        model.eval()

    # Convert feature names to list if not already
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    # Wrapper function for PyTorch models
    def model_predict(X):
        if model_type == 'torch':
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(X_tensor)
            return output.cpu().numpy()
        else:
            return model.predict_proba(X)

    # Choose the correct SHAP explainer
    if model_type == 'torch':
        explainer = shap.KernelExplainer(model_predict, X_sample[:100])  # Use a subset as background
    elif model_type in ['tree', 'xgboost']:
        explainer = shap.TreeExplainer(model)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Debugging info
    print("Feature Names:", feature_names)
    print("SHAP Values Shape:", np.array(shap_values).shape)

    output_dir = "shap_outputs"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    if isinstance(shap_values, list) or (isinstance(shap_values, np.ndarray) and shap_values.ndim == 3): 
        # Multi-class case
        for i, class_name in enumerate(class_names):
            print(f"Generating SHAP summary plot for class: {class_name}")
            shap_values_class = shap_values[i] if isinstance(shap_values, list) else shap_values[:, :, i]
            
            # Summary plot
            shap.summary_plot(
                shap_values_class,
                X_sample,
                feature_names=feature_names,
                show=False
            )
            plt.title(f"SHAP Summary Plot for {class_name}")
            # Save as 300 DPI image
            filename = os.path.join(output_dir, f"{model_type}_{db_name}_{description}_{class_name}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved SHAP summary plot to {filename}")
            plt.close()

    else: 
        # Single-class case
        print("SHAP values for single-output model")
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title("SHAP Summary Plot")
        # Save as 300 DPI image
        filename = os.path.join(output_dir, f"{model_type}_{db_name}_{description}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved SHAP summary plot to {filename}")
        plt.close()

    # Save SHAP values for debugging or reuse
    explanations_path = os.path.join(output_dir, f"{model_type}_{db_name}_{description}_shap_values.npy")
    np.save(explanations_path, shap_values)
    print(f"SHAP explanations saved to {explanations_path}")