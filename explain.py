import matplotlib.pyplot as plt
import numpy as np
import os
import shap
import torch

def plot_feature_importance(shap_values, feature_names, description, output_dir, class_name=None):
    """
    Plot feature importance based on mean absolute SHAP values.
    """
    plt.rc('font', family='serif', size=12)  # Use serif font for scientific style

    # Calculate mean absolute SHAP values for each feature
    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_indices = np.argsort(shap_mean_abs)[::-1]  # Sort in descending order
    sorted_shap_mean_abs = shap_mean_abs[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_feature_names[:20], sorted_shap_mean_abs[:20], align='center')
    plt.gca().invert_yaxis()  # Invert y-axis to show the most important features at the top
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Features")
    title = "Feature Importance (SHAP)"
    if class_name:
        title += f" for Class: {class_name}"
    plt.title(title)

    # Save the figure
    filename = f"{description}_feature_importance"
    if class_name:
        filename += f"_{class_name}"
    filename += ".png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {filepath}")
    plt.close()

def explain_with_shap(model, X_sample, device, feature_names, class_names, origin_model_type, model_type='torch', db_name='dataset', description='explanation'):
    """
    Explain model predictions using SHAP and save plots with dynamic filenames, including feature importance.
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

    output_dir = "shap_outputs"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    if isinstance(shap_values, list) or (isinstance(shap_values, np.ndarray) and shap_values.ndim == 3): 
        # Multi-class case
        for i, class_name in enumerate(class_names):
            shap_values_class = shap_values[i] if isinstance(shap_values, list) else shap_values[:, :, i]
            
            # Generate SHAP summary plot
            print(f"Generating SHAP summary plot for class: {class_name}")
            shap.summary_plot(
                shap_values_class,
                X_sample,
                feature_names=feature_names,
                show=False
            )
            plt.title(f"SHAP Summary Plot for {class_name}")
            # Save as 300 DPI image
            summary_filename = os.path.join(output_dir, f"{origin_model_type}_{db_name}_{description}_{class_name}.png")
            plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
            print(f"Saved SHAP summary plot to {summary_filename}")
            plt.close()

            # Generate feature importance plot
            print(f"Generating feature importance plot for class: {class_name}")
            plot_feature_importance(
                shap_values_class,
                feature_names,
                description=f"{origin_model_type}_{db_name}_{description}",
                output_dir=output_dir,
                class_name=class_name
            )

    else: 
        # Single-class case
        print("Generating SHAP summary plot for single-output model")
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title("SHAP Summary Plot")
        summary_filename = os.path.join(output_dir, f"{origin_model_type}_{db_name}_{description}.png")
        plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
        print(f"Saved SHAP summary plot to {summary_filename}")
        plt.close()

        # Generate feature importance plot
        print("Generating feature importance plot for single-output model")
        plot_feature_importance(
            shap_values,
            feature_names,
            description=f"{origin_model_type}_{db_name}_{description}",
            output_dir=output_dir
        )

    # Save SHAP values for debugging or reuse
    explanations_path = os.path.join(output_dir, f"{origin_model_type}_{db_name}_{description}_shap_values.npy")
    np.save(explanations_path, shap_values)
    print(f"SHAP explanations saved to {explanations_path}")
