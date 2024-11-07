import torch
import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_with_shap(model, X_sample, device, feature_names, class_names):
    model.eval()  # Switch to evaluation mode

    # Convert the X_sample to a torch tensor
    X_sample_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)

    # Define a wrapper function to pass numpy data to the model
    def model_predict(X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # Convert numpy to torch tensor
        with torch.no_grad():  # Disable gradients for faster computation
            output = model(X_tensor)  # Model prediction
        return output.cpu().numpy()  # Convert the output back to numpy

    # Create SHAP Kernel Explainer using the wrapper function
    explainer = shap.KernelExplainer(model_predict, X_sample_tensor.cpu().numpy())
    
    # Calculate SHAP values for the sample data
    shap_values = explainer.shap_values(X_sample)

    # Print the shape of SHAP values
    print(f"Shape of shap_values: {shap_values[0].shape}")  # Should be [n_samples, n_features, n_classes]

    # Calculate mean SHAP values for each feature across all samples for each class
    shap_values_class_0 = shap_values[:, :, 0].mean(axis=0)  # Mean SHAP values for class 0
    shap_values_class_1 = shap_values[:, :, 1].mean(axis=0)  # Mean SHAP values for class 1

    # Calculate standard deviation (to show variability across samples)
    shap_std_class_0 = shap_values[:, :, 0].std(axis=0)  # Std for class 0
    shap_std_class_1 = shap_values[:, :, 1].std(axis=0)  # Std for class 1

    # Create bar plot comparing both classes for each feature, with standard deviation as error bars
    x = np.arange(len(feature_names))  # Feature indices
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot SHAP values for class 0 with error bars (std deviation)
    ax.bar(x - width/2, shap_values_class_0, width, yerr=shap_std_class_0, capsize=5, label=f"Class {class_names[0]}")
    
    # Plot SHAP values for class 1 with error bars (std deviation)
    ax.bar(x + width/2, shap_values_class_1, width, yerr=shap_std_class_1, capsize=5, label=f"Class {class_names[1]}")

    # Add labels, title, and legend
    ax.set_xlabel('Features')
    ax.set_ylabel('Mean SHAP value')
    ax.set_title('Feature importance with SHAP values (with error bars)')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Now plot a beeswarm plot for all the SHAP values (class 0)
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values[:, :, 0], X_sample, feature_names=feature_names, plot_type="dot", show=False)
    plt.title(f"SHAP Summary Plot for {class_names[0]}")
    plt.show()

    # Beeswarm plot for class 1
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values[:, :, 1], X_sample, feature_names=feature_names, plot_type="dot", show=False)
    plt.title(f"SHAP Summary Plot for {class_names[1]}")
    plt.show()

# Example usage
# explain_with_shap(model, X_sample, device, feature_names, ['Class 0', 'Class 1'])
